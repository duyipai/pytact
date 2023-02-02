#!/usr/bin/env python3

import argparse
import glob
import math
import os
import random
from csv import writer
from datetime import datetime as dt

import cv2
import numpy as np
import pytact


def floatName(filepath):
    filename = filepath.split("/")[-1]
    purefilename = int(filename.split(".")[0])
    return purefilename


FIELD_NAMES = ["img_name", "R", "G", "B", "x", "y", "gx", "gy"]

# Parse CLI arguments
parser = argparse.ArgumentParser(
    description="""
Utility to quickly create a pixel -> gradient dataset. This script reads from the 
folder of provided images and displays their preprocessed form accoording to the
selected sensor. On each image, an estimate of the circle deformation is labeled.
If this label is incorrect, you can redraw the label by clicking and dragging.
If you want to ignore this image, press 'n'. If you are happy with the label, you
can add the image by pressing 'y'. To finish, press 'q'.
Gradients are estimated using the provided parameters about the sphere's actual radius.
"""
)
parser.add_argument(
    "--input_path",
    type=str,
    default=os.path.join(os.getcwd(), "data", "pan_pan_1"),
    help="Path to read images from",
)
parser.add_argument(
    "--ball_radius",
    type=float,
    default=1.0,
    help="Radius of ball used in data collection (mm)",
)
parser.add_argument(
    "--mmpp", type=float, default=0.0487334006, help="Measure of mm per pixel"
)
parser.add_argument(
    "--sensor",
    type=str,
    choices=pytact.sensors.get_sensor_names(),
    default="GelsightR15",
    help="Sensor that images were collected from",
)
parser.add_argument(
    "--output_path",
    type=str,
    dest="output",
    default=os.path.join(os.getcwd(), "data"),
    help="Path to save CSV dataset to",
)
parser.add_argument(
    "--amt-empty",
    type=float,
    dest="amt_empty",
    default=0.03,
    help="Amount of empty data points to include in dataset",
)
args = parser.parse_args()

sensor = pytact.sensors.sensor_from_args(args.sensor, **vars(args))

# Store CLI args
radius = args.ball_radius / 1000.0
mpp = args.mmpp / 1000.0

# Setup dataset file
output_file = args.output + f"/data-{dt.now().strftime('%H-%M-%S')}.csv"
with open(output_file, "w") as f:
    w = writer(f)
    w.writerow(FIELD_NAMES)

# Retrieve stored images
imgs = glob.glob(args.input_path + "/*.jpg")
imgs.sort(key=lambda x: floatName(x))
imgs = [f for f in imgs]
print(imgs)
# Callback variables
current_frame = None
circle = None
click_start = None


def click_cb(event, x, y, _a, _b):
    global current_frame, circle, click_start
    if event == cv2.EVENT_LBUTTONDOWN:
        click_start = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        x_len = click_start[0] - x
        y_len = click_start[1] - y
        circle = (
            int(x_len / 2 + x),
            int(y_len / 2 + y),
            int(math.sqrt(x_len * x_len + y_len * y_len) / 2),
        )

        display_frame = current_frame.image.copy() * 3.0
        display_frame[display_frame < 0.0] *= -1.0
        display_frame[display_frame > 255.0] = 255.0
        display_frame = np.uint8(display_frame)
        cv2.circle(
            display_frame,
            (int(circle[0]), int(circle[1])),
            int(circle[2]),
            (0, 255, 0),
            2,
        )
        cv2.circle(display_frame, (int(circle[0]), int(circle[1])), 2, (0, 0, 255), 3)

        cv2.imshow("label_data", display_frame)


# Configure cv window
cv2.namedWindow("label_data", cv2.WINDOW_GUI_EXPANDED)
cv2.setMouseCallback("label_data", click_cb)
total_pixels = 0
init_markers = None
while len(imgs) > 0:
    # Collect next frame and preprocess using sensor
    img = cv2.imread(imgs[0], cv2.IMREAD_COLOR)
    print("Processing image: " + imgs[0])
    current_frame = pytact.types.Frame(pytact.types.FrameEnc.BGR, img)
    if sensor.has_marker:
        markers, mask = sensor.get_markers_from_frame(current_frame)
        print(
            np.abs(markers.astype(np.uint8) * 255 - mask).astype(np.float32).sum(),
            mask.sum() / 255,
        )
        if init_markers is None:
            init_markers = markers.copy()
        cv2.imwrite(os.path.join("data", "markers.png"), markers.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join("data", "mask.png"), mask)
    current_frame = sensor.preprocess_for(
        pytact.types.ModelType.Pixel2Grad, current_frame
    )

    # Convert to grayscale and find circles using hough transform
    display_image = current_frame.image.copy() * 3.0
    display_image[display_image < 0.0] *= -1.0
    display_image[display_image > 255.0] = 255.0
    display_image = np.uint8(display_image)
    grayscale_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        grayscale_image,
        cv2.HOUGH_GRADIENT,
        1,
        20,
        param1=30,
        param2=30,
        minRadius=5,
        maxRadius=30,
    )
    if circles is not None:
        for circle in circles[0]:
            cv2.circle(
                display_image,
                (int(circle[0]), int(circle[1])),
                int(circle[2]),
                (0, 255, 0),
                2,
            )
            cv2.circle(
                display_image, (int(circle[0]), int(circle[1])), 2, (0, 0, 255), 3
            )
            break  # Only print first

    cv2.imshow("label_data", display_image)

    while True:
        k = cv2.waitKey()
        if k == ord("y"):
            if circle is None:
                print("No circle selected.")
                continue

            # Find distance in meters from circle radius
            x_range = np.arange(current_frame.image.shape[1])
            y_range = np.arange(current_frame.image.shape[0])
            xv, yv = np.meshgrid(x_range, y_range)
            gx = (circle[0] - xv) * mpp
            gy = (circle[1] - yv) * mpp

            # Compute x and y gradients using equation of a sphere
            dist = np.power(gx, 2) + np.power(gy, 2)
            dist_from_im = (circle[2] * mpp) ** 2 - dist
            dist_from_real = radius**2 - dist
            gx = np.where(
                dist_from_im > 0.0, -gx / np.sqrt(np.abs(dist_from_real)), 0.0
            )
            gy = np.where(
                dist_from_im > 0.0, -gy / np.sqrt(np.abs(dist_from_real)), 0.0
            )

            # Turn gradients into dataset labels

            labels = []
            for x in range(current_frame.image.shape[1]):
                for y in range(current_frame.image.shape[0]):
                    # Discard gradients outside of circle
                    if sensor.has_marker:  # with markers
                        if markers[y, x] or init_markers[y, x]:
                            continue
                    if (
                        gx[y, x] == 0.0
                        and gy[y, x] == 0.0
                        and random.random() > args.amt_empty
                    ):
                        continue

                    r = current_frame.image[y, x, 0]
                    g = current_frame.image[y, x, 1]
                    b = current_frame.image[y, x, 2]

                    labels.append((imgs[0], r, g, b, x, y, gx[y, x], gy[y, x]))

            total_pixels += len(labels)

            # Write all labels to CSV file
            with open(output_file, "a", newline="") as f:
                print(f"Writing {len(labels)} labels to {output_file}")
                w = writer(f)
                for label in labels:
                    w.writerow(label)
            break
        elif k == ord("q"):
            cv2.destroyAllWindows()
            exit()
        elif k == ord("n"):
            break

    # Move to next image
    imgs = imgs[1:]

with open(output_file, "a", newline="") as f:
    print(f"Writing {len(labels)} labels to {output_file}")
    w = writer(f)
    for label in labels:
        w.writerow(label)

cv2.destroyAllWindows()
