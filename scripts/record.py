#!/usr/bin/env python3

import argparse
import cv2
import os
import pytact
import time

parser = argparse.ArgumentParser(description="Record a sequence of sensor images")
parser.add_argument(
    "sensor",
    type=str,
    choices=pytact.sensors.get_sensor_names(),
    help="Sensor type to display",
)
parser.add_argument(
    "--url",
    type=str,
    dest="url",
    default=None,
    help="Location of sensor stream (if needed)",
)
parser.add_argument(
    "--serial",
    type=str,
    dest="serial",
    default=None,
    help="Serial number of sensor (if needed)",
)
parser.add_argument(
    "--roi",
    dest="roi",
    nargs=4,
    default=None,
    help="Region of interest in sensor frame, specify in order of top-left, top-right, "
    + "bottom-right, and bottom-left. Format should be as follows: x,y x,y x,y x,y",
)
parser.add_argument(
    "--output",
    type=str,
    dest="output",
    default=os.getcwd(),
    help="Path to save images in",
)
parser.add_argument(
    "--count", type=int, dest="count", default=5000, help="Number of images to save"
)
parser.add_argument(
    "--duration",
    type=int,
    dest="duration",
    default=60,
    help="Number of seconds to record for",
)
args = parser.parse_args()

i = 0
start = time.time()
if not os.path.exists(args.output):
    print(f"Created output directory: {args.output}")
    os.makedirs(args.output)

sensor = pytact.sensors.sensor_from_args(args.sensor, **vars(args))
if args.sensor == "DigitV2":
    sensor.set_fps("30fps")
for i in range(120):
    sensor.get_frame()
while sensor.is_running():
    if time.time() - start > args.duration or i > args.count - 1:
        break

    frame = sensor.get_frame()
    if frame is not None:
        cv2.imwrite(f"{args.output}/{i}.jpg", frame.image)
        cv2.imshow("frame", frame.image)
        cv2.waitKey(1)
        i += 1
sensor.stop()
print(f"Saved {i} photos to {args.output}")
