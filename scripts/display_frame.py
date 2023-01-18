import cv2
import pytact
import argparse

parser = argparse.ArgumentParser(description="Display raw sensor images")
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
args = parser.parse_args()

sensor = pytact.sensors.sensor_from_args(args.sensor, **vars(args))

cv2.namedWindow("raw", cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow("markers", cv2.WINDOW_GUI_EXPANDED)
while sensor.is_running():
    frame = sensor.get_frame()
    if frame is not None:
        cv2.imshow("raw", frame.image)
        if sensor.has_marker:
            markers = sensor.get_markers().markers
            img = frame.image.copy()
            for i in range(markers.shape[0]):
                cv2.circle(
                    img,
                    (int(markers[i][0]), int(markers[i][1])),
                    1,
                    (0, 0, 255),
                    -1,
                )
            cv2.imshow("markers", img)

    if cv2.waitKey(2) == ord("q"):
        sensor.stop()
        break
cv2.destroyAllWindows()
