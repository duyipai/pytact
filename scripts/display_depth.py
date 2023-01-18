import argparse
import os
import cv2
import numpy as np
import pytact
import torch

parser = argparse.ArgumentParser(description="Display raw sensor images")
parser.add_argument(
    "sensor",
    type=str,
    choices=pytact.sensors.get_sensor_names(),
    help="Sensor type to display",
)
parser.add_argument(
    "model_path",
    type=str,
    help="Path for saved model, with same name for both .npz and .pth",
)
parser.add_argument(
    "--url",
    type=str,
    dest="url",
    default=None,
    help="Location of sensor stream (for Gelsight)",
)
parser.add_argument(
    "--serial",
    type=str,
    dest="serial",
    default=None,
    help="Serial number of sensor (for Digit)",
)
parser.add_argument(
    "--input_path",
    type=str,
    dest="input_path",
    default=None,
    help="Input image path (for image input)",
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
    "--device",
    type=str,
    choices=["cuda", "cpu"],
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
)
parser.add_argument("--scale", type=float, dest="scale", default=1.0)
parser.add_argument("--mmpp", type=float, dest="mmpp", default=0.0487334006)
args = parser.parse_args()
sensor = pytact.sensors.sensor_from_args(args.sensor, **vars(args))
npzfile = np.load(args.model_path + ".npz")
mean = npzfile["mean"]
std = npzfile["std"]
model = pytact.models.Pixel2GradModel(
    hidden_size=64, dropout_p=0.1, activation=torch.nn.ReLU(inplace=True)
).to(args.device)
model.load_state_dict(torch.load(args.model_path + ".pth"))
lookupTable = pytact.tasks.DepthFromLookup(
    model=model,
    mmpp=args.mmpp,
    scale=args.scale,
    mean=mean,
    std=std,
    use_cuda=args.device == "cuda",
)
diff_max = 255
depth_range = 0.004
depth_bias = 0.001
if args.input_path is None:  # real-time sensor input
    if sensor.is_running():
        for i in range(100):  # skip first 100 frames to give sensor time to warm up
            _ = sensor.get_frame()
        sensor.set_reference(sensor.get_frame())
    print("Finished setting reference frame")
    while sensor.is_running():
        frame = sensor.get_frame()
        if frame is not None:
            cv2.imshow("display", frame.image)
            cv2.imshow("reference", sensor.get_reference().image)
            diff = sensor.preprocess_for(lookupTable.model.model_type, frame)
            cv2.imshow("diff", (np.abs(diff.image) / diff_max * 255.0).astype(np.uint8))
            depth = lookupTable(diff)
            # print("diff min/max:", diff.image.min(), diff.image.max())
            # print("Depth min/max:", depth.data.min(), depth.data.max())
            cv2.imshow(
                "depth",
                ((depth_bias - depth.data) / depth_range * 255.0).astype(np.uint8),
            )
        if cv2.waitKey(2) == ord("q"):
            break
    cv2.destroyAllWindows()
else:  # image input
    imgs = [
        args.input_path + "/" + f
        for f in sorted(os.listdir(args.input_path))
        if os.path.isfile(os.path.join(args.input_path, f))
    ]
    i = 0
    while True:
        frame = pytact.types.Frame(pytact.types.FrameEnc.BGR, cv2.imread(imgs[i]))
        cv2.imshow("display", frame.image)
        diff = sensor.preprocess_for(lookupTable.model_type, frame)
        depth = lookupTable(diff)
        # print("Depth min/max:", depth.data.min(), depth.data.max())
        cv2.imshow(
            "depth",
            ((depth_bias - depth.data) / depth_range * 255.0).astype(np.uint8),
        )
        if cv2.waitKey(1) == ord("n"):
            i = i + 1
            if i >= len(imgs):
                break
            else:
                continue
        elif cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()
sensor.stop()
