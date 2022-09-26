from copy import deepcopy
from typing import List, Optional

import cv2
from digit_interface import Digit
from pytact.types import Frame, FrameEnc, ModelType

from .sensors import Sensor, UnsupportedModelError


class DigitV2(Sensor):
    """
    Retrieves images from an http stream and assumes markers are present.

    Parameters
    ----------
    size: Tuple[int, int], optional
        Downscaled resolution of image (default (160, 120))
    encoding: FrameEnc, optional
        Encoding to process frames as (default FrameEnc.BGR)
    serial: str, serial number of digit camera
    """

    _encoding = FrameEnc.BGR

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "serial" in kwargs and kwargs["serial"] is not None:
            self._serial = kwargs["serial"]
            self._sensor = Digit(self._serial)
            self._sensor.connect()
            self._size = [
                self._sensor.resolution["width"],
                self._sensor.resolution["height"],
            ]
            self._sample_rate = self._sensor.fps
            self._resolution_mode = "QVGA"
        else:
            self._sensor = None

        # Start frame sampling
        self.has_marker = False
        self._is_running: bool = True
        self._frame: Optional[Frame] = None
        self._ref: Optional[Frame] = None

    def __del__(self):
        if self._sensor is not None:
            self._sensor.disconnect()

    @property
    def supported_models(self) -> List[ModelType]:
        return [ModelType.Pixel2Grad]

    def set_resolution(self, resolution_mode):
        """resolutin_mode = "QVGA", "VGA" """
        if self._sensor is not None:
            if resolution_mode not in self._sensor.STEAMS:
                raise ValueError(
                    f"Digit: resolution_mode not supported: {resolution_mode}"
                )
            self._resolution_mode = resolution_mode
            self._sensor.set_resolution(self._sensor.STEAMS[resolution_mode])

    def set_fps(self, fps):
        """fps = "15fps", "30fps","60fps" """
        if self._sensor is not None:
            if fps not in self._sensor.STREAMS[self._resolution_mode]["fps"]:
                raise ValueError(f"Digit: fps not supported: {fps}")
            self._sensor.set_fps(
                self._sensor.STREAMS[self._resolution_mode]["fps"][fps]
            )

    def set_reference(self, frame: Frame):
        self._ref = deepcopy(frame)

    def get_frame(self) -> Optional[Frame]:
        """Returns frame collected in the last sample."""
        frame = self._sensor.get_frame()
        frame = cv2.medianBlur(frame, 5)
        self._frame = Frame(self._encoding, frame)
        if self._ref is None:
            self._ref = deepcopy(self._frame)
        return deepcopy(self._frame)

    def get_reference(self) -> Optional[Frame]:
        return deepcopy(self._ref)

    def is_running(self):
        return self.is_running

    def preprocess_for(self, model: ModelType, frame: Frame) -> Frame:
        if model == ModelType.Pixel2Grad:
            if self._ref is None:
                self._ref = deepcopy(frame)
                if self._ref is None:
                    raise RuntimeError("Digit: unable to copy frame")

            image = frame.image.astype("float32") - self._ref.image.astype("float32") # use float image now
            return Frame(FrameEnc.DIFF, image)
        else:
            raise UnsupportedModelError(f"Digit: model not supported: {model}")
