import threading
from typing import Optional, List
from copy import deepcopy

from pytact.types import ModelType, FrameEnc, Frame
from .sensors import Sensor, UnsupportedModelError
from digit_interface import Digit


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
    _diff_intensity: float = 3.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "serial" in kwargs:
            self._serial = kwargs["serial"]
            self._sensor = Digit(self._serial)
            self._sensor.connect()
            self._size = [
                self._sensor.resolution["width"],
                self._sensor.resolution["height"],
            ]
            self._sample_rate = self._sensor.fps
        else:
            self._sensor = None

        # Start frame sampling
        self._is_running: bool = True
        self._frame: Optional[Frame] = None
        self._ref: Optional[Frame] = None
        threading.Timer(1.0 / self._sample_rate, self._collect_frame).start()

    def __del__(self):
        if self._sensor is not None:
            self._sensor.disconnect()

    @property
    def supported_models(self) -> List[ModelType]:
        return [ModelType.Pixel2Grad]

    def set_resolution(self, resolution_mode):
        """resolutin_mode = "QVGA", "VGA" """
        if self._sensor is not None:
            self._sensor.set_resolution(self._sensor.STEAMS[resolution_mode])

    def set_reference(self, frame: Frame):
        self._ref = deepcopy(frame)

    def _collect_frame(self):
        """Runs at predetermined rate to collect frames from the sensor."""
        frame = self._sensor.get_frame()

        self._frame = Frame(self._encoding, frame)

        # Set reference frame if one isn't set
        if self._ref is None:
            self._ref = deepcopy(self._frame)

    def get_frame(self) -> Optional[Frame]:
        """Returns frame collected in the last sample."""
        return deepcopy(self._frame)

    def is_running(self):
        return self.is_running

    def preprocess_for(self, model: ModelType, frame: Frame) -> Frame:
        if model == ModelType.Pixel2Grad:
            if self._ref is None:
                self._ref = deepcopy(frame)
                if self._ref is None:
                    raise RuntimeError("Digit: unable to copy frame")

            image = (
                (frame.image.astype("float32") * self._diff_intensity)
                - self._ref.image.astype("float32")
            ) * self._diff_intensity  # use float image now
            return Frame(FrameEnc.DIFF, image)
        else:
            raise UnsupportedModelError(f"Digit: model not supported: {model}")
