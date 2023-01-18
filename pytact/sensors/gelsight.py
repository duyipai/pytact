import cv2
import threading
from typing import Tuple, Optional, List
from copy import deepcopy
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import maximum_filter, minimum_filter

from pytact.types import ModelType, FrameEnc, Frame, Markers
from .sensors import Sensor, UnsupportedModelError


class GelsightR15(Sensor):
    """
    Retrieves images from an http stream and assumes markers are present.

    Parameters
    ----------
    url: str
        URL of the HTTP stream
    size: Tuple[int, int], optional
        Downscaled resolution of image (default (160, 120))
    roi: List[Tuple[int, int], ...], optional
        Sequence of coordinates that indicate the sensor frame's region of interest.
        List should include 4 coordinates in order of the top-left, top-right,
        bottom-right and bottom-left (default frame is left as is).
    encoding: FrameEnc, optional
        Encoding to process frames as (default FrameEnc.BGR)
    sample_rate: float, optional
        Rate to capture new frames at (default 30.0)
    marker_shape: Tuple[int, int], optional
        Number of marker rows and columns (default (10, 12))
    marker_block_size: int, optional
    marker_neg_bias: int, optional
    marker_neighborhood_size: int, optional
    """

    _encoding = FrameEnc.BGR
    _size: Tuple[int, int] = (160, 120)  # width, height
    _roi: Optional[List[Tuple[int, int]]] = [
        [95, 135],
        [360, 130],
        [340, 470],
        [130, 470],
    ]  # TL, TR, BR, BL
    _sample_rate: float = 30.0
    _dev = None
    _marker_shape: Tuple[int, int] = (10, 14)  # rows, cols
    _marker_block_size: int = 51
    _marker_neg_bias: int = 19
    _marker_neighborhood_size: int = 33

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "url" in kwargs:
            self._dev = cv2.VideoCapture(kwargs["url"])
        if "roi" in kwargs:
            self._roi = kwargs["roi"]
        self.output_coords = [
            (0, 0),
            (self._size[0], 0),
            (self._size[0], self._size[1]),
            (0, self._size[1]),
        ]

        # Start frame sampling
        self._is_running: bool = True
        self.has_marker = True
        self._frame: Optional[Frame] = None
        self._ref: Optional[Frame] = None
        if self._dev is not None:
            threading.Timer(1.0 / self._sample_rate, self._collect_frame).start()

    @property
    def marker_shape(self) -> Tuple[int, int]:
        return self._marker_shape

    @property
    def supported_models(self) -> List[ModelType]:
        return [ModelType.Pixel2Grad]

    def set_reference(self, frame: Frame):
        self._ref = deepcopy(frame)

    def get_reference(self) -> Optional[Frame]:
        return deepcopy(self._ref)

    def _collect_frame(self):
        """Runs at predetermined rate to collect frames from the sensor."""
        ret, frame = self._dev.read()
        if not ret:
            self.is_running = False
            return

        # Warp to match ROI
        if self._roi is not None:
            M = cv2.getPerspectiveTransform(
                np.float32(self._roi), np.float32(self.output_coords)
            )
            frame = cv2.warpPerspective(frame, M, self._size)

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
                    raise RuntimeError("GelsightR15: unable to copy frame")

            image = frame.image.astype("float32") - self._ref.image.astype(
                "float32"
            )  # use float image now
            return Frame(FrameEnc.DIFF, image)
        else:
            raise UnsupportedModelError("GelsightR15: model not supported: {model}")

    def get_markers(self) -> Optional[Markers]:
        if self._frame is None:
            return None

        # Convert to grayscale and compute mask
        if self._frame.encoding == FrameEnc.BGR:
            gray_im = cv2.cvtColor(self._frame.image, cv2.COLOR_BGR2GRAY)
        elif self._frame.encoding == FrameEnc.GRAY:
            gray_im = self._frame.image
        else:
            raise ValueError("GelsightR15: invalid frame encoding")
        mask = cv2.adaptiveThreshold(
            gray_im,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self._marker_block_size,
            self._marker_neg_bias,
        )

        # Find peaks
        max = maximum_filter(mask, self._marker_neighborhood_size)
        maxima = mask == max
        min = minimum_filter(mask, self._marker_neighborhood_size)
        diff = (max - min) > 1
        maxima[diff == 0] = 0

        # Label peaks as markers
        labeled, n = ndimage.label(maxima)
        xy = np.array(ndimage.center_of_mass(mask, labeled, range(1, n + 1)))
        xy[:, [0, 1]] = xy[:, [1, 0]]
        return Markers(self._marker_shape[0], self._marker_shape[1], xy)

    def get_markers_from_frame(self, frame) -> Optional[Markers]:
        # Convert to grayscale and compute mask
        if frame.encoding == FrameEnc.BGR:
            if frame.image.dtype == np.dtype("uint8"):
                gray_im = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            else:
                gray_im = cv2.cvtColor(np.uint8(frame.image), cv2.COLOR_BGR2GRAY)
        elif frame.encoding != FrameEnc.GRAY:
            gray_im = frame.image

        mask = cv2.adaptiveThreshold(
            gray_im,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self._marker_block_size,
            self._marker_neg_bias,
        )

        # Find peaks
        max = maximum_filter(mask, self._marker_neighborhood_size)
        maxima = mask == max
        min = minimum_filter(mask, self._marker_neighborhood_size)
        diff = (max - min) > 1
        maxima[diff == 0] = 0
        return maxima, mask
