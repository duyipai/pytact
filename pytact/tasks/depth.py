import math

import intel_extension_for_pytorch as ipex
import numpy as np
import scipy
import torch
from pytact.models import Pixel2GradModel
from pytact.types import DepthMap, Frame

from .tasks import Task


def poisson_reconstruct(gradx, grady, boundarysrc):
    # Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
    # Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

    # Laplacian
    gyy = grady[1:, :-1] - grady[:-1, :-1]
    gxx = gradx[:-1, 1:] - gradx[:-1, :-1]
    f = np.zeros(boundarysrc.shape)
    f[:-1, 1:] += gxx
    f[1:, :-1] += gyy

    # Boundary image
    boundary = boundarysrc.copy()
    boundary[1:-1, 1:-1] = 0

    # Subtract boundary contribution
    f_bp = (
        -4 * boundary[1:-1, 1:-1]
        + boundary[1:-1, 2:]
        + boundary[1:-1, 0:-2]
        + boundary[2:, 1:-1]
        + boundary[0:-2, 1:-1]
    )
    f = f[1:-1, 1:-1] - f_bp

    # Discrete Sine Transform
    tt = scipy.fft.dst(f, norm="ortho")
    fsin = scipy.fft.dst(tt.T, norm="ortho").T

    # Eigenvalues
    (x, y) = np.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    denom = (2 * np.cos(math.pi * x / (f.shape[1] + 2)) - 2) + (
        2 * np.cos(math.pi * y / (f.shape[0] + 2)) - 2
    )

    f = fsin / denom

    # Inverse Discrete Sine Transform
    tt = scipy.fft.idst(f, norm="ortho")
    img_tt = scipy.fft.idst(tt.T, norm="ortho").T

    # New center + old boundary
    result = boundary
    result[1:-1, 1:-1] = img_tt

    return result


class DepthFromLookup(Task):
    """
    Computes a sensor's depth map using a 3-layer MLP which learned
    the lookup table for each pixel's gradient.

    Paper: https://doi.org/10.1109/ICRA48506.2021.9560783

    Parameters
    ----------
    model_path: str
        Path to model parameters; must match MLPGradModel in models/.
    """

    def __init__(
        self, model_path: str, use_cuda=False, mean=None, std=None, optimize=False
    ):

        self.model = Pixel2GradModel()
        self.use_cuda = use_cuda
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        if mean is not None and std is not None:
            self.mean = torch.from_numpy(mean).float()
            self.std = torch.from_numpy(std).float()
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.mean is not None and self.std is not None:
                self.mean = self.mean.cuda()
                self.std = self.std.cuda()
        elif optimize:
            self.model = ipex.optimize(self.model)

    def __call__(self, frame: Frame) -> DepthMap:
        height, width = frame.image.shape[:2]
        batch_len = height * width
        X = frame.image.reshape((batch_len, 3))
        xv, yv = np.meshgrid(np.arange(height), np.arange(width))
        X = np.concatenate((X, np.reshape(xv, (batch_len, 1))), axis=1)
        X = np.concatenate((X, np.reshape(yv, (batch_len, 1))), axis=1)

        # Collect gradients from model and reshape
        X = torch.from_numpy(X).float()
        if self.use_cuda:
            X = X.cuda()
        if self.mean is not None and self.std is not None:
            X = (X - self.mean) / self.std
        with torch.no_grad():
            grad = self.model(X).cpu()
            grad = (
                grad.detach()
                .type(torch.FloatTensor)
                .numpy()
                .reshape((height, width, 2))
            )
        dm = poisson_reconstruct(
            grad[:, :, 0], grad[:, :, 1], np.zeros((height, width))
        )
        dm = np.reshape(dm, (height, width))
        return DepthMap(dm)


class DepthFromPix2Pix(Task):
    """
    Computes a sensor's depth map using a Pix2Pix architecture.

    TODO: Implement.

    Parameters
    ----------
    model_path: str
        Path to model parameters; must match MLPGradModel in models/.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        raise NotImplementedError()

    def __call__(self, frame: Frame) -> DepthMap:
        if frame is None:
            raise RuntimeError("Could not retrieve frame")

        raise NotImplementedError()
