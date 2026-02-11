import math
from dataclasses import dataclass

import numpy as np

from tractor.utils import BaseParams
from tractor import ducks
from tractor.psf import PixelizedPSF, GaussianMixturePSF


def _moffat_alpha_from_fwhm(fwhm_pix: float, beta: float) -> float:
    """
    Circular Moffat:
        I(r) ∝ (1 + (r/alpha)^2)^(-beta)
    FWHM relation:
        FWHM = 2 * alpha * sqrt(2^(1/beta) - 1)
    """
    fwhm_pix = float(fwhm_pix)
    beta = float(beta)
    if not np.isfinite(fwhm_pix) or fwhm_pix <= 0:
        raise ValueError(f"Bad fwhm_pix: {fwhm_pix}")
    if not np.isfinite(beta) or beta <= 1:
        # beta>1 ensures finite total flux in 2D
        raise ValueError(f"Bad beta (must be >1): {beta}")
    denom = 2.0 * math.sqrt(math.pow(2.0, 1.0 / beta) - 1.0)
    return fwhm_pix / denom


def _moffat_stamp(alpha: float, beta: float, radius_pix: float) -> np.ndarray:
    """
    Build a normalized, odd-sized Moffat stamp centered at the middle pixel.
    """
    alpha = float(alpha)
    beta = float(beta)
    radius_pix = float(radius_pix)
    if not (np.isfinite(alpha) and alpha > 0):
        raise ValueError(f"Bad alpha: {alpha}")
    if not (np.isfinite(beta) and beta > 1):
        raise ValueError(f"Bad beta: {beta}")
    if not (np.isfinite(radius_pix) and radius_pix >= 1):
        raise ValueError(f"Bad radius_pix: {radius_pix}")

    r = int(math.ceil(radius_pix))
    size = 2 * r + 1
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    rr2 = (xx * xx + yy * yy).astype(np.float32)
    a2 = float(alpha * alpha)
    core = (1.0 + rr2 / a2).astype(np.float64)
    img = np.power(core, -float(beta)).astype(np.float64)

    # Normalize to sum=1
    s = float(np.nansum(img))
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError("Failed to normalize Moffat stamp (non-positive sum).")
    img /= s

    return img.astype(np.float32)


@dataclass
class MoffatPSFConfig:
    fwhm_pix: float
    beta: float = 3.5
    # Rendering radius of the stamp (pixels). Bigger captures more wings.
    radius_pix: float = 25.0
    # PixelizedPSF Lanczos order for subpixel shifting
    Lorder: int = 3


class MoffatPSF(BaseParams, ducks.ImageCalibration):
    """
    Tractor-compatible circular Moffat PSF.
    """

    def __init__(
        self,
        fwhm_pix: float,
        beta: float = 3.5,
        radius_pix: float = 25.0,
        Lorder: int = 3,
        mog_N: int = 3,
    ):
        self.fwhm_pix = float(fwhm_pix)
        self.beta = float(beta)
        self.radius_pix = float(radius_pix)
        self.Lorder = int(Lorder)
        self.mog_N = int(mog_N)

        alpha = _moffat_alpha_from_fwhm(self.fwhm_pix, self.beta)
        self.alpha_pix = float(alpha)

        stamp = _moffat_stamp(alpha=self.alpha_pix, beta=self.beta, radius_pix=self.radius_pix)
        self._pix = PixelizedPSF(stamp, sampling=1.0, Lorder=self.Lorder)
        self._pix.radius = float(self.radius_pix)

        try:
            N = max(1, int(self.mog_N))
            self._gauss = GaussianMixturePSF.fromStamp(stamp, N=N)
        except Exception:
            self._gauss = None

    @classmethod
    def from_config(cls, cfg: MoffatPSFConfig) -> "MoffatPSF":
        return cls(fwhm_pix=cfg.fwhm_pix, beta=cfg.beta, radius_pix=cfg.radius_pix, Lorder=cfg.Lorder)

    def __str__(self) -> str:
        return f"MoffatPSF(fwhm_pix={self.fwhm_pix:.3f}, beta={self.beta:.3f}, radius_pix={self.radius_pix:.1f})"

    def copy(self) -> "MoffatPSF":
        return MoffatPSF(self.fwhm_pix, beta=self.beta, radius_pix=self.radius_pix, Lorder=self.Lorder, mog_N=self.mog_N)

    def hashkey(self):
        return ("MoffatPSF", float(self.fwhm_pix), float(self.beta), float(self.radius_pix), int(self.Lorder), int(self.mog_N))

    def getRadius(self) -> float:
        return float(self.radius_pix)

    def getShifted(self, dx, dy):
        return self._pix.getShifted(dx, dy)

    def constantPsfAt(self, x, y):
        return self

    def getMixtureOfGaussians(self, **kwargs):
        if self._gauss is None:
            return None
        return self._gauss.getMixtureOfGaussians(**kwargs)

    def __getattr__(self, name):
        return getattr(self._pix, name)
