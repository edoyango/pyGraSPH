import numpy as _np
from pydantic import BaseModel, Field, computed_field, validate_call
from functools import cached_property
from numpydantic import NDArray, Shape
from typing import Callable

class _template_kernel(BaseModel):
    k: float = Field(gt=0)
    h: float = Field(gt=0)
    """A template to create other kernels"""

    @computed_field
    @cached_property
    def _alpha(self) -> float:
        """
        Computes the kernel normalization factor used in kernel calculations, 
        using the provided function.
        """
        return self.alpha(self.h)
    
    @validate_call
    def __call__(self, r: NDArray[Shape["* n"], _np.float64]) -> NDArray[Shape["* n"], _np.float64]:
        """
        Calculates the kernel value for all the given inter-particle distances.
        r: a 1D NDArray of inter-particle distances.
        """
        return self.w(self.h, self._alpha, r)
    
    @validate_call
    def grad(self, dx: NDArray[Shape["* n, 2 d"], _np.float64]) -> NDArray[Shape["* n, 2 d"], _np.float64]:
        """
        Calculates the kernel gradients for all the given inter-particle relative positions.
        dx: a 2D NDArray of interparticle relative positions.
        """
        return self.dwdx(self.h, self._alpha, dx)

    @staticmethod
    def w(h, alpha, r):
        raise NotImplementedError("This method should be overridden by the user.")
    
    @staticmethod
    def dwdx(h, alpha, dx):
        raise NotImplementedError("This method should be overridden by the user.")
    
    @staticmethod
    def alpha(h):
        raise NotImplementedError("This method should be overridden by the user.")


class wendland_c2(_template_kernel):
    """
    The Wendland (C2) kernel function class.
    """

    @staticmethod
    def alpha(h) -> float:
        return 7./(64.*_np.pi*h*h)

    @staticmethod
    def w(h, alpha, r: NDArray[Shape["* n"], _np.float64]) -> NDArray[Shape["* n"], _np.float64]:
        q = r/h
        return alpha*_np.maximum(0., 2.-q)**4*(2.*q+1.)
    
    @staticmethod
    def dwdx(h, alpha, dx: NDArray[Shape["* n, 2 d"], _np.float64]) -> NDArray[Shape["* n, 2 d"], _np.float64]:
        r = _np.sqrt(_np.einsum("ij,ij->i", dx, dx))
        q = r/h
        dwdx_coeff = -alpha*10.*q[:]*_np.maximum(0., 2.-q[:])**3/(r[:]*h)
        return dwdx_coeff[:, _np.newaxis]*dx