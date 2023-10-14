import numpy as _np

class _template_kernel:
    """A template to create other kernels"""
    def __init__(self, k, h):
        """
        Initializes the kernel with k and h values. 
        Calculates kernel coefficient for reuse in later calls.
        k: the number of smoothing lengths that define the cutoff.
        h: the smoothing length of the kernel.
        """
        self.k = k
        self.h = h
        self.alpha = self.alpha()
    def w(self, r: _np.ndarray) -> _np.ndarray:
        """
        Calculates the kernel value for all the given inter-particle distances.
        r: a 1D NDArray of inter-particle distances.
        """
        raise NotImplementedError("This method should be overridden by the user.")
    
    def dwdx(self, dx: _np.ndarray) -> _np.ndarray:
        """
        Calculates the kernel gradients for all the given inter-particle relative positions.
        dx: a 2D NDArray of interparticle relative positions.
        """
        raise NotImplementedError("This method should be overridden by the user.")
    
    def alpha(self):
        """
        Calculates the kernel normalization factor.
        """
        raise NotImplementedError("This method should be overridden by the user.")

class wendland_c2(_template_kernel):
    """
    The Wendland (C2) kernel function class.
    """
    def alpha(self) -> float:
        return 7./(64.*_np.pi*self.h*self.h)

    def w(self, r: _np.ndarray) -> _np.ndarray:
        q = r/self.h
        return self.alpha*_np.maximum(0., 2.-q)**4*(2.*q+1.)
    def dwdx(self, dx: _np.ndarray) -> _np.ndarray:
        r = _np.sqrt(_np.einsum("ij,ij->i", dx, dx))
        q = r/self.h
        dwdx_coeff = -self.alpha*10.*q[:]*_np.maximum(0., 2.-q[:])**3/(r[:]*self.h)
        return dwdx_coeff[:, _np.newaxis]*dx