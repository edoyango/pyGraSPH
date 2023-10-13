import numpy as _np

class wendland_c2:
    """
    The Wendland (C2) kernel function class.
    """
    def __init__(self, k, h):
        """
        Initializes the kernel with k and h values. 
        Calculates kernel coefficient for reuse in later calls.
        k: the number of smoothing lengths that define the cutoff.
        h: the smoothing length of the kernel.
        """
        self.k = k
        self.h = h
        self.alpha = 7./(64.*_np.pi*self.h*self.h)

    def w(self, r: _np.ndarray):
        """
        Calculates the kernel value for all the given inter-particle distances.
        r: a 1D NDArray of inter-particle distances.
        """
        q = r/self.h
        return self.alpha*_np.maximum(0., 2.-q)**4*(2.*q+1.)
    def dwdx(self, dx: _np.ndarray):
        """
        Calculates the kernel gradients for all the given inter-particle relative positions.
        dx: a 2D NDArray of interparticle relative positions.
        """
        r = _np.sqrt(_np.einsum("ij,ij->i", dx, dx))
        q = r/self.h
        dwdx_coeff = -self.alpha*10.*q[:]*_np.maximum(0., 2.-q[:])**3/(r[:]*self.h)
        return dwdx_coeff[:, _np.newaxis]*dx