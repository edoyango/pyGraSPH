import numpy as _np

class wenland_c2:
    def __init__(self, k, h):
        self.k = k
        self.h = h
        self.alpha = 7./(64.*_np.pi*self.h*self.h)
    def w(self, r: _np.ndarray):
        q = r/self.h
        return self.alpha*_np.maximum(0., 2.-q)**4*(2.*q+1.)
    def dwdx(self, dx: _np.ndarray):
        r = _np.sqrt(_np.einsum("ij,ij->i", dx, dx))
        q = r/self.h
        dwdx_coeff = -self.alpha*10.*q[:]*_np.maximum(0., 2.-q[:])**3/(r[:]*self.h)
        return dwdx_coeff[:, _np.newaxis]*dx