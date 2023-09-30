import numpy as np

class wenland_c2:
    def __init__(self, k, h):
        self.k = k
        self.h = h
        self.alpha = 7./(64.*np.pi*self.h*self.h)
    def w(self, r: np.ndarray):
        q = r/self.h
        return self.alpha*np.maximum(0., 2.-q)**4*(2.*q+1.)
    def dwdx(self, dx: np.ndarray):
        r = np.apply_along_axis(np.linalg.norm, 1, dx)
        q = r/self.h
        dwdx_coeff = -self.alpha*10.*q[:]*np.maximum(0., 2.-q[:])**3/(r[:]*self.h)
        return np.einsum("i,ij->ij", dwdx_coeff, dx)