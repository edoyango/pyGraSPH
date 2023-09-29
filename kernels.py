import numpy as np

class wenland_c2:
    def __init__(self, k, h):
        self.k = k
        self.h = h
        self.alpha = 7./(64.*np.pi*self.h*self.h)
    def w(self, r: float):
        q = r/self.h
        return self.alpha*max(0., 2.-q)**4*(2.*q+1.)
    def dwdx(self, dx: np.ndarray):
        r = np.linalg.norm(dx)
        q = r/self.h
        return -self.alpha*10.*q*max(0., 2.-q)**3*dx[:]/(r*self.h)