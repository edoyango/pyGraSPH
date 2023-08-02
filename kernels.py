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
        q = np.sqrt(dx[0]*dx[0]+dx[1]*dx[1])/self.h
        return -self.alpha*10.*max(0., 2.-q)**3*dx[:]/(self.h*self.h)