import numpy as np
from numpy import sqrt, pi

class wenland_c2:
    def __init__(self, k, h):
        self.k = k
        self.h = h
        self.alpha = 7./(64.*pi*h*h)
    def w(self, r: float):
        q = r/self.h
        return self.alpha*max(0., 2.-q)**4*(2.*q+1.)
    def dwdx(self, dx: np.ndarray):
        h = self.h
        q = sqrt(dx[0]*dx[0]+dx[1]*dx[1])/h
        return -self.alpha*10.*max(0., 2.-q)**3*dx[0:2]/(h*h)