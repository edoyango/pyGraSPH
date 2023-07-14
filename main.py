import scipy as sp
import numpy as npy
import classes
import kernels

class my_particles(classes.particles):

    def generate_real_coords(self, mp: int, np: int):

        for i in range(mp):
            for j in range(np):
                self.x[self.ntotal, 0] = (i + 0.5)*self.dx
                self.x[self.ntotal, 1] = (j + 0.5)*self.dx
                self.type[self.ntotal] = 1
                self.ntotal += 1

    def generate_virt_coords(self, pp: int, op: int, nlayer: int):

        for i in range(-nlayer, pp+nlayer):
            for j in range(nlayer):
                self.x[self.ntotal+self.nvirt, 0] = (i + 0.5)*self.dx
                self.x[self.ntotal+self.nvirt, 1] =-(j + 0.5)*self.dx
                self.type[self.ntotal+self.nvirt] = -1
                self.nvirt += 1

        for i in range(op):
            for j in range(nlayer):
                self.x[self.ntotal+self.nvirt, 0] =-(j + 0.5)*self.dx
                self.x[self.ntotal+self.nvirt, 1] = (i + 0.5)*self.dx
                self.type[self.ntotal+self.nvirt] = -2
                self.nvirt += 1

if __name__ == '__main__':

    mp = 50
    np = 25
    pp = 3*mp
    op = np
    nlayer = 4
    maxn = 15000
    pts = my_particles(maxn=maxn, dx=0.5, rho_ini=1600., maxinter=25*maxn, c=npy.sqrt(10e6/1600))
    
    pts.generate_real_coords(mp=mp, np=np)
    pts.generate_virt_coords(pp=pp, op=op, nlayer=nlayer)

    g = [0., -9.81]
    
    wc2_kernel = kernels.wenland_c2(k=2, h=pts.dx*1.5)
    print(wc2_kernel.h)
    itgs = classes.integrators(f=g, kernel=wc2_kernel, maxtimestep=1000, savetimestep=100, printtimestep=10, cfl=0.2)
    
    pts.integrate(itgs.LF)
    