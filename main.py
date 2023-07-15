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
    maxn = 10000

    E = 0.84e6
    v = 0.3
    Kb = E/(3.*(1.-2.*v))
    Gs = E/(2.*(1.+v))
    rho_ini = 2650
    DEcoeff = E/((1.+v)*(1.-2.*v))
    phi = npy.pi/9
    psi = 0.
    cohesion = 0.
    alpha_phi = 2.*npy.sin(phi)/(npy.sqrt(3.)*(3.-npy.sin(phi)))
    alpha_psi = 2.*npy.sin(psi)/(npy.sqrt(3.)*(3.-npy.sin(psi)))
    k_c = 6.*cohesion*npy.cos(phi)/(npy.sqrt(3.)*(3.-npy.sin(phi)))
    DE = DEcoeff*npy.ascontiguousarray([[1.-v,    v,    v, 0.     ], 
                                        [   v, 1.-v,    v, 0.     ],
                                        [   v,    v, 1.-v, 0.     ],
                                        [  0.,   0.,   0., 1.-2.*v]])
    
    pts = my_particles(maxn=maxn, 
                       dx=0.002, 
                       rho_ini=rho_ini, 
                       maxinter=25*maxn, 
                       c=npy.sqrt(E/rho_ini), 
                       E=E, v=v, Kb=Kb, Gs=Gs, DE=DE,
                       alpha_phi=alpha_phi, alpha_psi=alpha_psi, k_c=k_c)
    
    pts.generate_real_coords(mp=mp, np=np)
    pts.generate_virt_coords(pp=pp, op=op, nlayer=nlayer)

    g = [0., -9.81]
    
    wc2_kernel = kernels.wenland_c2(k=2, h=pts.dx*1.5)
    
    itgs = classes.integrators(f=g, kernel=wc2_kernel, maxtimestep=100, savetimestep=100, printtimestep=10, cfl=0.2)

    itgs.LF(pts)