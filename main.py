import numpy as npy
import pygrasph
from pygrasph.stress_update import DP, linear_EOS


# add function to original particles class to generate real particles
def generate_real_coords(pts: pygrasph.particles, mp: int, np: int):

    # generate rectangle of particles
    for i in range(mp):
        for j in range(np):
            pts.add_real_particle(
                x=((i + 0.5)*pts.dx, (j + 0.5)*pts.dx),
                # velocity defaults to 0
                # rho (density) defaults to rho_ini
                # strain defaults to 0
                # stress defaults to 0
                # type defaults to 1 (real particles must be positive)
            )

# add function to original particles class to generate virtual particles
def generate_virt_coords(pts: pygrasph.particles, pp: int, op: int, nlayer: int):

    # bottom layer
    for i in range(-nlayer, pp+nlayer):
        for j in range(nlayer):
            pts.add_virt_particle(
                x=((i + 0.5)*pts.dx, -(j + 0.5)*pts.dx),
                type=-1 # for virtual particles, type default is -1.
            )

    # left layer
    for i in range(op):
        for j in range(nlayer):
            pts.add_virt_particle(
                x=(-(j + 0.5)*pts.dx, (i + 0.5)*pts.dx),
                type=-1
            )

if __name__ == '__main__':

    # variables to help define geometry
    mp = 50 # no. real particles in x
    np = 25 #                       y
    pp = 3*mp # no. virtual particles in x
    op = np #                            y
    nlayer = 4 # number of layers for boundary
    maxn = 10000 # max number of particles (real+virtual)

    # DP material properties
    E = 0.84e6             # young's modulus (Pa)
    v = 0.3                # poisson's ratio ()
    Kb = E/(3.*(1.-2.*v))  # bulk modulus    (Pa)
    Gs = E/(2.*(1.+v))     # shear modulus   (Pa)
    rho_ini = 2650         # initial/reference density (kg/m3)
    phi = npy.pi/9         # friction angle (rad)
    psi = 0.               # dilation angle (rad)
    cohesion = 0.          # cohesion       (Pa)
    # DP friction coefficient
    alpha_phi = 2.*npy.sin(phi)/(npy.sqrt(3.)*(3.-npy.sin(phi)))
    # DP dilation coefficient
    alpha_psi = 2.*npy.sin(psi)/(npy.sqrt(3.)*(3.-npy.sin(psi)))
    # DP cohesion constant
    k_c = 6.*cohesion*npy.cos(phi)/(npy.sqrt(3.)*(3.-npy.sin(phi)))
    # Calculate elastic-stiffness matrix
    DEcoeff = E/((1.+v)*(1.-2.*v)) 
    DE = DEcoeff*npy.ascontiguousarray([[1.-v,    v,    v,      0.], 
                                        [   v, 1.-v,    v,      0.],
                                        [   v,    v, 1.-v,      0.],
                                        [  0.,   0.,   0., 1.-2.*v]])
    
    c = npy.sqrt((Kb+4./3.*Gs)/rho_ini)
    
    # Initialize particles
    pts = pygrasph.particles(maxn=maxn, 
                             dx=0.002, 
                             rho_ini=rho_ini, 
                             maxinter=25*maxn, 
                             c=c,
                             f_stress_update=DP,
                             # below args are passed to stress_update 
                             E=E, v=v, Kb=Kb, Gs=Gs, DE=DE,
                             alpha_phi=alpha_phi, alpha_psi=alpha_psi, k_c=k_c)
    
    # generate particle data
    generate_real_coords(pts, mp=mp, np=np)
    generate_virt_coords(pts, pp=pp, op=op, nlayer=nlayer)

    # define gravity vector (m/s2)
    g = [0., -9.81]
    
    # select SPH kernel
    wc2_kernel = pygrasph.kernels.wendland_c2(k=2, h=pts.dx*1.5)
    
    # initialize integrators class
    # integrate SPH particles using leap-frog time-integrator
    # itg = pytgrasph.integrators.LF(f=g, kernel=wc2_kernel)
    # integrate SPH particles using RK4 time-integrator
    itg = pygrasph.integrators.RK4(f=g, kernel=wc2_kernel)

    itg(pts, maxtimestep=500, savetimestep=10, printtimestep=10, cfl=2.5)