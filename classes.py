import numpy as np
import scipy as sp
import typing
import h5py
from numpy.linalg import norm

# define identity and D2 matrisices (Voigt notation)
IDE = np.ascontiguousarray([1, 1, 1, 0])
D2 = np.ascontiguousarray([1, 1, 1, 2])

# particles base class
class particles:
    def __init__(self, maxn: int, dx: float, rho_ini: float, maxinter: int, c: float, **customvals):

        # particle constants
        self.dx = dx               # particle spacing (m)
        self.h = 1.5*dx            # smoothing length (m)
        self.rho_ini = rho_ini     # initial/reference density (km/m3)
        self.mass = rho_ini*dx**2  # mass (kg)
        self.maxn = maxn           # max number of particles
        self.ntotal = 0            # initialise no. real particles
        self.nvirt = 0             # initialise no. virtual particles
        self.c = c                 # speed of sound in material

        # initialize arrays (dtype necessary to mitigate roundoff errors)
        self.x = np.zeros((maxn, 2), dtype=np.float64)
        self.v = np.zeros((maxn, 2), dtype=np.float64)
        self.rho = np.full(maxn, rho_ini, dtype=np.float64)
        self.id = np.arange(maxn, dtype=np.int32)
        self.type = np.ones(maxn, dtype=np.int32)
        self.strain = np.zeros((maxn, 4), dtype=np.float64)
        self.sigma = np.zeros((maxn, 4), dtype=np.float64)

        self.pairs = np.ndarray((maxinter, 2))

        # custom data in dict
        self.customvals = customvals

    # generate real particles (user to define)
    def generate_real_coords(self):

        pass

    # generate virtual particles (user to define)
    def generate_virt_coords(self):

        pass

    # stress update function (DP model)
    def stress_update(self, dstrain: np.ndarray, drxy: float, sigma0: np.ndarray):

        self.sigma[0:self.ntotal, :] = sigma0[0:self.ntotal, :] + np.matmul(dstrain[0:self.ntotal], self.customvals['DE'])
        self.sigma[0:self.ntotal, 3] += sigma0[0:self.ntotal, 0]*drxy[0:self.ntotal] - sigma0[0:self.ntotal, 1]*drxy[0:self.ntotal]
        
        s = np.zeros(4)
        dfdsig = np.zeros(4)
        dgdsig = np.zeros(4)

        for i in range(self.ntotal):

            # stress invariants
            I1 = self.sigma[i, 0] + self.sigma[i, 1] + self.sigma[i, 2]
            s[0:4] = self.sigma[i, 0:4] - I1/3.*IDE[0:4]
            J2 = 0.5*(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]+2.*s[3]*s[3])

            # tensile cracking check 1: 
            # J2 is zero but I1 is beyond apex of yield surface
            if J2 == 0 and I1 > self.customvals['k_c']:
                I1 = self.customvals['k_c']
                self.sigma[i, 0:4] = I1/3.*IDE[0:4]

            # calculate yield function
            f = self.customvals['alpha_phi']*I1 + np.sqrt(J2) - self.customvals['k_c']

            # Perform corrector step
            if f > 0.:
                dfdsig[0] = self.customvals['alpha_phi'] + s[0]/(2.*np.sqrt(J2))
                dfdsig[1] = self.customvals['alpha_phi'] + s[1]/(2.*np.sqrt(J2))
                dfdsig[2] = self.customvals['alpha_phi'] + s[2]/(2.*np.sqrt(J2))
                dfdsig[3] = s[3]/(2.*np.sqrt(J2))
                dgdsig[0] = self.customvals['alpha_psi'] + s[0]/(2.*np.sqrt(J2))
                dgdsig[1] = self.customvals['alpha_psi'] + s[1]/(2.*np.sqrt(J2))
                dgdsig[2] = self.customvals['alpha_psi'] + s[2]/(2.*np.sqrt(J2))
                dgdsig[3] = s[3]/(2.*np.sqrt(J2))

                dlambda = f/(np.dot(dfdsig[0:4], D2[0:4]*np.matmul(self.customvals['DE'][0:4, 0:4], dgdsig[0:4])))

                self.sigma[i, 0:4] -= np.matmul(self.customvals['DE'][0:4, 0:4], dlambda*dgdsig[0:4])

            # tensile cracking check 2:
            # corrected stress state is outside yield surface
            I1 = self.sigma[i, 0] + self.sigma[i, 1] + self.sigma[i, 2]
            if I1 > self.customvals['k_c']/self.customvals['alpha_phi']:
                self.sigma[i, 0:3] = self.customvals['k_c']/self.customvals['alpha_phi']/3
                self.sigma[i, 3] = 0.

        # simple fluid equation of state.
        # for i in range(self.ntotal+self.nvirt):
        #     p = self.c*self.c*(self.rho[i] - self.rho_ini)
        #     self.sigma[i, 0:3] = -p/3.
        #     self.sigma[i, 3] = 0.

    # function to update self.pairs - list of particle pairs
    def findpairs(self):

        tree = sp.spatial.cKDTree(self.x[0:self.ntotal+self.nvirt, :])
        self.pairs = tree.query_pairs(3*self.dx, output_type='ndarray')

    def update_virtual_particles_properties(self, kernel):

        # zeroing virutal particles' properties
        self.v[self.ntotal:self.ntotal+self.nvirt, :].fill(0.)
        self.rho[self.ntotal:self.ntotal+self.nvirt].fill(0.)
        self.sigma[self.ntotal:self.ntotal+self.nvirt, :].fill(0.)
        vw = np.zeros(self.ntotal+self.nvirt, dtype=np.float64)

        # cache reference to kernel.w
        kernelw = kernel.w

        mass = self.mass

        # cache the references to data
        pairs = self.pairs
        x = self.x
        v = self.v
        rho = self.rho
        sigma = self.sigma
        
        # sweep over all pairs and update virtual particles' properties
        # only consider real-virtual pairs
        for k in range(pairs.shape[0]):
            i = pairs[k, 0]
            j = pairs[k, 1]

            if self.type[i] < 0 and self.type[j] > 0:
                w = kernelw(norm(x[i, :]-x[j,:]))
                vw[i] += w*mass/rho[j]
                v[i, 0:2] -= v[j, 0:2]*mass/rho[j]*w
                rho[i] += mass*w
                sigma[i, 0:4] += sigma[j, 0:4]*mass/rho[j]*w
            elif self.type[i] > 0 and self.type[j] < 0:
                w = kernelw(norm(x[i, 0:2]-x[j, 0:2]))
                vw[j] += w*mass/rho[i]
                v[j, 0:2] -= v[i, 0:2]*mass/rho[i]*w
                rho[j] += mass*w
                sigma[j, 0:4] += sigma[i, 0:4]*mass/rho[i]*w

        # normalize virtual particle properties with summed kernels
        for i in range(self.ntotal, self.ntotal+self.nvirt):
            if vw[i] > 0.:
                v[i, 0:2] /= vw[i]
                rho[i] /= vw[i]
                sigma[i, 0:4] /= vw[i]
            else:
                rho[i] = self.rho_ini

    # function to perform sweep over all particle pairs
    def pair_sweep(self, 
                   dvdt: np.ndarray, 
                   drhodt: np.ndarray, 
                   dstraindt: np.ndarray, 
                   rxy: np.ndarray,
                   kernel: typing.Type):
        
        ## update virtual particles' properties first --------------------------
        self.update_virtual_particles_properties(kernel)

        mass = self.mass

        # cache the references to data
        pairs = self.pairs
        x = self.x
        v = self.v
        rho = self.rho
        sigma = self.sigma

        he = np.zeros(4, dtype=np.float64)
        dv = np.zeros(2)
        dx = np.zeros(2)

        kerneldwdx = kernel.dwdx

        # sweep over all pairs to update real particles' material rates --------
        for k in range(pairs.shape[0]):
            i = pairs[k, 0]
            j = pairs[k, 1]

            # only consider real-real or real-virtual (exclude virtual-virtual)
            if self.type[i] > 0 or self.type[j] > 0:

                rhoi = rho[i]
                rhoj = rho[j]

                sigmai = sigma[i, 0:4]
                sigmaj = sigma[j, 0:4]
                
                # calculate differential position vector and kernel gradient
                dx[0:2] = x[i, 0:2] - x[j, 0:2]
                dwdx = kerneldwdx(dx)

                # update acceleration with artificial viscosity
                dv[0:2] = v[i, 0:2] - v[j, 0:2]
                vr = dv[0]*dx[0] + dv[1]*dv[1]
                if vr > 0.: vr = 0.
                rr = dx[0]*dx[0] + dx[1]*dx[1]
                muv = self.h*vr/(rr + self.h*self.h*0.01)
                mrho = 0.5*(rhoi+rhoj)
                piv = mass*0.2*(muv-self.c)*muv/mrho*dwdx
                dvdt[i, 0:2] -= piv[0:2]
                dvdt[j, 0:2] += piv[0:2]

                # update acceleration with div stress
                # using momentum consertive form
                h = mass*((sigmai[0]*dwdx[0]+sigmai[3]*dwdx[1])/rhoi**2 + 
                                (sigmaj[0]*dwdx[0]+sigmaj[3]*dwdx[1])/rhoj**2)
                dvdt[i, 0] += h
                dvdt[j, 0] += h

                h = mass*((sigmai[3]*dwdx[0]+sigmai[1]*dwdx[1])/rhoi**2 +
                                (sigmaj[3]*dwdx[0]+sigmaj[1]*dwdx[1])/rhoj**2)
                dvdt[i, 1] += h
                dvdt[j, 1] -= h

                tmp_drhodt = mass*(dv[0]*dwdx[0] + dv[1]*dwdx[1])
                drhodt[i] += tmp_drhodt
                drhodt[j] += tmp_drhodt

                # calculating engineering strain rates
                he[0] = -dv[0]*dwdx[0]
                he[1] = -dv[1]*dwdx[1]
                #he[2] = 0.
                he[3] = -0.5*(dv[0]*dwdx[1]+dv[1]*dwdx[0])
                hrxy = -0.5*(dv[0]*dwdx[1] - dv[1]*dwdx[0])

                dstraindt[i, 0] += mass*he[0]/rhoj
                dstraindt[i, 1] += mass*he[1]/rhoj
                # dstraindt[i, 2] += mass*he[2]/self.rho[j]
                dstraindt[i, 3] += mass*he[3]/rhoj
                rxy[i] += mass*hrxy/rhoj

                dstraindt[j, 0] += mass*he[0]/rhoi
                dstraindt[j, 1] += mass*he[1]/rhoi
                # dstraindt[j, 2] += mass*he[2]/self.rho[i]
                dstraindt[j, 3] += mass*he[3]/rhoi
                rxy[j] += mass*hrxy/rhoi

    # function to save particle data
    def save_data(self, itimestep: int):

        with h5py.File(f'output/sph_{itimestep}.h5', 'w') as f:
            f.attrs.create("n", data=self.ntotal+self.nvirt, dtype="i")
            f.create_dataset("x", data=self.x[0:self.ntotal+self.nvirt, :], dtype="f8", compression="gzip")
            f.create_dataset("v", data=self.v[0:self.ntotal+self.nvirt, :], dtype="f8", compression="gzip")
            f.create_dataset("type", data=self.type[0:self.ntotal+self.nvirt], dtype="i", compression="gzip")
            f.create_dataset("rho", data=self.rho[0:self.ntotal+self.nvirt], dtype="f8", compression="gzip")
            f.create_dataset("sigma", data=self.sigma[0:self.ntotal+self.nvirt, :], dtype="f8", compression="gzip")
            f.create_dataset("strain", data=self.strain[0:self.ntotal+self.nvirt, :], dtype="f8", compression="gzip")

# container class to hold time integration functions
class integrators:
    def __init__(self,
                 f: np.ndarray,
                 kernel: typing.Type,
                 maxtimestep: int,
                 savetimestep: int,
                 printtimestep: int,
                 cfl: float):
        self.cfl = cfl # Courant-Freidrichs-Lewy coefficient for time-step size
        self.f = f     # body force vector e.g. gravity
        self.kernel = kernel # kernel function of choice
        self.maxtimestep = maxtimestep # timestep to run simulation for
        self.savetimestep = savetimestep # timestep interval to save data to disk
        self.printtimestep = printtimestep # timestep interval to print timestep

    def LF(self, pts: particles):

        # initialize arrays needed for time integration
        dvdt = np.tile(self.f, (pts.ntotal+pts.nvirt, 1)) # acceleration
        v0 = np.empty((pts.ntotal+pts.nvirt, 2), dtype=np.float64) # velocity at start of timestep
        drhodt = np.zeros(pts.ntotal+pts.nvirt, dtype=np.float64) # density change rate
        rho0 = np.empty(pts.ntotal+pts.nvirt, dtype=np.float64) # density at start of timestep
        dstraindt = np.zeros((pts.ntotal+pts.nvirt, 4), dtype=np.float64) # strain rate
        rxy = np.zeros(pts.ntotal+pts.nvirt, dtype=np.float64) # spin rate (for jaumann stress-rate)
        sigma0 = np.empty((pts.ntotal+pts.nvirt, 4), dtype=np.float64) # stress at start of timestep

        # timestep size (s)
        dt = self.cfl*pts.dx*3./pts.c

        # begin time integration loop
        for itimestep in range(1, self.maxtimestep+1):

            # find pairs
            pts.findpairs()

            # save data from start of timestep
            v0 = np.copy(pts.v[0:pts.ntotal+pts.nvirt, :])
            rho0 = np.copy(pts.rho[0:pts.ntotal+pts.nvirt])
            sigma0 = np.copy(pts.sigma[0:pts.ntotal+pts.nvirt, :])

            # Update data to mid-timestep
            pts.rho[0:pts.ntotal] += 0.5*dt*drhodt[0:pts.ntotal]
            pts.v[0:pts.ntotal, :] += 0.5*dt*dvdt[0:pts.ntotal, :]
            pts.stress_update(0.5*dt*dstraindt[0:pts.ntotal, :], 0.5*dt*rxy[0:pts.ntotal], sigma0[0:pts.ntotal, :])

            # initialize material rate arrays
            dvdt = np.tile(self.f, (pts.ntotal+pts.nvirt, 1))
            drhodt = np.zeros(pts.ntotal+pts.nvirt)
            dstraindt = np.zeros((pts.ntotal+pts.nvirt, 4))
            rxy = np.zeros(pts.ntotal+pts.nvirt)
            
            # perform sweep of pairs
            pts.pair_sweep(dvdt, drhodt, dstraindt, rxy, self.kernel)

            # update data to full-timestep
            pts.rho[0:pts.ntotal] = rho0[0:pts.ntotal] + dt*drhodt[0:pts.ntotal]
            pts.v[0:pts.ntotal, :] = v0[0:pts.ntotal, :] + dt*dvdt[pts.ntotal, :]
            pts.x[0:pts.ntotal, :] += dt*pts.v[0:pts.ntotal, :]
            pts.stress_update(dt*dstraindt[0:pts.ntotal, :], dt*rxy[0:pts.ntotal], sigma0[0:pts.ntotal, :])
            pts.strain[0:pts.ntotal, :] += dt*dstraindt[0:pts.ntotal, :]

            # print data to terminal if needed
            if itimestep % self.printtimestep == 0:
                print(f'time-step: {itimestep}')
            
            # save data to disk if needed
            if itimestep % self.savetimestep == 0:
                pts.save_data(itimestep)
