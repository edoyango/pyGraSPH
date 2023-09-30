import numpy as np
import scipy as sp
import typing
import h5py

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
    def stress_update(self, i: int, dstraini: np.ndarray, drxyi: float, sigma0: np.ndarray):

        # elastic predictor stress increment
        dsig = np.matmul(self.customvals['DE'], dstraini[:])
        # update stress increment with Jaumann stress-rate
        dsig[3] += sigma0[0]*drxyi - sigma0[1]*drxyi

        # update stress state
        self.sigma[i, :] = sigma0[:] + dsig[:]

        # define identity and D2 matrisices (Voigt notation)
        Ide = np.ascontiguousarray([1, 1, 1, 0])
        D2 = np.ascontiguousarray([1, 1, 1, 2])

        # stress invariants
        I1 = self.sigma[i, 0] + self.sigma[i, 1] + self.sigma[i, 2]
        s = self.sigma[i, :] - I1/3.*Ide[:]
        J2 = 0.5*np.dot(s[:], D2[:]*s[:])

        # tensile cracking check 1: 
        # J2 is zero but I1 is beyond apex of yield surface
        if J2 == 0 and I1 > self.customvals['k_c']:
            I1 = self.customvals['k_c']
            self.sigma[i, 0:3] = I1/3.
            self.sigma[i, 3] = 0.

        # calculate yield function
        f = self.customvals['alpha_phi']*I1 + np.sqrt(J2) - self.customvals['k_c']

        # Perform corrector step
        if f > 0.:
            dfdsig = self.customvals['alpha_phi']*Ide[:] + s[:]/(2.*np.sqrt(J2))
            dgdsig = self.customvals['alpha_psi']*Ide[:] + s[:]/(2.*np.sqrt(J2))

            dlambda = f/(np.dot(dfdsig[:], D2[:]*np.matmul(self.customvals['DE'][:, :], dgdsig[:])))

            self.sigma[i, :] -= np.matmul(self.customvals['DE'][:, :], dlambda*dgdsig[:])

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

    def update_virtualparticle_properties(self, kernel: typing.Type):

        # zeroing virutal particles' properties
        self.v[self.ntotal:self.ntotal+self.nvirt, :].fill(0.)
        self.rho[self.ntotal:self.ntotal+self.nvirt].fill(0.)
        self.sigma[self.ntotal:self.ntotal+self.nvirt, :].fill(0.)
        vw = np.zeros(self.ntotal+self.nvirt, dtype=np.float64)

        # sweep over all pairs and update virtual particles' properties
        # only consider real-virtual pairs
        for k in range(self.pairs.shape[0]):
            i = self.pairs[k, 0]
            j = self.pairs[k, 1]

            if self.type[i] < 0 and self.type[j] > 0:
                w = kernel.w(np.linalg.norm(self.x[i, :]-self.x[j,:]))
                vw[i] += w*self.mass/self.rho[j]
                self.v[i, :] -= self.v[j, :]*self.mass/self.rho[j]*w
                self.rho[i] += self.mass*w
                self.sigma[i, :] += self.sigma[j, :]*self.mass/self.rho[j]*w
            elif self.type[i] > 0 and self.type[j] < 0:
                w = kernel.w(np.linalg.norm(self.x[i, :]-self.x[j,:]))
                vw[j] += w*self.mass/self.rho[i]
                self.v[j, :] -= self.v[i, :]*self.mass/self.rho[i]*w
                self.rho[j] += self.mass*w
                self.sigma[j, :] += self.sigma[i, :]*self.mass/self.rho[i]*w

        # normalize virtual particle properties with summed kernels
        for i in range(self.ntotal, self.ntotal+self.nvirt):
            if vw[i] > 0.:
                self.v[i, :] /= vw[i]
                self.rho[i] /= vw[i]
                self.sigma[i, :] /= vw[i]
            else:
                self.rho[i] = self.rho_ini

    # function to perform sweep over all particle pairs
    def pair_sweep(self, 
                   dvdt: np.ndarray, 
                   drhodt: np.ndarray, 
                   dstraindt: np.ndarray, 
                   rxy: np.ndarray,
                   kernel: typing.Type):
        
        ## update virtual particles' properties first --------------------------
        self.update_virtualparticle_properties(kernel)

        # sweep over all pairs to update real particles' material rates --------
        # trim pairs to only consider real-real or real-virtual
        nonvirtvirt_mask = np.logical_or(
            self.pairs[:, 0] > 0,
            self.pairs[:, 1] > 0
        )
        pair_i = self.pairs[nonvirtvirt_mask, 0]
        pair_j = self.pairs[nonvirtvirt_mask, 1]

        # calculate differential position vector and kernel gradient
        dx = self.x[pair_i, :] - self.x[pair_j, :]
        dv = self.v[pair_i, :] - self.v[pair_j, :]
        dwdx = np.apply_along_axis(kernel.dwdx, 1, dx)

        for k in range(pair_i.shape[0]):
            i = pair_i[k]
            j = pair_j[k]

            # update acceleration with artificial viscosity
            vr = np.dot(dv[k, :], dx[k, :])
            if vr > 0.: vr = 0.
            rr = np.dot(dx[k, :], dx[k, :])
            muv = self.h*vr/(rr + self.h*self.h*0.01)
            mrho = 0.5*(self.rho[i]+self.rho[j])
            piv = self.mass*0.2*(muv-self.c)*muv/mrho*dwdx[k, :]
            dvdt[i, :] -= piv[:]
            dvdt[j, :] += piv[:]

            # update acceleration with div stress
            # using momentum consertive form
            h = self.mass*((self.sigma[i, 0]*dwdx[k, 0]+self.sigma[i, 3]*dwdx[k, 1])/self.rho[i]**2 + 
                            (self.sigma[j, 0]*dwdx[k, 0]+self.sigma[j, 3]*dwdx[k, 1])/self.rho[j]**2)
            dvdt[i, 0] += h
            dvdt[j, 0] -= h

            h = self.mass*((self.sigma[i, 3]*dwdx[k, 0]+self.sigma[i, 1]*dwdx[k, 1])/self.rho[i]**2 +
                            (self.sigma[j, 3]*dwdx[k, 0]+self.sigma[j, 1]*dwdx[k, 1])/self.rho[j]**2)
            dvdt[i, 1] += h
            dvdt[j, 1] -= h

            tmp_drhodt = self.mass*np.dot(dv[k, :], dwdx[k, :])
            drhodt[i] += tmp_drhodt
            drhodt[j] += tmp_drhodt

            # calculating engineering strain rates
            he = np.zeros(4, dtype=np.float64)
            he[0] = -dv[k, 0]*dwdx[k, 0]
            he[1] = -dv[k, 1]*dwdx[k, 1]
            #he[2] = 0.
            he[3] = -0.5*(dv[k, 0]*dwdx[k, 1]+dv[k, 1]*dwdx[k, 0])
            hrxy = -0.5*(dv[k, 0]*dwdx[k, 1] - dv[k, 1]*dwdx[k, 0])

            dstraindt[i, 0] += self.mass*he[0]/self.rho[j]
            dstraindt[i, 1] += self.mass*he[1]/self.rho[j]
            # dstraindt[i, 2] += self.mass*he[2]/self.rho[j]
            dstraindt[i, 3] += self.mass*he[3]/self.rho[j]
            rxy[i] += self.mass*hrxy/self.rho[j]

            dstraindt[j, 0] += self.mass*he[0]/self.rho[i]
            dstraindt[j, 1] += self.mass*he[1]/self.rho[i]
            # dstraindt[j, 2] += self.mass*he[2]/self.rho[i]
            dstraindt[j, 3] += self.mass*he[3]/self.rho[i]
            rxy[j] += self.mass*hrxy/self.rho[i]

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
            for i in range(pts.ntotal):
                pts.rho[i] += 0.5*dt*drhodt[i]
                pts.v[i, :] += 0.5*dt*dvdt[i, :]
                pts.stress_update(i, 0.5*dt*dstraindt[i, :], 0.5*dt*rxy[i], sigma0[i, :])

            # initialize material rate arrays
            dvdt = np.tile(self.f, (pts.ntotal+pts.nvirt, 1))
            drhodt = np.zeros(pts.ntotal+pts.nvirt)
            dstraindt = np.zeros((pts.ntotal+pts.nvirt, 4))
            rxy = np.zeros(pts.ntotal+pts.nvirt)
            
            # perform sweep of pairs
            pts.pair_sweep(dvdt, drhodt, dstraindt, rxy, self.kernel)

            # update data to full-timestep
            for i in range(pts.ntotal):
                pts.rho[i] = rho0[i] + dt*drhodt[i]
                pts.v[i, :] = v0[i, :] + dt*dvdt[i, :]
                pts.x[i, :] += dt*pts.v[i, :]
                pts.stress_update(i, dt*dstraindt[i, :], dt*rxy[i], sigma0[i, :])
                pts.strain[i, :] += dt*dstraindt[i, :]

            # print data to terminal if needed
            if itimestep % self.printtimestep == 0:
                print(f'time-step: {itimestep}')
            
            # save data to disk if needed
            if itimestep % self.savetimestep == 0:
                pts.save_data(itimestep)