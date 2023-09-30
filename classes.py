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
    def stress_update(self, dstrain: np.ndarray, drxy: np.ndarray, sigma0: np.ndarray):

        # cache some values
        DE = self.customvals['DE'][:, :]
        k_c = self.customvals['k_c']
        alpha_phi = self.customvals['alpha_phi']
        alpha_psi = self.customvals['alpha_psi']
        sigma = self.sigma # this stores reference, not the data.
        ntotal = self.ntotal

        npdot = np.dot
        npmatmul = np.matmul
        npsqrt = np.sqrt

        # elastic predictor stress increment
        dsig = npmatmul(dstrain[0:ntotal, :], DE[:, :])
        # update stress increment with Jaumann stress-rate
        dsig[:, 3] += sigma0[0:ntotal, 0]*drxy[0:ntotal] - sigma0[0:ntotal, 1]*drxy[0:ntotal]

        # update stress state
        self.sigma[0:ntotal] = sigma0[0:ntotal, :] + dsig[:, :]

        # define identity and D2 matrisices (Voigt notation)
        Ide = np.ascontiguousarray([1, 1, 1, 0])
        D2 = np.ascontiguousarray([1, 1, 1, 2])

        s = np.zeros(4)

        for i in range(ntotal):

            # stress invariants
            I1 = sigma[i, 0] + sigma[i, 1] + sigma[i, 2]
            s[0:3] = sigma[i, 0:3] - I1/3.
            s[3] = sigma[i, 3]
            J2 = 0.5*npdot(s[:], D2[:]*s[:])

            # tensile cracking check 1: 
            # J2 is zero but I1 is beyond apex of yield surface
            if J2 == 0 and I1 > k_c:
                I1 = k_c
                sigma[i, 0:3] = I1/3.
                sigma[i, 3] = 0.

            # calculate yield function
            f = alpha_phi*I1 + npsqrt(J2) - k_c

            # Perform corrector step
            if f > 0.:
                dfdsig = alpha_phi*Ide[:] + s[:]/(2.*npsqrt(J2))
                dgdsig = alpha_psi*Ide[:] + s[:]/(2.*npsqrt(J2))

                dlambda = f/(npdot(dfdsig[:], D2[:]*npmatmul(DE[:, :], dgdsig[:])))

                sigma[i, :] -= npmatmul(DE[:, :], dlambda*dgdsig[:])

            # tensile cracking check 2:
            # corrected stress state is outside yield surface
            I1 = sigma[i, 0] + sigma[i, 1] + sigma[i, 2]
            if I1 > k_c/alpha_phi:
                sigma[i, 0:3] = k_c/alpha_phi/3
                sigma[i, 3] = 0.

        # simple fluid equation of state.
        # for i in range(self.ntotal+self.nvirt):
        #     p = self.c*self.c*(self.rho[i] - self.rho_ini)
        #     self.sigma[i, 0:3] = -p/3.
        #     self.sigma[i, 3] = 0.

    # function to update self.pairs - list of particle pairs
    def findpairs(self):

        tree = sp.spatial.cKDTree(self.x[0:self.ntotal+self.nvirt, :])
        pairs = tree.query_pairs(3*self.dx, output_type='ndarray')
        
        # trim pairs to only consider real-real or real-virtual
        nonvirtvirt_mask = np.logical_or(
            self.type[pairs[:, 0]] > 0,
            self.type[pairs[:, 1]] > 0
        )
        self.pair_i, self.pair_j = pairs[nonvirtvirt_mask, 0], pairs[nonvirtvirt_mask, 1]

    def update_virtualparticle_properties(self, kernel: typing.Type):

        # zeroing virutal particles' properties
        self.v[self.ntotal:self.ntotal+self.nvirt, :].fill(0.)
        self.rho[self.ntotal:self.ntotal+self.nvirt].fill(0.)
        self.sigma[self.ntotal:self.ntotal+self.nvirt, :].fill(0.)
        vw = np.zeros(self.ntotal+self.nvirt, dtype=np.float64)

        def update_virti(pair_i, pair_j):
            if pair_i.shape[0] > 0:
                r = np.linalg.norm(self.x[pair_i, :] - self.x[pair_j, :], axis=1)
                w = np.apply_along_axis(kernel.w, 0, r)
                dvol = self.mass/self.rho[pair_j]
                np.add.at(vw, pair_i, w[:]*dvol[:])
                np.subtract.at(self.v[:, 0], pair_i, self.v[pair_j, 0]*w*dvol[:])
                np.subtract.at(self.v[:, 1], pair_i, self.v[pair_j, 1]*w*dvol[:])
                np.add.at(self.rho, pair_i, self.mass*w)
                np.add.at(self.sigma[:, 0], pair_i, self.sigma[pair_j, 0]*w*dvol[:])
                np.add.at(self.sigma[:, 1], pair_i, self.sigma[pair_j, 1]*w*dvol[:])
                np.add.at(self.sigma[:, 2], pair_i, self.sigma[pair_j, 2]*w*dvol[:])
                np.add.at(self.sigma[:, 3], pair_i, self.sigma[pair_j, 3]*w*dvol[:])

        # sweep over all pairs and update virtual particles' properties
        # only consider real-virtual pairs
        nonvirtvirt_mask = np.logical_and(
            self.type[self.pair_i] < 0,
            self.type[self.pair_j] > 0
        )
        pair_i = self.pair_i[nonvirtvirt_mask]
        pair_j = self.pair_j[nonvirtvirt_mask]

        update_virti(pair_i, pair_j)

        nonvirtvirt_mask = np.logical_and(
            self.type[self.pair_i] > 0,
            self.type[self.pair_j] < 0
        )
        pair_i = self.pair_i[nonvirtvirt_mask]
        pair_j = self.pair_j[nonvirtvirt_mask]

        update_virti(pair_j, pair_i)

        # normalize virtual particle properties with summed kernels
        vw_mask = vw[self.ntotal:self.ntotal+self.nvirt]>0.
        self.v[self.ntotal:self.ntotal+self.nvirt, 0] = np.divide(self.v[self.ntotal:self.ntotal+self.nvirt, 0], vw[self.ntotal:self.ntotal+self.nvirt], where=vw_mask)
        self.v[self.ntotal:self.ntotal+self.nvirt, 1] = np.divide(self.v[self.ntotal:self.ntotal+self.nvirt, 1], vw[self.ntotal:self.ntotal+self.nvirt], where=vw_mask)
        self.sigma[self.ntotal:self.ntotal+self.nvirt, 0] = np.divide(self.sigma[self.ntotal:self.ntotal+self.nvirt, 0], vw[self.ntotal:self.ntotal+self.nvirt], where=vw_mask)
        self.sigma[self.ntotal:self.ntotal+self.nvirt, 1] = np.divide(self.sigma[self.ntotal:self.ntotal+self.nvirt, 1], vw[self.ntotal:self.ntotal+self.nvirt], where=vw_mask)
        self.sigma[self.ntotal:self.ntotal+self.nvirt, 2] = np.divide(self.sigma[self.ntotal:self.ntotal+self.nvirt, 2], vw[self.ntotal:self.ntotal+self.nvirt], where=vw_mask)
        self.sigma[self.ntotal:self.ntotal+self.nvirt, 3] = np.divide(self.sigma[self.ntotal:self.ntotal+self.nvirt, 3], vw[self.ntotal:self.ntotal+self.nvirt], where=vw_mask)
        self.rho[self.ntotal:self.ntotal+self.nvirt] = np.divide(self.rho[self.ntotal:self.ntotal+self.nvirt], vw[self.ntotal:self.ntotal+self.nvirt], where=vw_mask)
        self.rho[self.ntotal:self.ntotal+self.nvirt] = np.where(vw_mask, self.rho[self.ntotal:self.ntotal+self.nvirt], self.rho_ini)

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
        pair_i = self.pair_i
        pair_j = self.pair_j

        # calculate differential position vector and kernel gradient
        dx = self.x[pair_i, :] - self.x[pair_j, :]
        dv = self.v[pair_i, :] - self.v[pair_j, :]
        dwdx = kernel.dwdx(dx)

        # update acceleration with artificial viscosity
        vr = np.einsum("ij,ij->i", dv, dx)
        vr = np.where(vr > 0, 0., vr)
        rr = np.einsum("ij,ij->i", dx, dx)
        muv = self.h*vr[:]/(rr[:] + self.h*self.h*0.01)
        mrho = 0.5*(self.rho[pair_i]+self.rho[pair_j])
        piv = -np.einsum("i,ij->ij", 
            self.mass*0.2*(muv[:]-self.c)*muv[:]/mrho[:],
            dwdx)

        np.add.at(dvdt[:, 0], pair_i, piv[:, 0])
        np.add.at(dvdt[:, 1], pair_i, piv[:, 1])
        np.subtract.at(dvdt[:, 0], pair_j, piv[:, 0])
        np.subtract.at(dvdt[:, 1], pair_j, piv[:, 1])

        # update acceleration with div stress
        # using momentum consertive form
        sigma_rho2 = np.einsum("ij,i->ij", self.sigma, 1./self.rho**2)
        sigma_rho2_pairs = sigma_rho2[pair_i, :]+sigma_rho2[pair_j, :]
        h = sigma_rho2_pairs[:, 0:2] * dwdx[:, 0:2]
        h += np.einsum("i,ij->ij", sigma_rho2_pairs[:, 3], np.fliplr(dwdx[:, 0:2]))
        h *= self.mass

        np.add.at(dvdt[:, 0], pair_i, h[:, 0])
        np.add.at(dvdt[:, 1], pair_i, h[:, 1])
        np.subtract.at(dvdt[:, 0], pair_j, h[:, 0])
        np.subtract.at(dvdt[:, 1], pair_j, h[:, 1])

        # update density change rate with continuity density
        drhodt_pairs = self.mass*np.einsum("ij,ij->i", dv, dwdx)

        np.add.at(drhodt, pair_i, drhodt_pairs)
        np.add.at(drhodt, pair_j, drhodt_pairs)

        # calculating engineering strain rates
        he = np.zeros((pair_i.shape[0], 4))
        he[:, 0:2] = -dv[:, 0:2]*dwdx[:, 0:2]
        he[:, 3] = -0.5*np.einsum("ij,ij->i", dv[:, 0:2], np.fliplr(dwdx[:, 0:2]))
        he[:, :] *= self.mass
        hrxy = -self.mass*0.5*(dv[:, 0]*dwdx[:, 1] - dv[:, 1]*dwdx[:, 0])

        np.add.at(dstraindt[:, 0], pair_i, he[:, 0]/self.rho[pair_j])
        np.add.at(dstraindt[:, 1], pair_i, he[:, 1]/self.rho[pair_j])
        #np.add.at(dstraindt[:, 2], pair_i, he[:, 2]/self.rho[pair_j])
        np.add.at(dstraindt[:, 3], pair_i, he[:, 3]/self.rho[pair_j])
        np.add.at(dstraindt[:, 0], pair_j, he[:, 0]/self.rho[pair_i])
        np.add.at(dstraindt[:, 1], pair_j, he[:, 1]/self.rho[pair_i])
        #np.add.at(dstraindt[:, 2], pair_j, he[:, 2]/self.rho[pair_i])
        np.add.at(dstraindt[:, 3], pair_j, he[:, 3]/self.rho[pair_i])

        np.add.at(rxy, pair_i, hrxy/self.rho[pair_j])
        np.add.at(rxy, pair_j, hrxy/self.rho[pair_i])

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
            pts.stress_update(0.5*dt*dstraindt, 0.5*dt*rxy, sigma0)

            # initialize material rate arrays
            dvdt = np.tile(self.f, (pts.ntotal+pts.nvirt, 1))
            drhodt = np.zeros(pts.ntotal+pts.nvirt)
            dstraindt = np.zeros((pts.ntotal+pts.nvirt, 4))
            rxy = np.zeros(pts.ntotal+pts.nvirt)
            
            # perform sweep of pairs
            pts.pair_sweep(dvdt, drhodt, dstraindt, rxy, self.kernel)

            pts.stress_update(dt*dstraindt, dt*rxy, sigma0)

            # update data to full-timestep
            for i in range(pts.ntotal):
                pts.rho[i] = rho0[i] + dt*drhodt[i]
                pts.v[i, :] = v0[i, :] + dt*dvdt[i, :]
                pts.x[i, :] += dt*pts.v[i, :]
                # pts.stress_update(i, dt*dstraindt[i, :], dt*rxy[i], sigma0[i, :])
                pts.strain[i, :] += dt*dstraindt[i, :]

            # print data to terminal if needed
            if itimestep % self.printtimestep == 0:
                print(f'time-step: {itimestep}')
            
            # save data to disk if needed
            if itimestep % self.savetimestep == 0:
                pts.save_data(itimestep)