import numpy as np
import typing
import h5py
from closefriends import query_pairs

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

        # cache some references
        DE = self.customvals['DE'][:, :]
        k_c = self.customvals['k_c']
        alpha_phi = self.customvals['alpha_phi']
        alpha_psi = self.customvals['alpha_psi']
        sigma = self.sigma # this stores reference, not the data.
        ntotal = self.ntotal
        nvirt = self.nvirt
        realmask = self.type[0:ntotal+nvirt] > 0

        npmatmul = np.matmul
        npsqrt = np.sqrt

        # elastic predictor stress increment
        dsig = npmatmul(dstrain[0:ntotal+nvirt, :], DE[:, :])
        # update stress increment with Jaumann stress-rate
        dsig[:, 3] += sigma0[0:ntotal+nvirt, 0]*drxy[0:ntotal+nvirt] - sigma0[0:ntotal+nvirt, 1]*drxy[0:ntotal+nvirt]

        # update stress state
        np.add(sigma0[0:ntotal+nvirt, :], dsig[:, :], out=sigma[0:ntotal+nvirt, :], where=realmask[:, np.newaxis])

        # define identity and D2 matrisices (Voigt notation)
        Ide = np.ascontiguousarray([1, 1, 1, 0])
        D2 = np.ascontiguousarray([1, 1, 1, 2])

        # stress invariants
        I1 = np.sum(sigma[0:ntotal+nvirt, 0:3], axis=1)
        s = sigma[0:ntotal+nvirt, :] - np.outer(I1[:], Ide[:]/3.)
        J2 = 0.5*np.einsum("ij,ij->i", s, s*D2)

        # tensile cracking check 1: 
        # J2 is zero but I1 is beyond apex of yield surface
        tensile_crack_check1_mask = np.logical_and(J2==0., I1 > k_c)

        I1 = np.where(tensile_crack_check1_mask, k_c, I1)
        sigma[0:ntotal+nvirt, 0:3] = np.where(
            tensile_crack_check1_mask[:, np.newaxis],
            k_c/3., 
            sigma[0:ntotal+nvirt, 0:3]
        )
        sigma[0:ntotal+nvirt, 3] = np.where(tensile_crack_check1_mask, 0., sigma[0:ntotal+nvirt, 3])

        # calculate yield function
        f = alpha_phi*I1 + npsqrt(J2) - k_c

        ## Start performing corrector step.
        # Calculate mask where stress state is outside yield surface
        f_mask = f > 0.
        
        # normalize deviatoric stress tensor by its frobenius norm/2 (only for pts with f > 0)
        shat = s.copy()
        np.divide(
            shat,
            2.*npsqrt(J2)[:, np.newaxis], 
            out=shat,
            where=f_mask[:, np.newaxis],
        )

        # update yield potential and plastic potential matrices with normalized deviatoric stress tensor
        dfdsig = np.add(alpha_phi*Ide[np.newaxis, :], shat, where=f_mask[:, np.newaxis])
        dgdsig = np.add(alpha_psi*Ide[np.newaxis, :], shat, where=f_mask[:, np.newaxis])

        # calculate plastic multipler
        dlambda = np.divide(
            f,
            np.einsum("ij,ij->i", dfdsig, (D2[:]*npmatmul(dgdsig[:, :], DE[:, :]))),
            where=f_mask
        )

        # Apply plastic corrector stress
        np.subtract(
            sigma[0:ntotal+nvirt, :], 
            npmatmul(dlambda[:, np.newaxis]*dgdsig[:, :], DE[:, :]), 
            out=sigma[0:ntotal+nvirt, :],
            where=f_mask[:, np.newaxis]
        )

        ## tensile cracking check 2:
        # corrected stress state is outside yield surface
        I1 = np.sum(sigma[0:ntotal+nvirt, 0:3], axis=1)
        tensile_crack_check2_mask = I1 > k_c/alpha_phi
        sigma[0:ntotal+nvirt, 0:3] = np.where(
            tensile_crack_check2_mask[:, np.newaxis],
            k_c/alpha_phi/3, 
            sigma[0:ntotal+nvirt, 0:3]
        )
        sigma[0:ntotal+nvirt, 3] = np.where(tensile_crack_check2_mask, 0., sigma[0:ntotal+nvirt, 3])

        # simple fluid equation of state.
        # for i in range(self.ntotal+self.nvirt):
        #     p = self.c*self.c*(self.rho[i] - self.rho_ini)
        #     self.sigma[i, 0:3] = -p/3.
        #     self.sigma[i, 3] = 0.

    # function to update self.pairs - list of particle pairs
    def findpairs(self):

        # find pairs using closefriends.query_pairs
        pair_i, pair_j = query_pairs(self.x[0:self.ntotal+self.nvirt, :], 3*self.dx, 30*(self.ntotal+self.nvirt), retain_order=True)
        
        # trim pairs to only consider real-real or real-virtual
        nonvirtvirt_mask = np.logical_or(
            self.type[pair_i[:]] > 0,
            self.type[pair_j[:]] > 0
        )

        # remove virt-virt pairs with mask
        self.pair_i = pair_i[nonvirtvirt_mask]
        self.pair_j = pair_j[nonvirtvirt_mask]


    def update_virtualparticle_properties(self, kernel: typing.Type):

        # cache some references
        ntotal = self.ntotal
        nvirt = self.nvirt
        x = self.x
        v = self.v
        sigma = self.sigma
        rho = self.rho
        type = self.type

        virtmask = type[0:ntotal+nvirt] < 0

        # zeroing virutal particles' properties
        v[0:ntotal+nvirt, :] = np.where(virtmask[:, np.newaxis], 0, v[0:ntotal+nvirt, :])
        rho[0:ntotal+nvirt] = np.where(virtmask, 0, rho[0:ntotal+nvirt])
        sigma[0:ntotal+nvirt] = np.where(virtmask[:, np.newaxis], 0, sigma[0:ntotal+nvirt])
        vw = np.zeros(ntotal+nvirt, dtype=np.float64)

        # define function to update virtual particles' properties
        def update_virti(pair_i, pair_j):
            if pair_i.shape[0] > 0:
                r = np.linalg.norm(x[pair_i, :] - x[pair_j, :], axis=1)
                w = np.apply_along_axis(kernel.w, 0, r)
                dvol = self.mass/rho[pair_j]
                np.add.at(vw, pair_i, w[:]*dvol[:])
                np.subtract.at(v[:, 0], pair_i, v[pair_j, 0]*w*dvol[:])
                np.subtract.at(v[:, 1], pair_i, v[pair_j, 1]*w*dvol[:])
                np.add.at(rho, pair_i, self.mass*w)
                np.add.at(sigma[:, 0], pair_i, sigma[pair_j, 0]*w*dvol[:])
                np.add.at(sigma[:, 1], pair_i, sigma[pair_j, 1]*w*dvol[:])
                np.add.at(sigma[:, 2], pair_i, sigma[pair_j, 2]*w*dvol[:])
                np.add.at(sigma[:, 3], pair_i, sigma[pair_j, 3]*w*dvol[:])

        # sweep over all pairs and update virtual particles' properties
        # only consider real-virtual pairs
        # i particles are virtual
        nonvirtvirt_mask = np.logical_and(
            type[self.pair_i] < 0,
            type[self.pair_j] > 0
        )
        pair_i = self.pair_i[nonvirtvirt_mask]
        pair_j = self.pair_j[nonvirtvirt_mask]

        update_virti(pair_i, pair_j)

        # j particles are virtual
        nonvirtvirt_mask = np.logical_and(
            type[self.pair_i] > 0,
            type[self.pair_j] < 0
        )
        pair_i = self.pair_i[nonvirtvirt_mask]
        pair_j = self.pair_j[nonvirtvirt_mask]

        update_virti(pair_j, pair_i)

        # normalize virtual particle properties with summed kernels
        vw_mask = vw[0:ntotal+nvirt]>0.
        np.divide(v[0:ntotal+nvirt, :], vw[:, np.newaxis], where=vw_mask[:, np.newaxis], out=v[0:ntotal+nvirt, :])
        np.divide(sigma[0:ntotal+nvirt, :], vw[:, np.newaxis], where=vw_mask[:, np.newaxis], out=sigma[0:ntotal+nvirt, :])
        np.divide(rho[0:ntotal+nvirt], vw[:], where=vw_mask[:], out=rho[0:ntotal+nvirt])
        rho[0:ntotal+nvirt] = np.where(vw_mask, rho[0:ntotal+nvirt], self.rho_ini)

    # function to perform sweep over all particle pairs
    def pair_sweep(self, 
                   dvdt: np.ndarray, 
                   drhodt: np.ndarray, 
                   dstraindt: np.ndarray, 
                   rxy: np.ndarray,
                   kernel: typing.Type):

        # cache some references
        pair_i = self.pair_i
        pair_j = self.pair_j
        ntotal = self.ntotal
        nvirt = self.nvirt
        x = self.x
        v = self.v
        rho = self.rho
        sigma = self.sigma
        h = self.h; c = self.c; mass = self.mass

        ## calculate differential position vector and kernel gradient first ----
        dx = x[pair_i, :] - x[pair_j, :]
        dv = v[pair_i, :] - v[pair_j, :]
        dwdx = kernel.dwdx(dx)

        ## update virtual particles' properties --------------------------------
        self.update_virtualparticle_properties(kernel)

        ## sweep over all pairs to update real particles' material rates -------
        # update acceleration with artificial viscosity
        vr = np.einsum("ij,ij->i", dv, dx)
        vr = np.where(vr > 0, 0., vr)
        rr = np.einsum("ij,ij->i", dx, dx)
        muv = h*vr[:]/(rr[:] + h*h*0.01)
        mrho = 0.5*(rho[pair_i]+rho[pair_j])
        piv = -np.einsum("i,ij->ij", 
            mass*0.2*(muv[:]-c)*muv[:]/mrho[:],
            dwdx)

        np.add.at(dvdt[:, 0], pair_i, piv[:, 0])
        np.add.at(dvdt[:, 1], pair_i, piv[:, 1])
        np.subtract.at(dvdt[:, 0], pair_j, piv[:, 0])
        np.subtract.at(dvdt[:, 1], pair_j, piv[:, 1])

        # update acceleration with div stress
        # using momentum consertive form
        sigma_rho2 = sigma[0:ntotal+nvirt, :] / (rho[0:ntotal+nvirt, np.newaxis]**2)
        sigma_rho2_pairs = sigma_rho2[pair_i, :]+sigma_rho2[pair_j, :]
        h = sigma_rho2_pairs[:, 0:2] * dwdx[:, 0:2]
        h += np.einsum("i,ij->ij", sigma_rho2_pairs[:, 3], np.fliplr(dwdx[:, 0:2]))
        h *= mass

        np.add.at(dvdt[:, 0], pair_i, h[:, 0])
        np.add.at(dvdt[:, 1], pair_i, h[:, 1])
        np.subtract.at(dvdt[:, 0], pair_j, h[:, 0])
        np.subtract.at(dvdt[:, 1], pair_j, h[:, 1])

        # update density change rate with continuity density
        drhodt_pairs = mass*np.einsum("ij,ij->i", dv, dwdx)

        np.add.at(drhodt, pair_i, drhodt_pairs)
        np.add.at(drhodt, pair_j, drhodt_pairs)

        # calculating engineering strain rates
        he = np.zeros((pair_i.shape[0], 4))
        he[:, 0:2] = -dv[:, 0:2]*dwdx[:, 0:2]
        he[:, 3] = -0.5*np.einsum("ij,ij->i", dv[:, 0:2], np.fliplr(dwdx[:, 0:2]))
        he[:, :] *= mass
        hrxy = -mass*0.5*(dv[:, 0]*dwdx[:, 1] - dv[:, 1]*dwdx[:, 0])

        np.add.at(dstraindt[:, 0], pair_i, he[:, 0]/rho[pair_j])
        np.add.at(dstraindt[:, 1], pair_i, he[:, 1]/rho[pair_j])
        #np.add.at(dstraindt[:, 2], pair_i, he[:, 2]/rho[pair_j])
        np.add.at(dstraindt[:, 3], pair_i, he[:, 3]/rho[pair_j])
        np.add.at(dstraindt[:, 0], pair_j, he[:, 0]/rho[pair_i])
        np.add.at(dstraindt[:, 1], pair_j, he[:, 1]/rho[pair_i])
        #np.add.at(dstraindt[:, 2], pair_j, he[:, 2]/rho[pair_i])
        np.add.at(dstraindt[:, 3], pair_j, he[:, 3]/rho[pair_i])

        np.add.at(rxy, pair_i, hrxy/rho[pair_j])
        np.add.at(rxy, pair_j, hrxy/rho[pair_i])

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

            realmask = pts.type[0:pts.ntotal+pts.nvirt] > 0

            # save data from start of timestep
            v0 = np.copy(pts.v[0:pts.ntotal+pts.nvirt, :])
            rho0 = np.copy(pts.rho[0:pts.ntotal+pts.nvirt])
            sigma0 = np.copy(pts.sigma[0:pts.ntotal+pts.nvirt, :])

            # Update data to mid-timestep
            pts.rho[0:pts.ntotal+pts.nvirt] += 0.5*dt*drhodt[0:pts.ntotal+pts.nvirt]
            pts.v[0:pts.ntotal+pts.nvirt, :] += 0.5*dt*dvdt[0:pts.ntotal+pts.nvirt, :]

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
            np.add(rho0[0:pts.ntotal+pts.nvirt], dt*drhodt[0:pts.ntotal+pts.nvirt], where=realmask, out=pts.rho[0:pts.ntotal+pts.nvirt])
            np.add(v0[0:pts.ntotal+pts.nvirt, :], dt*dvdt[0:pts.ntotal+pts.nvirt, :], where=realmask[:, np.newaxis], out=pts.v[0:pts.ntotal+pts.nvirt, :])
            np.add(pts.x[0:pts.ntotal+pts.nvirt, :], dt*pts.v[0:pts.ntotal+pts.nvirt, :], where=realmask[:, np.newaxis], out=pts.x[0:pts.ntotal+pts.nvirt, :])
            np.add(pts.strain[0:pts.ntotal+pts.nvirt, :], dt*dstraindt[0:pts.ntotal+pts.nvirt, :], where=realmask[:, np.newaxis], out=pts.strain[0:pts.ntotal+pts.nvirt, :])

            # print data to terminal if needed
            if itimestep % self.printtimestep == 0:
                print(f'time-step: {itimestep}')
            
            # save data to disk if needed
            if itimestep % self.savetimestep == 0:
                pts.save_data(itimestep)