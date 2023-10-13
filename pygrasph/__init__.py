import numpy as _np
import typing as _typing
import h5py as _h5py
from closefriends import query_pairs as _query_pairs
from . import kernels

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
        self.x = _np.zeros((maxn, 2), dtype=_np.float64)
        self.v = _np.zeros((maxn, 2), dtype=_np.float64)
        self.rho = _np.full(maxn, rho_ini, dtype=_np.float64)
        self.id = _np.arange(maxn, dtype=_np.int32)
        self.type = _np.ones(maxn, dtype=_np.int32)
        self.strain = _np.zeros((maxn, 4), dtype=_np.float64)
        self.sigma = _np.zeros((maxn, 4), dtype=_np.float64)

        self.v0 = _np.zeros((maxn, 2), dtype=_np.float64)
        self.rho0 = _np.zeros((maxn, 2), dtype=_np.float64)
        self.sigma0 = _np.zeros((maxn, 4), dtype=_np.float64)

        self.pairs = _np.ndarray((maxinter, 2))

        # custom data in dict
        self.customvals = customvals

    # generate real particles (user to define)
    def generate_real_coords(self):

        pass

    # generate virtual particles (user to define)
    def generate_virt_coords(self):

        pass

    # stress update function (DP model)
    def stress_update(self, dstrain: _np.ndarray, drxy: _np.ndarray, sigma0: _np.ndarray):

        # cache some references
        DE = self.customvals['DE'][:, :]
        k_c = self.customvals['k_c']
        alpha_phi = self.customvals['alpha_phi']
        alpha_psi = self.customvals['alpha_psi']
        sigma = self.sigma # this stores reference, not the data.
        ntotal = self.ntotal
        nvirt = self.nvirt
        realmask = self.type[0:ntotal+nvirt] > 0

        npmatmul = _np.matmul
        npsqrt = _np.sqrt

        # elastic predictor stress increment
        dsig = npmatmul(dstrain[0:ntotal+nvirt, :], DE[:, :])
        # update stress increment with Jaumann stress-rate
        dsig[:, 3] += sigma0[0:ntotal+nvirt, 0]*drxy[0:ntotal+nvirt] - sigma0[0:ntotal+nvirt, 1]*drxy[0:ntotal+nvirt]

        # update stress state
        _np.add(sigma0[0:ntotal+nvirt, :], dsig[:, :], out=sigma[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis])

        # define identity and D2 matrisices (Voigt notation)
        Ide = _np.ascontiguousarray([1, 1, 1, 0])
        D2 = _np.ascontiguousarray([1, 1, 1, 2])

        # stress invariants
        I1 = _np.sum(sigma[0:ntotal+nvirt, 0:3], axis=1)
        s = sigma[0:ntotal+nvirt, :] - _np.outer(I1[:], Ide[:]/3.)
        J2 = 0.5*_np.einsum("ij,ij->i", s, s*D2)

        # tensile cracking check 1: 
        # J2 is zero but I1 is beyond apex of yield surface
        tensile_crack_check1_mask = _np.logical_and(J2==0., I1 > k_c)

        I1 = _np.where(tensile_crack_check1_mask, k_c, I1)
        sigma[0:ntotal+nvirt, 0:3] = _np.where(
            tensile_crack_check1_mask[:, _np.newaxis],
            k_c/3., 
            sigma[0:ntotal+nvirt, 0:3]
        )
        sigma[0:ntotal+nvirt, 3] = _np.where(tensile_crack_check1_mask, 0., sigma[0:ntotal+nvirt, 3])

        # calculate yield function
        f = alpha_phi*I1 + npsqrt(J2) - k_c

        ## Start performing corrector step.
        # Calculate mask where stress state is outside yield surface
        f_mask = f > 0.
        
        # normalize deviatoric stress tensor by its frobenius norm/2 (only for pts with f > 0)
        shat = s.copy()
        _np.divide(
            shat,
            2.*npsqrt(J2)[:, _np.newaxis], 
            out=shat,
            where=f_mask[:, _np.newaxis],
        )

        # update yield potential and plastic potential matrices with normalized deviatoric stress tensor
        dfdsig = _np.add(alpha_phi*Ide[_np.newaxis, :], shat, where=f_mask[:, _np.newaxis])
        dgdsig = _np.add(alpha_psi*Ide[_np.newaxis, :], shat, where=f_mask[:, _np.newaxis])

        # calculate plastic multipler
        dlambda = _np.divide(
            f,
            _np.einsum("ij,ij->i", dfdsig, (D2[:]*npmatmul(dgdsig[:, :], DE[:, :]))),
            where=f_mask
        )

        # Apply plastic corrector stress
        _np.subtract(
            sigma[0:ntotal+nvirt, :], 
            npmatmul(dlambda[:, _np.newaxis]*dgdsig[:, :], DE[:, :]), 
            out=sigma[0:ntotal+nvirt, :],
            where=f_mask[:, _np.newaxis]
        )

        ## tensile cracking check 2:
        # corrected stress state is outside yield surface
        I1 = _np.sum(sigma[0:ntotal+nvirt, 0:3], axis=1)
        tensile_crack_check2_mask = I1 > k_c/alpha_phi
        sigma[0:ntotal+nvirt, 0:3] = _np.where(
            tensile_crack_check2_mask[:, _np.newaxis],
            k_c/alpha_phi/3, 
            sigma[0:ntotal+nvirt, 0:3]
        )
        sigma[0:ntotal+nvirt, 3] = _np.where(tensile_crack_check2_mask, 0., sigma[0:ntotal+nvirt, 3])

        # simple fluid equation of state.
        # for i in range(self.ntotal+self.nvirt):
        #     p = self.c*self.c*(self.rho[i] - self.rho_ini)
        #     self.sigma[i, 0:3] = -p/3.
        #     self.sigma[i, 3] = 0.

    # function to update self.pairs - list of particle pairs
    def findpairs(self):

        # find pairs using closefriends.query_pairs
        pair_i, pair_j, idx = _query_pairs(self.x[0:self.ntotal+self.nvirt, :], 3*self.dx, 30*(self.ntotal+self.nvirt))

        self.v = self.v[idx, :]
        self.rho = self.rho[idx]
        self.id = self.id[idx]
        self.type = self.type[idx]
        self.strain = self.strain[idx, :]
        self.sigma = self.sigma[idx, :]
        self.v0 = self.v0[idx, :]
        self.rho0 = self.rho0[idx]
        self.sigma0 = self.sigma0[idx, :]
        
        # trim pairs to only consider real-real or real-virtual
        nonvirtvirt_mask = _np.logical_or(
            self.type[pair_i[:]] > 0,
            self.type[pair_j[:]] > 0
        )

        # remove virt-virt pairs with mask
        self.pair_i = pair_i[nonvirtvirt_mask]
        self.pair_j = pair_j[nonvirtvirt_mask]


    def update_virtualparticle_properties(self, kernel: _typing.Type):

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
        v[0:ntotal+nvirt, :] = _np.where(virtmask[:, _np.newaxis], 0, v[0:ntotal+nvirt, :])
        rho[0:ntotal+nvirt] = _np.where(virtmask, 0, rho[0:ntotal+nvirt])
        sigma[0:ntotal+nvirt] = _np.where(virtmask[:, _np.newaxis], 0, sigma[0:ntotal+nvirt])
        vw = _np.zeros(ntotal+nvirt, dtype=_np.float64)

        # define function to update virtual particles' properties
        def update_virti(pair_i, pair_j):
            if pair_i.shape[0] > 0:
                r = _np.linalg.norm(x[pair_i, :] - x[pair_j, :], axis=1)
                w = _np.apply_along_axis(kernel.w, 0, r)
                dvol = self.mass/rho[pair_j]
                _np.add.at(vw, pair_i, w[:]*dvol[:])
                _np.subtract.at(v[:, 0], pair_i, v[pair_j, 0]*w*dvol[:])
                _np.subtract.at(v[:, 1], pair_i, v[pair_j, 1]*w*dvol[:])
                _np.add.at(rho, pair_i, self.mass*w)
                _np.add.at(sigma[:, 0], pair_i, sigma[pair_j, 0]*w*dvol[:])
                _np.add.at(sigma[:, 1], pair_i, sigma[pair_j, 1]*w*dvol[:])
                _np.add.at(sigma[:, 2], pair_i, sigma[pair_j, 2]*w*dvol[:])
                _np.add.at(sigma[:, 3], pair_i, sigma[pair_j, 3]*w*dvol[:])

        # sweep over all pairs and update virtual particles' properties
        # only consider real-virtual pairs
        # i particles are virtual
        nonvirtvirt_mask = _np.logical_and(
            type[self.pair_i] < 0,
            type[self.pair_j] > 0
        )
        pair_i = self.pair_i[nonvirtvirt_mask]
        pair_j = self.pair_j[nonvirtvirt_mask]

        update_virti(pair_i, pair_j)

        # j particles are virtual
        nonvirtvirt_mask = _np.logical_and(
            type[self.pair_i] > 0,
            type[self.pair_j] < 0
        )
        pair_i = self.pair_i[nonvirtvirt_mask]
        pair_j = self.pair_j[nonvirtvirt_mask]

        update_virti(pair_j, pair_i)

        # normalize virtual particle properties with summed kernels
        vw_mask = vw[0:ntotal+nvirt]>0.
        _np.divide(v[0:ntotal+nvirt, :], vw[:, _np.newaxis], where=vw_mask[:, _np.newaxis], out=v[0:ntotal+nvirt, :])
        _np.divide(sigma[0:ntotal+nvirt, :], vw[:, _np.newaxis], where=vw_mask[:, _np.newaxis], out=sigma[0:ntotal+nvirt, :])
        _np.divide(rho[0:ntotal+nvirt], vw[:], where=vw_mask[:], out=rho[0:ntotal+nvirt])
        rho[0:ntotal+nvirt] = _np.where(vw_mask, rho[0:ntotal+nvirt], self.rho_ini)

    # function to perform sweep over all particle pairs
    def pair_sweep(self, 
                   dvdt: _np.ndarray, 
                   drhodt: _np.ndarray, 
                   dstraindt: _np.ndarray, 
                   rxy: _np.ndarray,
                   kernel: _typing.Type):

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
        vr = _np.einsum("ij,ij->i", dv, dx)
        vr = _np.where(vr > 0, 0., vr)
        rr = _np.einsum("ij,ij->i", dx, dx)
        muv = h*vr[:]/(rr[:] + h*h*0.01)
        mrho = 0.5*(rho[pair_i]+rho[pair_j])
        piv = -_np.einsum("i,ij->ij", 
            mass*0.2*(muv[:]-c)*muv[:]/mrho[:],
            dwdx)

        _np.add.at(dvdt[:, 0], pair_i, piv[:, 0])
        _np.add.at(dvdt[:, 1], pair_i, piv[:, 1])
        _np.subtract.at(dvdt[:, 0], pair_j, piv[:, 0])
        _np.subtract.at(dvdt[:, 1], pair_j, piv[:, 1])

        # update acceleration with div stress
        # using momentum consertive form
        sigma_rho2 = sigma[0:ntotal+nvirt, :] / (rho[0:ntotal+nvirt, _np.newaxis]**2)
        sigma_rho2_pairs = sigma_rho2[pair_i, :]+sigma_rho2[pair_j, :]
        h = sigma_rho2_pairs[:, 0:2] * dwdx[:, 0:2]
        h += _np.einsum("i,ij->ij", sigma_rho2_pairs[:, 3], _np.fliplr(dwdx[:, 0:2]))
        h *= mass

        _np.add.at(dvdt[:, 0], pair_i, h[:, 0])
        _np.add.at(dvdt[:, 1], pair_i, h[:, 1])
        _np.subtract.at(dvdt[:, 0], pair_j, h[:, 0])
        _np.subtract.at(dvdt[:, 1], pair_j, h[:, 1])

        # update density change rate with continuity density
        drhodt_pairs = mass*_np.einsum("ij,ij->i", dv, dwdx)

        _np.add.at(drhodt, pair_i, drhodt_pairs)
        _np.add.at(drhodt, pair_j, drhodt_pairs)

        # calculating engineering strain rates
        he = _np.zeros((pair_i.shape[0], 4))
        he[:, 0:2] = -dv[:, 0:2]*dwdx[:, 0:2]
        he[:, 3] = -0.5*_np.einsum("ij,ij->i", dv[:, 0:2], _np.fliplr(dwdx[:, 0:2]))
        he[:, :] *= mass
        hrxy = -mass*0.5*(dv[:, 0]*dwdx[:, 1] - dv[:, 1]*dwdx[:, 0])

        _np.add.at(dstraindt[:, 0], pair_i, he[:, 0]/rho[pair_j])
        _np.add.at(dstraindt[:, 1], pair_i, he[:, 1]/rho[pair_j])
        #_np.add.at(dstraindt[:, 2], pair_i, he[:, 2]/rho[pair_j])
        _np.add.at(dstraindt[:, 3], pair_i, he[:, 3]/rho[pair_j])
        _np.add.at(dstraindt[:, 0], pair_j, he[:, 0]/rho[pair_i])
        _np.add.at(dstraindt[:, 1], pair_j, he[:, 1]/rho[pair_i])
        #_np.add.at(dstraindt[:, 2], pair_j, he[:, 2]/rho[pair_i])
        _np.add.at(dstraindt[:, 3], pair_j, he[:, 3]/rho[pair_i])

        _np.add.at(rxy, pair_i, hrxy/rho[pair_j])
        _np.add.at(rxy, pair_j, hrxy/rho[pair_i])

    # function to save particle data
    def save_data(self, itimestep: int):

        with _h5py.File(f'output/sph_{itimestep}.h5', 'w') as f:
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
                 f: _np.ndarray,
                 kernel: _typing.Type):
        self.f = f     # body force vector e.g. gravity
        self.kernel = kernel # kernel function of choice

    def LF(self, pts: particles,
           maxtimestep: int, # timestep to run simulation for
           savetimestep: int, # timestep interval to save data to disk
           printtimestep: int, # timestep interval to print timestep
           cfl: float): # Courant-Freidrichs-Lewy coefficient for time-step size

        # timestep size (s)
        dt = cfl*pts.dx*3./pts.c

        # cache some references
        ntotal = pts.ntotal
        nvirt = pts.nvirt

        # begin time integration loop
        for itimestep in range(1, maxtimestep+1):

            # save data from start of timestep
            pts.v0 = _np.copy(pts.v[0:ntotal+nvirt, :])
            pts.rho0 = _np.copy(pts.rho[0:ntotal+nvirt])
            pts.sigma0 = _np.copy(pts.sigma[0:ntotal+nvirt, :])

            # Update data to mid-timestep
            pts.rho[0:ntotal+nvirt] += 0.5*dt*drhodt[0:ntotal+nvirt]
            pts.v[0:ntotal+nvirt, :] += 0.5*dt*dvdt[0:ntotal+nvirt, :]

            pts.stress_update(0.5*dt*dstraindt, 0.5*dt*rxy, pts.sigma0)

            # find pairs
            pts.findpairs()

            realmask = pts.type[0:ntotal+nvirt] > 0

            # initialize material rate arrays
            dvdt = _np.tile(self.f, (ntotal+nvirt, 1)) # acceleration
            drhodt = _np.zeros(ntotal+nvirt) # density change rate
            dstraindt = _np.zeros((ntotal+nvirt, 4)) # strain rate
            rxy = _np.zeros(ntotal+nvirt) # spin rate (for jaumann stress-rate)
            
            # perform sweep of pairs
            pts.pair_sweep(dvdt, drhodt, dstraindt, rxy, self.kernel)

            pts.stress_update(dt*dstraindt, dt*rxy, pts.sigma0)

            # update data to full-timestep
            _np.add(pts.rho0[0:ntotal+nvirt], dt*drhodt[0:ntotal+nvirt], where=realmask, out=pts.rho[0:ntotal+nvirt])
            _np.add(pts.v0[0:ntotal+nvirt, :], dt*dvdt[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis], out=pts.v[0:ntotal+nvirt, :])
            _np.add(pts.x[0:ntotal+nvirt, :], dt*pts.v[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis], out=pts.x[0:ntotal+nvirt, :])
            _np.add(pts.strain[0:ntotal+nvirt, :], dt*dstraindt[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis], out=pts.strain[0:ntotal+nvirt, :])

            # print data to terminal if needed
            if itimestep % printtimestep == 0:
                print(f'time-step: {itimestep}')
            
            # save data to disk if needed
            if itimestep % savetimestep == 0:
                pts.save_data(itimestep)

    def RK4(self, pts: particles,
           maxtimestep: int, # timestep to run simulation for
           savetimestep: int, # timestep interval to save data to disk
           printtimestep: int, # timestep interval to print timestep
           cfl: float): # Courant-Freidrichs-Lewy coefficient for time-step size

        # timestep size (s)
        dt = cfl*pts.dx*3./pts.c

        RK4_weights = _np.array([1., 2., 2., 1.])

        # cache some references
        ntotal = pts.ntotal
        nvirt = pts.nvirt

        # begin time integration loop
        for itimestep in range(1, maxtimestep+1):

            # initialize arrays needed for time integration
            dvdt = _np.tile(self.f, (4, ntotal+nvirt, 1)) # acceleration
            drhodt = _np.zeros((4, ntotal+nvirt), dtype=_np.float64) # density change rate
            dstraindt = _np.zeros((4, ntotal+nvirt, 4), dtype=_np.float64) # strain rate
            rxy = _np.zeros((4, ntotal+nvirt), dtype=_np.float64) # spin rate (for jaumann stress-rate)

            # save data from start of timestep
            pts.v0 = _np.copy(pts.v[0:ntotal+nvirt, :])
            pts.rho0 = _np.copy(pts.rho[0:ntotal+nvirt])
            pts.sigma0 = _np.copy(pts.sigma[0:ntotal+nvirt, :])

            # find pairs
            pts.findpairs()

            realmask = pts.type[0:ntotal+nvirt] > 0

            # k1 ---------------------------------------------------------------
            
            # perform first sweep of pairs (k1)
            pts.pair_sweep(dvdt[0, :, :], drhodt[0, :], dstraindt[0, :, :], rxy[0, :], self.kernel)

            # perform 2nd - 4th RK4 iteration
            for k in range(1, 4):

                # update properties according to RK4_weights
                _np.add(
                    pts.rho0[0:ntotal+nvirt], 
                    dt/RK4_weights[k]*drhodt[k-1, 0:ntotal+nvirt], 
                    where=realmask, 
                    out=pts.rho[0:ntotal+nvirt]
                )

                _np.add(
                    pts.v0[0:ntotal+nvirt, :], 
                    dt/RK4_weights[k]*dvdt[k-1, 0:ntotal+nvirt, :], 
                    where=realmask[:, _np.newaxis], 
                    out=pts.v[0:ntotal+nvirt, :]
                )

                # update stress according to RK4_weights
                pts.stress_update(dt/RK4_weights[k]*dstraindt[k-1, :, :], dt/RK4_weights[k]*rxy[k-1, :], pts.sigma0)

                # perform sweep of pairs
                pts.pair_sweep(dvdt[k, :, :], drhodt[k, :], dstraindt[k, :, :], rxy[k, :], self.kernel)

            # final update -----------------------------------------------------

            drdhot_tot = _np.einsum("i,ij->j", RK4_weights/6., drhodt[:, 0:ntotal+nvirt])
            dvdt_tot = _np.einsum("i,ijk->jk", RK4_weights/6., dvdt[:, 0:ntotal+nvirt, :])
            dstraindt_tot = _np.einsum("i,ijk->jk", RK4_weights/6., dstraindt[:, 0:ntotal+nvirt, :])
            rxy_tot = _np.einsum("i,ij->j", RK4_weights/6., rxy[:, 0:ntotal+nvirt])

            # update data to full-timestep
            _np.add(pts.rho0[0:ntotal+nvirt], dt*drdhot_tot[0:ntotal+nvirt], where=realmask, out=pts.rho[0:ntotal+nvirt])
            _np.add(pts.v0[0:ntotal+nvirt, :], dt*dvdt_tot[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis], out=pts.v[0:ntotal+nvirt, :])
            _np.add(pts.x[0:ntotal+nvirt, :], dt*pts.v[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis], out=pts.x[0:ntotal+nvirt, :])
            _np.add(pts.strain[0:ntotal+nvirt, :], dt*dstraindt_tot[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis], out=pts.strain[0:ntotal+nvirt, :])

            pts.stress_update(dt*dstraindt_tot, dt*rxy_tot, pts.sigma0)

            # print data to terminal if needed
            if itimestep % printtimestep == 0:
                print(f'time-step: {itimestep}')
            
            # save data to disk if needed
            if itimestep % savetimestep == 0:
                pts.save_data(itimestep)