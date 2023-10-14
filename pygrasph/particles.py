import numpy as _np
import typing as _typing
from closefriends import query_pairs as _query_pairs
import h5py as _h5py
import math as _math

# particles base class
class particles:
    """
    Class to store the particles' properties and necessary functions for an SPH
    simulation.
    """
    def __init__(self, maxn: int, dx: float, rho_ini: float, maxinter: int, c: float, **customvals):

        # particle constants
        self.dx = dx               # particle spacing (m)
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
    def generate_real_coords(self) -> None:

        pass

    # generate virtual particles (user to define)
    def generate_virt_coords(self) -> None:

        pass

    # stress update function (DP model)
    def stress_update(self, dstrain: _np.ndarray, drxy: _np.ndarray, sigma0: _np.ndarray) -> None:
        """
        Updates the particles' stress (sigma) using a semi-implicit 
        elasto-plastic stress update procedure with Drucker-Prager yield
        surface.
        dstrain: a 2D ndarray storing the strain increment to be applied for
                 each particle. Rows represent particles, and columns represent
                 their incremental strain tensor (voigt notation).
        drxy: a 1D ndarray storing the xy-component of the rotation incrememt
              tensor of all particles.
        sigma0: a 2D ndarray storing the initial stress states of all particles.
                Rows represent particles, and columns represent their initial
                stress tensor (voigt notation).
        """

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
    def findpairs(self, kh: float) -> None:
        """
        Function to find pairs of the particles. 
        """

        # max number of interactions
        maxnpair = _np.pi*(_math.ceil(kh)+1)**2
        maxnpair *= self.ntotal+self.nvirt
        maxnpair = int(maxnpair)

        # find pairs using closefriends.query_pairs
        pair_i, pair_j, idx = _query_pairs(self.x[0:self.ntotal+self.nvirt, :], kh, maxnpair)

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


    def update_virtualparticle_properties(self, kernel: _typing.Type) -> None:
        """
        Function to update virtual particle properties by performing a kernel
        interpolation of real particles' properties around each virtual
        particle.
        """

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
                   kernel: _typing.Type) -> None:
        """
        Performs a sweep over all particle pairs.
        dvdt: a 2D ndarray representing the acceleration of all particles.
              Should be initialized with the body force vector.
        drdhot: a 1D ndarray representing the density change rate of all
                particles. Should be initialized to 0.
        dstraindt: a 2D ndarray representing the strain rate tensor of all 
                   particles (in voigt notation). Should be initialized to 0.
        rxy: a 1D ndarray representing the xy-component of the rotation tensor.
             Should be initialized to 0.
        """

        # cache some references
        pair_i = self.pair_i
        pair_j = self.pair_j
        ntotal = self.ntotal
        nvirt = self.nvirt
        x = self.x
        v = self.v
        rho = self.rho
        sigma = self.sigma
        h = kernel.h; c = self.c; mass = self.mass

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
    def save_data(self, itimestep: int) -> None:
        """
        Function to save (compressed) particle data as hdf5 files.
        """

        with _h5py.File(f'output/sph_{itimestep}.h5', 'w') as f:
            f.attrs.create("n", data=self.ntotal+self.nvirt, dtype="i")
            f.create_dataset("x", data=self.x[0:self.ntotal+self.nvirt, :], dtype="f8", compression="gzip")
            f.create_dataset("v", data=self.v[0:self.ntotal+self.nvirt, :], dtype="f8", compression="gzip")
            f.create_dataset("type", data=self.type[0:self.ntotal+self.nvirt], dtype="i", compression="gzip")
            f.create_dataset("rho", data=self.rho[0:self.ntotal+self.nvirt], dtype="f8", compression="gzip")
            f.create_dataset("sigma", data=self.sigma[0:self.ntotal+self.nvirt, :], dtype="f8", compression="gzip")
            f.create_dataset("strain", data=self.strain[0:self.ntotal+self.nvirt, :], dtype="f8", compression="gzip")