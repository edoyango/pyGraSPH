import numpy as _np
import typing as _typing
from closefriends import query_pairs as _query_pairs
import h5py as _h5py
import math as _math
from . import material_rates as _material_rates
from pydantic import Field, validate_call
from . import stress_update

# particles base class
class particles:

    """
    Class to store the particles' properties and necessary functions for an SPH
    simulation.
    """
    @validate_call
    def __init__(self, 
                 maxn: int = Field(gt=0), # positive non-zero particles (zero particles would be pointless)
                 dx: float = Field(gt=0), # positive non-zero distance (zero is non-physical)
                 rho_ini: float = Field(gt=0), # positive non-zero reference density (zero is non-physical)
                 maxinter: int = Field(ge=0), # positive max no. of particles
                 c: float = Field(gt=0), # positive non-zero speed of sound (zero is non-physical)
                 f_stress_update: _typing.Callable = stress_update.DP,
                 **customvals): # customvals used in stress update

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

        self._stress_update = f_stress_update

        # custom data in dict
        self.customvals = customvals

    # stress update function (DP model)
    def stress_update(self, dstrain: _np.ndarray, drxy: _np.ndarray, sigma0: _np.ndarray) -> None:
        """
        Updates the particles' stress (sigma) using the _stress_update function.
        dstrain: a 2D ndarray storing the strain increment to be applied for
                 each particle. Rows represent particles, and columns represent
                 their incremental strain tensor (voigt notation).
        drxy: a 1D ndarray storing the xy-component of the rotation incrememt
              tensor of all particles.
        sigma0: a 2D ndarray storing the initial stress states of all particles.
                Rows represent particles, and columns represent their initial
                stress tensor (voigt notation).
        """
        self._stress_update(self, dstrain, drxy, sigma0)

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
                w = _np.apply_along_axis(kernel, 0, r)
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
        dwdx = kernel.grad(dx)

        ## update virtual particles' properties --------------------------------
        self.update_virtualparticle_properties(kernel)

        ## sweep over all pairs to update real particles' material rates -------
        # update acceleration with artificial viscosity
        _material_rates.art_visc(dv, dx, dwdx, rho, pair_i, pair_j, h, mass, c, dvdt)

        # update acceleration with div stress
        # using momentum consertive form
        _material_rates.int_force(dwdx, sigma[0:ntotal+nvirt, :], rho[0:ntotal+nvirt], pair_i, pair_j, mass, dvdt)

        # update density change rate with continuity density
        _material_rates.con_density(dv, dwdx, pair_i, pair_j, mass, drhodt)

        # calculating engineering strain rates
        _material_rates.strain_rate(dv, dwdx, rho, pair_i, pair_j, mass, dstraindt, rxy)

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
