import numpy as _np
import typing as _typing
from . import particles as _particles
import logging
from pydantic import Field, validate_call

class base_integrator:
    """
    A base integrator to build on top of.
    """
    def __init__(self, f: _np.ndarray, kernel):
        self. f = f
        self.kernel = kernel
    
    @validate_call
    def __call__(self, pts, maxtimestep: int = Field(ge=0), savetimestep: int = Field(ge=0), printtimestep: int = Field(ge=0), cfl: float = Field(gt=0), debug: bool = False) -> None:
        """
        Function to provide basic validation and creat the logger object for derived integrators.

        Args:
            pts: the set of particles to simulate.
            maxtimestep: maximum timesteps to run the simulation for.
            savetimestep: the frequency (in timesteps) with which to save a snapshot
                        of the particles to disc.
            printtimestep: the frequency (in timesteps) with which to print the 
                        current timestep to stdout.
            cfl: the constant used to control the time-step size where dt = cfl*h/c.
        """
        loglevel = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=loglevel)
        logger = logging.getLogger(__name__ + self.__class__.__name__)
        self._integrate(pts, maxtimestep, savetimestep, printtimestep, cfl, logger)
    
    def _integrate(self, pts: _particles, maxtimestep: int, savetimestep: int, printtimestep: int, cfl: float, logger):

        raise NotImplementedError("_integrate function not defined!")

class LF(base_integrator):
    
    def _integrate(self, pts: _particles, maxtimestep: int, savetimestep: int, printtimestep: int, cfl: float, logger):
        """
        Leap-Frog time-integration
        """

        # timestep size (s)
        dt = cfl*self.kernel.h/pts.c

        # cache some references
        ntotal = pts.ntotal
        nvirt = pts.nvirt

        # initialize material rate arrays at start of loop
        logger.debug("Initializing rate-of-change arrays at start of loop.")
        dvdt = _np.tile(self.f, (ntotal+nvirt, 1)) # acceleration
        drhodt = _np.zeros(ntotal+nvirt) # density change rate
        dstraindt = _np.zeros((ntotal+nvirt, 4)) # strain rate
        rxy = _np.zeros(ntotal+nvirt) # spin rate (for jaumann stress-rate)
        logger.debug("Completed initialization of rate-of-change arrays.")

        # begin time integration loop
        for itimestep in range(1, maxtimestep+1):

            logger.debug(f"Starting timestep {itimestep}")

            # save data from start of timestep
            logger.debug("Saving data from start of timestep.")
            pts.v0 = _np.copy(pts.v[0:ntotal+nvirt, :])
            pts.rho0 = _np.copy(pts.rho[0:ntotal+nvirt])
            pts.sigma0 = _np.copy(pts.sigma[0:ntotal+nvirt, :])
            logger.debug("Completed saving data from start of timestep.")

            # Update data to mid-timestep
            logger.debug("Updating density and velocity to mid-timestep.")
            pts.rho[0:ntotal+nvirt] += 0.5*dt*drhodt[0:ntotal+nvirt]
            pts.v[0:ntotal+nvirt, :] += 0.5*dt*dvdt[0:ntotal+nvirt, :]
            logger.debug("Completed density and velocity mid-timestep update.")

            logger.debug("Updating stress to mid-timestep.")
            pts.stress_update(0.5*dt*dstraindt, 0.5*dt*rxy, pts.sigma0)
            logger.debug("Completed stress mid-timestep update.")

            # find pairs
            logger.debug("Finding pairs.")
            pts.findpairs(self.kernel.k*self.kernel.h)
            logger.debug("Completed finding pairs.")

            realmask = pts.type[0:ntotal+nvirt] > 0

            # initialize material rate arrays
            logger.debug("Initializing rate-of-change arrays.")
            dvdt = _np.tile(self.f, (ntotal+nvirt, 1)) # acceleration
            drhodt = _np.zeros(ntotal+nvirt) # density change rate
            dstraindt = _np.zeros((ntotal+nvirt, 4)) # strain rate
            rxy = _np.zeros(ntotal+nvirt) # spin rate (for jaumann stress-rate)
            logger.debug("Completed initialization of rate-of-change arrays.")
            
            # perform sweep of pairs
            logger.debug("Performing pair sweep.")
            pts.pair_sweep(dvdt, drhodt, dstraindt, rxy, self.kernel)
            logger.debug("Completed pair sweep.")

            logger.debug("Updating stress to full-timestep.")
            pts.stress_update(dt*dstraindt, dt*rxy, pts.sigma0)
            logger.debug("Completed full-timestep stress update.")

            # update data to full-timestep
            logger.debug("Update density, velocity, position, strain to full-timestep.")
            _np.add(pts.rho0[0:ntotal+nvirt], dt*drhodt[0:ntotal+nvirt], where=realmask, out=pts.rho[0:ntotal+nvirt])
            _np.add(pts.v0[0:ntotal+nvirt, :], dt*dvdt[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis], out=pts.v[0:ntotal+nvirt, :])
            _np.add(pts.x[0:ntotal+nvirt, :], dt*pts.v[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis], out=pts.x[0:ntotal+nvirt, :])
            _np.add(pts.strain[0:ntotal+nvirt, :], dt*dstraindt[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis], out=pts.strain[0:ntotal+nvirt, :])
            logger.debug("Completed updating density, velocity, position, strain to full-timestep.")

            # print data to terminal if needed
            if itimestep % printtimestep == 0:
                logger.info(f'time-step: {itimestep}')
            
            # save data to disk if needed
            if itimestep % savetimestep == 0:
                logger.debug("Saving data to disk.")
                pts.save_data(itimestep)
                logger.debug("Completed saving data to disk.")
            
            logger.debug(f"Finished timestep: {itimestep}")

class RK4(base_integrator):

    def _integrate(self, pts: _particles, maxtimestep: int, savetimestep: int, printtimestep: int, cfl: float, logger):
        """
        Runge-Kutte fourth-order time-integration.
        pts: the set of particles to simulate.
        maxtimestep: maximum timesteps to run the simulation for.
        savetimestep: the frequency (in timesteps) with which to save a snapshot
                      of the particles to disc.
        printtimestep: the frequency (in timesteps) with which to print the 
                       current timestep to stdout.
        cfl: the constant used to control the time-step size where dt = cfl*h/c
        """

        logger.debug("Initializing time integration parameters.")

        # timestep size (s)
        dt = cfl*self.kernel.h/pts.c

        RK4_weights = _np.array([1., 2., 2., 1.])

        # cache some references
        ntotal = pts.ntotal
        nvirt = pts.nvirt

        # begin time integration loop
        for itimestep in range(1, maxtimestep+1):

            logger.debug(f"Starting timestep {itimestep}")

            # initialize rate-of-change arrays needed for time integration
            logger.debug("Initializing rate-of-change arrays at start of time-step.")
            dvdt = _np.tile(self.f, (4, ntotal+nvirt, 1)) # acceleration
            drhodt = _np.zeros((4, ntotal+nvirt), dtype=_np.float64) # density change rate
            dstraindt = _np.zeros((4, ntotal+nvirt, 4), dtype=_np.float64) # strain rate
            rxy = _np.zeros((4, ntotal+nvirt), dtype=_np.float64) # spin rate (for jaumann stress-rate)
            logger.debug("Completed initialization of rate-of-change arrays.")

            # save data from start of timestep
            logger.debug("Saving data from start of timestep.")
            pts.v0 = _np.copy(pts.v[0:ntotal+nvirt, :])
            pts.rho0 = _np.copy(pts.rho[0:ntotal+nvirt])
            pts.sigma0 = _np.copy(pts.sigma[0:ntotal+nvirt, :])
            logger.debug("Completed saving data from start of timestep.")

            # find pairs
            logger.debug("Finding pairs.")
            pts.findpairs(self.kernel.k*self.kernel.h)
            logger.debug("Completed finding pairs.")

            realmask = pts.type[0:ntotal+nvirt] > 0

            # k1 ---------------------------------------------------------------
            
            # perform first sweep of pairs (k1)
            logger.debug("Performing first sweep of pairs (k1).")
            pts.pair_sweep(dvdt[0, :, :], drhodt[0, :], dstraindt[0, :, :], rxy[0, :], self.kernel)
            logger.debug("Completed first sweep of pairs (k1).")

            # perform 2nd - 4th RK4 iteration
            for k in range(1, 4):

                # update properties according to RK4_weights
                logger.debug(f"Updating density using {k}th RK4 weight and {k-1}th rate-of-change")
                _np.add(
                    pts.rho0[0:ntotal+nvirt], 
                    dt/RK4_weights[k]*drhodt[k-1, 0:ntotal+nvirt], 
                    where=realmask, 
                    out=pts.rho[0:ntotal+nvirt]
                )
                logger.debug(f"Completed density update")

                logger.debug(f"Updating velocity using {k}th RK4 weight and {k-1}th rate-of-change")
                _np.add(
                    pts.v0[0:ntotal+nvirt, :], 
                    dt/RK4_weights[k]*dvdt[k-1, 0:ntotal+nvirt, :], 
                    where=realmask[:, _np.newaxis], 
                    out=pts.v[0:ntotal+nvirt, :]
                )
                logger.debug(f"Completed velocity update")

                # update stress according to RK4_weights
                logger.debug(f"Updating stress using {k}th RK4 weight and {k-1}th rate-of-change")
                pts.stress_update(dt/RK4_weights[k]*dstraindt[k-1, :, :], dt/RK4_weights[k]*rxy[k-1, :], pts.sigma0)
                logger.debug(f"Completed stress update")

                # perform sweep of pairs
                logger.debug(f"Performing sweep of pairs k{k+1}.")
                pts.pair_sweep(dvdt[k, :, :], drhodt[k, :], dstraindt[k, :, :], rxy[k, :], self.kernel)
                logger.debug(f"Completed sweep of pairs k{k+1}.")

            # final update -----------------------------------------------------

            logger.debug("Calculating total rate-of-change arrays.")
            drdhot_tot = _np.einsum("i,ij->j", RK4_weights/6., drhodt[:, 0:ntotal+nvirt])
            dvdt_tot = _np.einsum("i,ijk->jk", RK4_weights/6., dvdt[:, 0:ntotal+nvirt, :])
            dstraindt_tot = _np.einsum("i,ijk->jk", RK4_weights/6., dstraindt[:, 0:ntotal+nvirt, :])
            rxy_tot = _np.einsum("i,ij->j", RK4_weights/6., rxy[:, 0:ntotal+nvirt])
            logger.debug("Completed calculating total rate-of-change.")

            # update data to full-timestep
            logger.debug("Update particle data to full-timestep.")
            _np.add(pts.rho0[0:ntotal+nvirt], dt*drdhot_tot[0:ntotal+nvirt], where=realmask, out=pts.rho[0:ntotal+nvirt])
            _np.add(pts.v0[0:ntotal+nvirt, :], dt*dvdt_tot[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis], out=pts.v[0:ntotal+nvirt, :])
            _np.add(pts.x[0:ntotal+nvirt, :], dt*pts.v[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis], out=pts.x[0:ntotal+nvirt, :])
            _np.add(pts.strain[0:ntotal+nvirt, :], dt*dstraindt_tot[0:ntotal+nvirt, :], where=realmask[:, _np.newaxis], out=pts.strain[0:ntotal+nvirt, :])
            logger.debug("Completed particle data update.")

            logger.debug("Update particle stress.")
            pts.stress_update(dt*dstraindt_tot, dt*rxy_tot, pts.sigma0)
            logger.debug("Completed update of particle stress.")

            # print data to terminal if needed
            if itimestep % printtimestep == 0:
                logger.info(f'time-step: {itimestep}')
            
            # save data to disk if needed
            if itimestep % savetimestep == 0:
                logger.debug("Saving data to disk.")
                pts.save_data(itimestep)
                logger.debug("Completed saving data to disk.")
            
            logger.debug(f"Finished timestep: {itimestep}")