import numpy as _np
import typing as _typing
from . import particles as _particles

# container class to hold time integration functions
class integrators:
    """
    Container class for integrators that evolve particles' properties over time.
    """
    def __init__(self,
                 f: _np.ndarray,
                 kernel: _typing.Type):
        self.f = f     # body force vector e.g. gravity
        self.kernel = kernel # kernel function of choice

    def LF(self, pts: _particles,
           maxtimestep: int, # timestep to run simulation for
           savetimestep: int, # timestep interval to save data to disk
           printtimestep: int, # timestep interval to print timestep
           cfl: float) -> None: # Courant-Freidrichs-Lewy coefficient for time-step size
        """
        Leap-Frog time-integration.
        pts: the set of particles to simulate.
        maxtimestep: maximum timesteps to run the simulation for.
        savetimestep: the frequency (in timesteps) with which to save a snapshot
                      of the particles to disc.
        printtimestep: the frequency (in timesteps) with which to print the 
                       current timestep to stdout.
        cfl: the constant used to control the time-step size where dt = cfl*h/c.
        """

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

    def RK4(self, pts: _particles,
           maxtimestep: int, # timestep to run simulation for
           savetimestep: int, # timestep interval to save data to disk
           printtimestep: int, # timestep interval to print timestep
           cfl: float) -> None: # Courant-Freidrichs-Lewy coefficient for time-step size
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