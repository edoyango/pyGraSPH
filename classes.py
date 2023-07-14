import numpy as np
import scipy as sp
import typing
import h5py

class particles:
    def __init__(self, maxn: int, dx: float, rho_ini: float, maxinter: int, c: float, **customvals):

        self.dx = dx
        self.h = 1.5*dx
        self.rho_ini = rho_ini
        self.mass = rho_ini*dx**2
        self.maxn = maxn
        self.ntotal = 0
        self.nvirt = 0
        self.c = c

        self.x = np.zeros((maxn, 2))
        self.v = np.zeros((maxn, 2))
        self.rho = np.full(maxn, rho_ini)
        self.id = np.arange(maxn)
        self.type = np.ones(maxn)
        self.strain = np.zeros((maxn, 4))
        self.sigma = np.zeros((maxn, 4))

        self.pairs = np.ndarray((maxinter, 2))

        self.customvals = customvals

    def generate_real_coords(self):

        pass

    def generate_virt_coords(self):

        pass

    def stress_update(self, i: int, dstraini: np.ndarray, drxyi: float, sigma0: np.ndarray):

        dsig = np.matmul(self.customvals['DE'], dstraini[:])
        dsig[3] += sigma0[0]*drxyi - sigma0[1]*drxyi

        self.sigma[i, :] = sigma0[:] + dsig[:]

        # for i in range(self.ntotal+self.nvirt):
        #     p = self.c*self.c*(self.rho[i] - self.rho_ini)
        #     self.sigma[i, 0:3] = -p/3.
        #     self.sigma[i, 3] = 0.

    def findpairs(self):

        tree = sp.spatial.cKDTree(self.x[0:self.ntotal+self.nvirt, :])
        self.pairs = tree.query_pairs(3*self.dx, output_type='ndarray')

    def pair_sweep(self, 
                   dvdt: np.ndarray, 
                   drhodt: np.ndarray, 
                   dstraindt: np.ndarray, 
                   rxy: np.ndarray,
                   kernel: typing.Type):

        for i, j in self.pairs:
            dx = self.x[i, :] - self.x[j, :]
            dwdx = kernel.dwdx(dx)

            dv = self.v[i, :] - self.v[j, :]
            vr = np.dot(dv[:], dx[:])
            if vr > 0.: vr = 0.
            rr = np.dot(dx[:], dx[:])
            muv = self.h*vr/(rr + self.h*self.h*0.01)
            mrho = 0.5*(self.rho[i]+self.rho[j])
            piv = self.mass*0.2*(muv-self.c)*muv/mrho*dwdx
            dvdt[i, :] -= piv[:]
            dvdt[j, :] += piv[:]
            
            # dvdt[i, :] += 0
            h = self.mass*((self.sigma[i, 0]*dwdx[0]+self.sigma[i, 3]*dwdx[1])/self.rho[i]**2 + 
                           (self.sigma[j, 0]*dwdx[0]+self.sigma[j, 3]*dwdx[1])/self.rho[j]**2)
            dvdt[i, 0] += h
            dvdt[j, 0] -= h

            h = self.mass*((self.sigma[i, 3]*dwdx[0]+self.sigma[i, 1]*dwdx[1])/self.rho[i]**2 +
                           (self.sigma[j, 3]*dwdx[0]+self.sigma[j, 1]*dwdx[1])/self.rho[j]**2)
            dvdt[i, 1] += h
            dvdt[j, 1] -= h

            tmp_drhodt = self.mass*np.dot(self.v[i,:]-self.v[j,:], dwdx)
            drhodt[i] += tmp_drhodt
            drhodt[j] += tmp_drhodt

            he = np.zeros(4)
            he[0] = -dv[0]*dwdx[0]
            he[1] = -dv[1]*dwdx[1]
            #he[2] = 0.
            he[3] = -0.5*(dv[0]*dwdx[1]+dv[1]*dwdx[0])
            hrxy = -0.5*(dv[0]*dwdx[1] - dv[1]*dwdx[0])

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

    def save_data(self, itimestep: int):

        with h5py.File(f'output/sph_{itimestep}.h5', 'w') as f:
            f.attrs.create("n", data=self.ntotal+self.nvirt, dtype="i")
            f.create_dataset("x", data=self.x[0:self.ntotal+self.nvirt, :], dtype="f8", compression="gzip")
            f.create_dataset("type", data=self.type[0:self.ntotal+self.nvirt], dtype="i", compression="gzip")
            f.create_dataset("rho", data=self.rho[0:self.ntotal+self.nvirt], dtype="f8", compression="gzip")
            f.create_dataset("sigma", data=self.sigma[0:self.ntotal+self.nvirt, :], dtype="f8", compression="gzip")
            f.create_dataset("strain", data=self.strain[0:self.ntotal+self.nvirt, :], dtype="f8", compression="gzip")
        
class integrators:
    def __init__(self,
                 f: np.ndarray,
                 kernel: typing.Type,
                 maxtimestep: int,
                 savetimestep: int,
                 printtimestep: int,
                 cfl: float):
        self.cfl = cfl
        self.f = f
        self.kernel = kernel
        self.maxtimestep = maxtimestep
        self.savetimestep = savetimestep
        self.printtimestep = printtimestep

    def LF(self, pts: particles):

        dvdt = np.tile(self.f, (pts.ntotal+pts.nvirt, 1))
        v0 = np.empty((pts.ntotal+pts.nvirt, 2))
        drhodt = np.zeros(pts.ntotal+pts.nvirt)
        rho0 = np.empty(pts.ntotal+pts.nvirt)
        dstraindt = np.zeros((pts.ntotal+pts.nvirt, 4))
        rxy = np.zeros(pts.ntotal+pts.nvirt)
        sigma0 = np.empty((pts.ntotal+pts.nvirt, 4))

        dt = self.cfl*pts.dx*3./pts.c

        for itimestep in range(self.maxtimestep):

            pts.findpairs()

            v0 = np.copy(pts.v[0:pts.ntotal+pts.nvirt, :])
            rho0 = np.copy(pts.rho[0:pts.ntotal+pts.nvirt])
            sigma0 = np.copy(pts.sigma[0:pts.ntotal+pts.nvirt, :])

            for i in range(pts.ntotal+pts.nvirt):
                pts.rho[i] += 0.5*dt*drhodt[i]
                if pts.type[i] > 0:
                    pts.v[i, :] += 0.5*dt*dvdt[i, :]
                pts.stress_update(i, 0.5*dt*dstraindt[i, :], 0.5*dt*rxy[i], sigma0[i, :])

            dvdt = np.tile(self.f, (pts.ntotal+pts.nvirt, 1))
            drhodt = np.zeros(pts.ntotal+pts.nvirt)
            dstraindt = np.zeros((pts.ntotal+pts.nvirt, 4))
            rxy = np.zeros(pts.ntotal+pts.nvirt)
            
            pts.pair_sweep(dvdt, drhodt, dstraindt, rxy, self.kernel)

            for i in range(pts.ntotal+pts.nvirt):
                pts.rho[i] = rho0[i] + dt*drhodt[i]
                if pts.type[i] > 0:
                    pts.v[i, :] = v0[i, :] + dt*dvdt[i, :]
                    pts.x[i, :] += dt*pts.v[i, :]
                pts.stress_update(i, dt*dstraindt[i, :], dt*rxy[i], sigma0[i, :])
                pts.strain[i, :] += dt*dstraindt[i, :]

            if itimestep % self.printtimestep == 0:
                print(f'time-step: {itimestep}')
            
            if itimestep % self.savetimestep == 0:
                pts.save_data(itimestep)