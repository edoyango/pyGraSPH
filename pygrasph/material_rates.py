import numpy as _np

def art_visc(dv: _np.ndarray,
             dx: _np.ndarray,
             dwdx: _np.ndarray,
             rho: _np.ndarray,
             pair_i: _np.ndarray,
             pair_j: _np.ndarray,
             h: float,
             mass: float,
             c: float,
             dvdt: _np.ndarray) -> None:
    
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

def int_force(dwdx: _np.ndarray,
              sigma: _np.ndarray,
              rho: _np.ndarray,
              pair_i: _np.ndarray, 
              pair_j: _np.ndarray,
              mass: float,
              dvdt: _np.ndarray) -> None:
    
    sigma_rho2 = sigma[:, :] / (rho[:, _np.newaxis]**2)
    sigma_rho2_pairs = sigma_rho2[pair_i, :]+sigma_rho2[pair_j, :]
    h = sigma_rho2_pairs[:, 0:2] * dwdx[:, 0:2]
    h += _np.einsum("i,ij->ij", sigma_rho2_pairs[:, 3], _np.fliplr(dwdx[:, 0:2]))
    h *= mass

    _np.add.at(dvdt[:, 0], pair_i, h[:, 0])
    _np.add.at(dvdt[:, 1], pair_i, h[:, 1])
    _np.subtract.at(dvdt[:, 0], pair_j, h[:, 0])
    _np.subtract.at(dvdt[:, 1], pair_j, h[:, 1])

def con_density(dv: _np.ndarray,
                dwdx: _np.ndarray,
                pair_i: _np.ndarray,
                pair_j: _np.ndarray,
                mass: float,
                drhodt: _np.ndarray) -> None:
    
    drhodt_pairs = mass*_np.einsum("ij,ij->i", dv, dwdx)

    _np.add.at(drhodt, pair_i, drhodt_pairs)
    _np.add.at(drhodt, pair_j, drhodt_pairs)

def strain_rate(dv: _np.ndarray,
                dwdx: _np.ndarray,
                rho: _np.ndarray,
                pair_i: _np.ndarray,
                pair_j: _np.ndarray,
                mass: float,
                dstraindt: _np.ndarray,
                rxy: _np.ndarray) -> None:
    
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