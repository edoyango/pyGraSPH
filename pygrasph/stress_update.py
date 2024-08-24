import numpy as _np
from . import particles

def DP(pts: particles, dstrain: _np.ndarray, drxy: _np.ndarray, sigma0: _np.ndarray) -> None:
    """
    Updates the particles' stress (sigma) using a semi-implicit 
    elasto-plastic stress update procedure with Drucker-Prager yield
    surface.
    pts: the particles whose stresses are to be updated.
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
    DE = pts.customvals['DE'][:, :]
    k_c = pts.customvals['k_c']
    alpha_phi = pts.customvals['alpha_phi']
    alpha_psi = pts.customvals['alpha_psi']
    sigma = pts.sigma # this stores reference, not the data.
    ntotal = pts.ntotal
    nvirt = pts.nvirt
    realmask = pts.type[0:ntotal+nvirt] > 0

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

def linear_EOS(pts: particles, *args, **kwargs) -> None:

    """
    Updates the particles' stress (sigma) using a linear equation of state.
    pts: the particles whose stresses are to be updated.
    """

    # simple fluid equation of state.
    p = pts.c*pts.c*(pts.rho[:] - pts.rho_ini)
    pts.sigma[:, :3] = _np.tile(-p, (3, 1)).T
    pts.sigma[:, 3] = 0.