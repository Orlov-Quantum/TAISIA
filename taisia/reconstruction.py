"""Pure state reconstruction from optical *tomograms*, without Wigner functions.
Uses Fourier analysis in quadrature angle and linear Fock-basis inversion.
Not an experimental-data estimator; the input consists of analytic probabilities.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .physics import State, basis, tomogram, wave_from_coeff

@dataclass
class Recovery:
    coeff: np.ndarray
    rho: np.ndarray
    largest_eigenvalue: float
    negative_eigenmass: float
    tomographic_relative_error: float
    dimension: int
    points: int
    angles: int


def recover(state: State, dimension:int=12, points:int=251, angle_count:int|None=None, xmax:float=8.5)->Recovery:
    """Reconstruct rho_{m,n} using angular Fourier components.

    w(X|theta) = sum_{m,n} rho[m,n] exp(-i(m-n)theta) phi_m(X)phi_n(X).
    Thus the kth angular Fourier coefficient is sum_n rho[n+k,n]phi[n+k]phi[n].
    Solve these small linear systems, symmetrize, project onto the leading
    eigenvector for a pure-state estimate. No true coefficients are supplied.
    """
    state.check()
    d=int(dimension)
    if not 2 <= d <= 20: raise ValueError('2 ≤ размерность d ≤ 20')
    if points<2*d+8: raise ValueError('Недостаточно точек X')
    angles = int(angle_count or max(2*d+8,40))
    if angles<2*d: raise ValueError('Для данного d недостаточно углов')
    xs=np.linspace(-xmax,xmax,points)
    thetas=2*np.pi*np.arange(angles)/angles
    w=np.array([tomogram(state,xs,float(t)) for t in thetas])
    # DFT normalization includes 1 / number_of_angles; Fourier sign as above.
    ang=np.fft.fft(w,axis=0)/angles   # F[k]=sum_theta w exp(-i k theta)
    # Since w contains exp(-i(m-n)theta), rho[m,n] lives in F[-(m-n)].
    b=basis(xs,d)
    rho=np.zeros((d,d),complex)
    for k in range(d):
        data=ang[(-k)%angles]
        A=(b[k:]*b[:d-k]).T # each column phi_{n+k}(x) phi_n(x)
        # Weak ridge stabilizes high oscillator levels without forcing positivity.
        # Linear systems are small, but high n can be poorly identifiable on finite grids.
        scale=np.linalg.norm(A,axis=0)
        scale=np.maximum(scale,1e-14)
        AA=A/scale[None,:]
        reg=3e-7
        sol=np.linalg.solve(AA.T@AA+reg*np.eye(d-k),AA.T@data)
        entries=sol/scale
        for n,v in enumerate(entries):
            rho[n+k,n]=v
            if k: rho[n,n+k]=np.conj(v)
    rho=(rho+rho.conj().T)/2
    # Linear inversion of a truncated analytic pure state may violate positivity.
    values,vecs=np.linalg.eigh(rho)
    largest=float(values[-1].real)
    negmass=float(-np.sum(np.minimum(values,0)))
    coeff=vecs[:,-1]
    anchor=np.argmax(np.abs(coeff))
    coeff=coeff*np.exp(-1j*np.angle(coeff[anchor]))
    # Compare physical reconstructed tomogram, not just algebraic linear fit.
    checks=np.array([0.0,.33*np.pi,.5*np.pi,.79*np.pi])
    bb=b
    pred=np.array([np.abs((coeff*np.exp(-1j*np.arange(d)*t))@bb)**2 for t in checks])
    target=np.array([tomogram(state,xs,float(t)) for t in checks])
    err=float(np.sum((pred-target)**2)/max(np.sum(target**2),1e-18))
    return Recovery(coeff,rho,largest,negmass,err,d,points,angles)
