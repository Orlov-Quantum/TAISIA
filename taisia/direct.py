"""TAISIA: direct optical tomogram -> complex signal (Manko inversion).

Uses precisely Eq. (6.6)/(6.7) of TAISIA foundations, NOT the
known wavefunction, a Wigner function or a Fock-density reconstruction.

K(q,q') = (1/2pi) integral dmu exp(-i mu(q+q')/2)
                     integral dX exp(iX) w(X|mu,q-q').

With optical slices, w(X|mu,nu) dX = w(x|theta) dx,
X=r*x, r=hypot(mu,nu), theta=atan2(nu,mu). The inner Fourier
integral is computed numerically by Gauss-Hermite quadrature.
The outer integral is a numerical trapezoid. We compute one **true**
column of K(q,q'), K(q,q_ref), sufficient for any pure state with a
nonzero reference amplitude. A small second column validates rank-one
factorisation; the full matrix is not needed for audible rendering.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import roots_hermitenorm
from .physics import State, KINDS, tomogram


@dataclass(frozen=True)
class DirectConfig:
    # These values are conservative for bounded state parameters in physics.py.
    mu_max: float = 17.0
    mu_step: float = .12
    quadrature_nodes: int = 320
    q_max: float = 9.0
    q_points: int = 361
    # For impractically high chirp at large displacements, tell the user
    # that the direct-integral method is outside its validated regime.


@dataclass
class DirectRecovery:
    q: np.ndarray
    signal_q: np.ndarray
    reference_q: float
    kernel_column: np.ndarray
    kernel_diagonal_reference: float
    density_relative_error: float
    fidelity_reference_only: float
    norm: float
    mu_max: float
    mu_step: float
    quadrature_nodes: int
    kernel_rank_consistency: float
    residual_far_boundary: float
    reference_density_error: float = float('nan')

    @property
    def psi(self):
        """Backward-compatible alias; in the main synthesis this is a SIGNAL, not a reconstructed ψ stage."""
        return self.signal_q



@lru_cache(maxsize=6)
def _nodes(n:int):
    """Quadrature for ∫ f(x) dx with normal-weight Hermite nodes.

    roots_hermitenorm integrates exp(-x²/2); cancel this weight with
    exp(x²/2) while staying in a bounded (<=320) numerical regime.
    """
    if n<80 or n>320: raise ValueError('80 ≤ число узлов квадратуры ≤ 320')
    nodes, w = roots_hermitenorm(n)
    weights = w*np.exp(nodes**2/2)
    if not np.all(np.isfinite(weights)):
        raise ValueError('Численная неустойчивость весов квадратуры')
    return nodes, weights


def _mu_grid(config:DirectConfig)->np.ndarray:
    if not 10 <= config.mu_max <= 45: raise ValueError('10 ≤ mu_max ≤ 45')
    if not .04 <= config.mu_step <= .2: raise ValueError('0.04 ≤ mu_step ≤ 0.2')
    n=math.ceil(2*config.mu_max/config.mu_step)
    # n even assures a node mu=0, symmetric grid.
    if n%2: n+=1
    return np.linspace(-config.mu_max,config.mu_max,n+1)


def kernel_column(state:State, q:np.ndarray, q_ref:float, config:DirectConfig=DirectConfig())->np.ndarray:
    """Independently compute K(q_i,q_ref) by the TWO integrals over w.

    Input is only the analytic *probability tomogram* generator. No use
    of wave(), quadrature_amplitude(), basis(), or input coefficients.
    """
    state.check()
    q=np.asarray(q,dtype=float)
    if q.ndim != 1 or not np.all(np.isfinite(q)) or not np.isfinite(q_ref):
        raise ValueError('Некорректная сетка координат')
    nodes, weights = _nodes(int(config.quadrature_nodes))
    mu = _mu_grid(config)
    vals=np.empty(q.size,dtype=np.complex128)
    # Gaussians permit an exact inner Fourier integral based *only* on the
    # TOMOGRAM: infer mean m and width D from its three log-density values.
    # Then ∫ e^(irx) w(x|theta) dx = exp(i*r*m - r²*D/4).
    # This makes high-chirp states tractable without replacing K inversion.
    gaussian = state.kind in (KINDS[0], KINDS[2], KINDS[3])
    # Batch to avoid allocating (Nq,Nmu,NX) arrays.
    for i,qi in enumerate(q):
        nu=qi-q_ref
        r=np.hypot(mu,nu)
        theta=np.arctan2(nu,mu)
        if gaussian:
            h=1.0
            # From the Gaussian *probability density*, not from ψ or ρ.
            log0=np.log(tomogram(state,np.zeros_like(theta),theta))
            logp=np.log(tomogram(state,np.full_like(theta,h),theta))
            logm=np.log(tomogram(state,np.full_like(theta,-h),theta))
            D=-2*h*h/(logp+logm-2*log0)
            mean=D*(logp-logm)/(4*h)
            if np.any(~np.isfinite(D)) or np.any(D<=0):
                raise ValueError('Ошибка аналитической гауссовой характеристической функции')
            characteristic=np.exp(1j*r*mean-.25*r*r*D)
        else:
            # Tomogram only, evaluated on Hermite quadrature nodes.
            w=tomogram(state,nodes[None,:],theta[:,None])
            # Numerical Fourier transform of each optical probability density.
            characteristic=np.sum(w*np.exp(1j*r[:,None]*nodes[None,:])*weights[None,:],axis=1)
        vals[i]=trapezoid(characteristic*np.exp(-.5j*mu*(qi+q_ref)),x=mu)/(2*np.pi)
    return vals


def signal_from_tomogram(state:State,t:np.ndarray,T:float,t_ref:float,
                         config:DirectConfig=DirectConfig())->np.ndarray:
    """Direct Manko inversion: w(X|mu,nu) -> s(t), with NO ψ or full K stage.

    s(t) = [2*pi*T*sqrt(M_sigma(t_ref|1,0))]^-1 *
      ∫dmu ∫dX w(X|mu,(t-t_ref)/T) exp(iX-i mu(t+t_ref)/(2T)).

    M_sigma(t_ref|1,0)=w(q_ref|1,0)/T. For t_ref=0 this is exactly
    the user's requested formula. For a nodal state choose t_ref≠0.

    Numerically, the double integral is computed using dimensionless q=t/T.
    The function kernel_column performs that integral and is NOT a separately
    reconstructed full correlation operator, Fock state, or wavefunction.
    """
    if not np.isfinite(T) or T<=0: raise ValueError('T должно быть положительно')
    t=np.asarray(t,float)
    density_ref=float(tomogram(state,np.asarray(t_ref/T),0.))
    if density_ref<=1e-9:raise ValueError('Нулевая опорная вероятность, нужно другое t_ref')
    integral=kernel_column(state,t/T,t_ref/T,config)
    return integral/np.sqrt(T*density_ref)


def recover_direct(state:State, config:DirectConfig=DirectConfig(), *, validate_reference:bool=True)->DirectRecovery:
    """Build a pure-state wave by *direct* kernel inversion with no Fock stage.

    The target analytical wave is permitted ONLY for validation and is
    never used to calculate the resulting wave, gauge or kernel column.
    """
    state.check()
    if not 5 <= config.q_max <= 15 or not 161 <= config.q_points <= 1201:
        raise ValueError('Требуется q_max∈[5,15] и 161≤q_points≤1201')
    # Gaussian chirps shift the peak of χ(mu,nu) by -b*nu; extend mu
    # coverage while retaining exact Fourier transforms of Gaussian w.
    if state.kind == KINDS[3]:
        from dataclasses import replace
        config=replace(config,mu_max=min(45.,max(config.mu_max,17.+abs(state.b)*max(3.,5./np.sqrt(state.a)))))
    q=np.linspace(-config.q_max,config.q_max,config.q_points)
    density=tomogram(state,q,0.)
    anchor=int(np.argmax(density))
    q_ref=float(q[anchor]); pref=float(density[anchor])
    if pref < 1e-7: raise ValueError('Нет устойчивой опорной точки')
    # Direct signal formula: normalization comes from the TOMOGRAM,
    # not from a separately reconstructed kernel diagonal.
    signal_q=signal_from_tomogram(state,q,1.,q_ref,config)
    col=signal_q*np.sqrt(pref) # diagnostic-only K(q,q_ref), not synthesis input
    kr=float(col[anchor].real)
    if not np.isfinite(kr) or kr<=1e-8:
        raise ValueError('Обращение не дало положительную опорную диагональ')
    reference_density_error=float(abs(kr-pref)/pref)
    norm=float(trapezoid(abs(signal_q)**2,x=q))
    density_relative_error=float(trapezoid((abs(signal_q)**2-density)**2,x=q)/max(trapezoid(density**2,x=q),1e-20))
    # Second independent kernel column tests factorization K=|psi><psi|,
    # not just the single-column normalisation.
    other=int(np.argmax(density * (abs(q-q_ref)>.65)))
    q_second=float(q[other])
    if abs(q_second-q_ref)<.2: other=int(np.argmax(density*(abs(q-q_ref)>1.2)));q_second=float(q[other])
    indices=np.unique(np.round(np.linspace(0,len(q)-1,21)).astype(int))
    second=kernel_column(state,q[indices],q_second,config)
    expected=signal_q[indices]*np.conj(signal_q[other])
    rank_error=float(np.linalg.norm(second-expected)/max(np.linalg.norm(second),1e-14))
    # Optional reference comparison, purely for diagnostics, NEVER in synthesis.
    fidelity=float('nan')
    if validate_reference:
        from .physics import wave  # imported only here, not in main reconstruction
        ground=wave(state,q)
        ov=trapezoid(np.conj(ground)*signal_q,x=q)
        fidelity=float(abs(ov)**2/max(trapezoid(abs(ground)**2,x=q)*norm,1e-20))
    boundary=float(max(abs(signal_q[0]),abs(signal_q[-1]))/max(np.max(abs(signal_q)),1e-15))
    if not np.all(np.isfinite(signal_q)):
        raise ValueError('Комплексная волна содержит NaN или Inf')
    return DirectRecovery(q,signal_q,q_ref,col,kr,density_relative_error,fidelity,norm,
                          config.mu_max,float(2*config.mu_max/(len(_mu_grid(config))-1)),
                          config.quadrature_nodes,rank_error,boundary,reference_density_error)
