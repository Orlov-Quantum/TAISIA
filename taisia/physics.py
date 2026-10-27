"""Analytic one-mode *pure* quantum states, in hbar=1, [q,p]=i.
No Wigner transform, stochastic mixture, or acoustic note mapping is used.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np

KINDS = (
    'Vacuum |0>', 'Fock |n>', 'Coherent |alpha>',
    'Squeezed / chirped', 'Even cat |C+>', 'Odd cat |C->',
    'Superposition |n0>+z|n1>',
    'Compass cat (4 components)', 'Yurke-Stoler cat (phase i)',
    'Photon-added coherent (a†)^m|alpha>',
    'Kerr-evolved coherent', 'q-deformed coherent (symmetric q)',
    'Squeezed Fock S(a,b)|n>', 'Binomial state |N,p,phase>',
    'Displaced Fock D(alpha)|n>',
)

@dataclass(frozen=True)
class State:
    kind: str = KINDS[0]
    n: int = 1
    n0: int = 0
    n1: int = 1
    ratio: float = 1.0
    phase: float = 0.0
    q0: float = 1.8
    p0: float = 0.0
    a: float = 1.0
    b: float = 0.0
    m: int = 1
    kerr: float = 0.8
    deformation: float = 0.35
    binomial_p: float = 0.5

    def check(self) -> None:
        if self.kind not in KINDS: raise ValueError('Неизвестный вид состояния')
        if self.n not in range(13) or self.n0 not in range(13) or self.n1 not in range(13):
            raise ValueError('Фоковские индексы должны лежать от 0 до 12')
        for key in ('ratio','phase','q0','p0','a','b'):
            if not math.isfinite(getattr(self,key)):
                raise ValueError(f'{key} должен быть конечным числом')
        if not 0 <= self.ratio <= 3: raise ValueError('0 ≤ относительная амплитуда ≤ 3')
        if not .35 <= self.a <= 3.0: raise ValueError('0.35 ≤ a ≤ 3.0')
        if abs(self.b)>3: raise ValueError('|b| ≤ 3')
        if abs(self.q0)>4 or abs(self.p0)>4: raise ValueError('|q0|, |p0| ≤ 4')
        if self.m not in range(5): raise ValueError('0 ≤ m ≤ 4')
        if not 0 <= self.kerr <= 2*np.pi: raise ValueError('0 ≤ χ ≤ 2π')
        if not 0 <= self.deformation <= 1.2: raise ValueError('0 ≤ κ ≤ 1.2')
        if not math.isfinite(self.binomial_p) or not 0 <= self.binomial_p <= 1:
            raise ValueError('Биномиальная вероятность p должна лежать между 0 и 1')
        if self.kind == 'Odd cat |C->' and self.q0*self.q0+self.p0*self.p0 < 1e-8:
            raise ValueError('Нечётный кот при α=0 не определён; задайте ненулевые q0 или p0')

    def describe(self):
        return asdict(self)


def basis(x: np.ndarray, d: int) -> np.ndarray:
    """Normalized Hermite functions phi_n(x), recurrence to avoid large factorials."""
    if d < 1 or d > 64: raise ValueError('1 ≤ d ≤ 64')
    x=np.asarray(x,dtype=float)
    out=np.empty((d,)+x.shape,dtype=np.float64)
    out[0]=np.pi**(-.25)*np.exp(-x*x/2)
    if d>1: out[1]=np.sqrt(2)*x*out[0]
    for n in range(1,d-1):
        out[n+1]=np.sqrt(2/(n+1))*x*out[n]-np.sqrt(n/(n+1))*out[n-1]
    return out


def coherent_wave(x: np.ndarray, q0: float, p0: float) -> np.ndarray:
    """Canonical phase convention for |alpha>, alpha=(q0+i*p0)/sqrt(2)."""
    x=np.asarray(x,dtype=float)
    return np.pi**(-.25)*np.exp(-.5*(x-q0)**2+1j*p0*(x-.5*q0))


def _rotated(q0:float,p0:float,theta:float):
    return q0*np.cos(theta)+p0*np.sin(theta), p0*np.cos(theta)-q0*np.sin(theta)


def _hermite_wave(x: np.ndarray, n:int) -> np.ndarray:
    return basis(x,n+1)[n].astype(complex)


def _cat(x:np.ndarray, q0:float,p0:float,parity:int) -> np.ndarray:
    alpha_sq=.5*(q0*q0+p0*p0)
    # overlap <-alpha|alpha> = exp(-2 |alpha|^2) is real.
    denom=2*(-np.expm1(-2*alpha_sq)) if parity == -1 else 2*(1+np.exp(-2*alpha_sq))
    if np.any(denom < 1e-12): raise ValueError('Cat normalization is singular')
    return (coherent_wave(x,q0,p0)+parity*coherent_wave(x,-q0,-p0))/np.sqrt(denom)


def _coherent_coeff(alpha:complex, d:int=48) -> np.ndarray:
    """Finite precision reference coefficients, normalized with tail detection."""
    coeff=np.zeros(d,complex)
    coeff[0]=np.exp(-abs(alpha)**2/2)
    for n in range(1,d):
        coeff[n]=coeff[n-1]*alpha/np.sqrt(n)
    norm=np.vdot(coeff,coeff).real
    if norm < .9999995: raise ValueError('Когерентное состояние выходит за усечение; уменьшите |α|')
    return coeff/np.sqrt(norm)


def fock_coefficients(state:State, d:int=48) -> np.ndarray:
    """Known analytic Fock coefficients solely for construction of w(X|θ).

    The *direct inversion* never receives these coefficients. For q-deformation
    use the explicitly declared symmetric q-number [n]_q=sinh(nκ)/sinh(κ),
    with q=exp(κ), and A=a sqrt([N]_q/N). This convention is not unique.
    """
    if d<20 or d>64:raise ValueError('20 ≤ d ≤ 64')
    k=state.kind
    alpha=(state.q0+1j*state.p0)/np.sqrt(2)
    if k==KINDS[9]: # photon added: (a†)^m |α>
        old=_coherent_coeff(alpha,d-state.m)
        c=np.zeros(d,complex)
        for n in range(state.m,d):
            # factorial ratio via small (m<=4) product
            mult=np.prod(np.sqrt(np.arange(n-state.m+1,n+1))) if state.m else 1
            c[n]=old[n-state.m]*mult
    elif k==KINDS[10]:
        c=_coherent_coeff(alpha,d)
        indices=np.arange(d)
        c*=np.exp(-.5j*state.kerr*indices*(indices-1))
    elif k==KINDS[12+1]: # Binomial pure state N=n; Stoler-Saleh-Teich.
        p=state.binomial_p
        c=np.zeros(d,complex)
        for n in range(state.n+1):
            c[n]=np.sqrt(math.comb(state.n,n)*p**n*(1-p)**(state.n-n))*np.exp(1j*n*state.phase)
    elif k==KINDS[11]:
        c=np.zeros(d,complex);c[0]=1
        for n in range(1,d):
            qn=n if state.deformation<1e-10 else np.sinh(n*state.deformation)/np.sinh(state.deformation)
            c[n]=c[n-1]*alpha/np.sqrt(qn)
    else:raise ValueError('Это состояние не задано фоковскими коэффициентами')
    norm=np.linalg.norm(c)
    if not np.isfinite(norm) or norm < 1e-14:raise ValueError('Неустойчивая нормировка нового состояния')
    c/=norm
    if abs(c[-1])**2+abs(c[-2])**2>2e-8:
        raise ValueError('Недостаточное фоковское усечение для выбранных параметров')
    return c


def coherent_superposition(state:State,x:np.ndarray,theta:float)->np.ndarray:
    """Coherent packets rotated together, with their original relative phases."""
    z=(state.q0+1j*state.p0)/np.sqrt(2)
    if state.kind==KINDS[7]:
        phases=np.arange(4)*np.pi/2
        weights=np.ones(4,complex)
    elif state.kind==KINDS[8]:
        phases=np.array([0.,np.pi]);weights=np.array([1.,1j])
    else:raise ValueError('Не многокомпонентный кот')
    alphas=z*np.exp(1j*phases)
    # coherent states |α> have overlap <α|β>=exp[-|α|²/2-|β|²/2+α*β]
    gram=np.exp(-.5*(abs(alphas[:,None])**2+abs(alphas[None,:])**2)
                +np.conj(alphas[:,None])*alphas[None,:])
    norm=np.real(np.vdot(weights,gram@weights))
    if norm<=1e-12:raise ValueError('Нулевая суперпозиция когерентных пакетов')
    vals=np.zeros(np.broadcast_shapes(np.shape(x),np.shape(theta)),dtype=complex)
    for a,w in zip(alphas,weights):
        rotated=a*np.exp(-1j*theta)
        vals+=w*coherent_wave(x,np.sqrt(2)*rotated.real,np.sqrt(2)*rotated.imag)
    return vals/np.sqrt(norm)


def wave(state: State, x:np.ndarray) -> np.ndarray:
    state.check()
    x=np.asarray(x,dtype=float)
    k=state.kind
    if k == KINDS[0]: return _hermite_wave(x,0)
    if k == KINDS[1]: return _hermite_wave(x,state.n)
    if k == KINDS[2]: return coherent_wave(x,state.q0,state.p0)
    if k == KINDS[3]:
        y=x-state.q0
        return (state.a/np.pi)**.25 * np.exp(-.5*(state.a-1j*state.b)*y*y+1j*state.p0*(x-.5*state.q0))
    if k == KINDS[4]: return _cat(x,state.q0,state.p0,1)
    if k == KINDS[5]: return _cat(x,state.q0,state.p0,-1)
    if k in (KINDS[7],KINDS[8]):
        return coherent_superposition(state,x,0.)
    if k==KINDS[12]: # Squeezed and chirped displaced number state, unit-normalised.
        y=x-state.q0
        return state.a**.25 * _hermite_wave(np.sqrt(state.a)*y,state.n) * np.exp(1j*(state.b*y*y/2 + state.p0*(x-.5*state.q0)))
    if k==KINDS[14]: # Displaced number state; fixed global phase irrelevant.
        return _hermite_wave(x-state.q0,state.n)*np.exp(1j*state.p0*(x-.5*state.q0))
    if k in (KINDS[9],KINDS[10],KINDS[11],KINDS[13]):
        return wave_from_coeff(fock_coefficients(state),x)
    # A proper coherent superposition, not a probabilistic mixture.
    den=1+state.ratio**2
    if state.n0 == state.n1:
        coeff=1+state.ratio*np.exp(1j*state.phase)
        if abs(coeff)<1e-12: raise ValueError('Нулевая суперпозиция: коэффициенты сократились')
        return coeff/abs(coeff)*_hermite_wave(x,state.n0)
    return (_hermite_wave(x,state.n0)+state.ratio*np.exp(1j*state.phase)*_hermite_wave(x,state.n1))/np.sqrt(den)


def quadrature_amplitude(state:State,x:np.ndarray,theta:float) -> np.ndarray:
    """Amplitude <X_theta|psi>, modulo an irrelevant common global phase.
    Only squared modulus is used to construct optical tomograms.
    """
    state.check()
    x=np.asarray(x,dtype=float)
    k=state.kind
    if k in (KINDS[0],KINDS[1]):return np.exp(-1j*(0 if k==KINDS[0] else state.n)*theta)*wave(state,x)
    if k==KINDS[2]:
        q,p=_rotated(state.q0,state.p0,theta)
        return coherent_wave(x,q,p)
    if k in (KINDS[4], KINDS[5]):
        q,p=_rotated(state.q0,state.p0,theta)
        return _cat(x,q,p,1 if k==KINDS[4] else -1)
    if k in (KINDS[7],KINDS[8]):
        return coherent_superposition(state,x,theta)
    if k in (KINDS[9],KINDS[10],KINDS[11],KINDS[13]):
        c=fock_coefficients(state)
        # Fast batched optical slices: angles (Nmu,1), X nodes (1,Nx).
        # This is ONLY used to form w(X|θ), not in the inversion.
        th=np.asarray(theta)
        if x.ndim==2 and x.shape[0]==1 and th.ndim==2 and th.shape[1]==1:
            rot=np.exp(-1j*th[:,0,None]*np.arange(len(c))[None,:])
            return (rot*c[None,:]) @ basis(x[0],len(c)).reshape(len(c),-1)
        nd=max(np.ndim(x),np.ndim(th))
        coef=c.reshape((len(c),)+(1,)*nd)
        idx=np.arange(len(c)).reshape((len(c),)+(1,)*nd)
        return np.sum(coef*np.exp(-1j*idx*th)*basis(x,len(c)),axis=0)
    if k==KINDS[6]:
        # Rotated Fock coefficients e^{-in theta}.
        if state.n0==state.n1:
            return np.exp(-1j*state.n0*theta)*wave(state,x)
        return (np.exp(-1j*state.n0*theta)*_hermite_wave(x,state.n0)
              +state.ratio*np.exp(1j*(state.phase-state.n1*theta))*_hermite_wave(x,state.n1))/np.sqrt(1+state.ratio**2)
    raise ValueError('Squeezed quadrature is computed from the covariance matrix')


def tomogram(state:State,x:np.ndarray,theta:float) -> np.ndarray:
    """Analytic optical tomogram w(X|theta) of a one-mode pure state."""
    state.check()
    x=np.asarray(x,dtype=float)
    if state.kind in (KINDS[3],KINDS[12]):
        c, s=np.cos(theta),np.sin(theta)
        D=(c+state.b*s)**2/state.a+state.a*s*s
        mean=state.q0*c+state.p0*s
        if state.kind == KINDS[3]:
            return np.exp(-((x-mean)**2)/D)/np.sqrt(np.pi*D)
        # Symplectic covariance: (q,p)=(Q/sqrt(a),sqrt(a)*P+bQ/sqrt(a))
        # so every quadrature of a number state is a scaled Hermite density.
        return np.abs(_hermite_wave((x-mean)/np.sqrt(D),state.n))**2/np.sqrt(D)
    if state.kind == KINDS[14]:
        c,s=np.cos(theta),np.sin(theta)
        return np.abs(_hermite_wave(x-state.q0*c-state.p0*s,state.n))**2
    return np.abs(quadrature_amplitude(state,x,theta))**2


def phase_fix(psi:np.ndarray) -> np.ndarray:
    """Fix global phase at the highest-amplitude *sample*; preserves |psi|."""
    psi=np.asarray(psi,dtype=np.complex128)
    if not np.all(np.isfinite(psi)): raise ValueError('Неопределённые значения ψ')
    i=int(np.argmax(np.abs(psi)))
    if abs(psi[i])<=1e-12: raise ValueError('Волновая функция исчезает в выбранном окне')
    return psi*np.exp(-1j*np.angle(psi[i]))


def wave_from_coeff(coeff:np.ndarray, x:np.ndarray, chunk:int=16384)->np.ndarray:
    """Evaluate a Fock expansion without allocating d x N for the whole audio."""
    coeff=np.asarray(coeff,complex)
    x=np.asarray(x,dtype=float)
    result=np.empty(x.shape,dtype=complex)
    flat=x.ravel(); out=result.ravel()
    for i in range(0,len(flat),chunk):
        p=basis(flat[i:i+chunk],len(coeff))
        out[i:i+chunk]=coeff @ p
    return result


def symplectic_tomogram(state:State, X:np.ndarray, mu:float, nu:float)->np.ndarray:
    """w(X|mu,nu) via optical-slice homogeneity, with both arguments real.
    Not defined at mu=nu=0 because that slice is a Dirac distribution.
    """
    if not math.isfinite(mu) or not math.isfinite(nu):
        raise ValueError('Tomogram parameters must be finite')
    scale=math.hypot(mu,nu)
    if scale==0:raise ValueError('(mu,nu)=(0,0) is a delta measure, not a density')
    return tomogram(state,np.asarray(X,dtype=float)/scale,math.atan2(nu,mu))/scale


def signal_tomogram(state:State, X:np.ndarray, alpha:float,beta:float,T:float)->np.ndarray:
    """M(X|alpha,beta) = w(X|T alpha, beta/T) for U_T psi(t)=psi(t/T)/sqrt(T)."""
    if not math.isfinite(T) or T<=0:raise ValueError('T must be positive')
    return symplectic_tomogram(state,X,T*alpha,beta/T)
