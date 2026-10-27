"""Independent forward FrFT/Fresnel check on a recovered complex waveform.

The kernel reconstruction itself never calls this module; it is
only a diagnostic when a displayed quadrature slice is desired.
"""
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline


def optical_from_wave(q:np.ndarray, psi:np.ndarray, X:np.ndarray, theta:float)->np.ndarray|None:
    """Return |<X_theta|psi>|² using the Fresnel integral.

    Very small |sin(theta)| is deliberately not evaluated from a coarse
    q mesh, because it is numerically oscillatory in that limit.
    """
    q=np.asarray(q,float);psi=np.asarray(psi,complex);X=np.asarray(X,float)
    if abs(np.sin(theta)) < .2:
        # θ exactly 0 and π have known coordinate-parity sections.
        if abs(theta)<1e-10: return abs(CubicSpline(q,psi)(X))**2
        if abs(theta-np.pi)<1e-10: return abs(CubicSpline(q,psi)(-X))**2
        return None
    nu=np.sin(theta);mu=np.cos(theta)
    wave_phase=psi*np.exp(.5j*(mu/nu)*q*q)
    out=np.empty(X.shape,float)
    for k in range(0,len(X),80):
        xx=X[k:k+80]
        integrands=wave_phase[None,:]*np.exp(-1j*xx[:,None]*q[None,:]/nu)
        amp=trapezoid(integrands,x=q,axis=1)
        out[k:k+len(xx)]=abs(amp)**2/(2*np.pi*abs(nu))
    return out
