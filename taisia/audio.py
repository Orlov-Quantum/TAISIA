"""Complex tomographic signal -> mono carrier, reversible audible I/Q or raw I/Q."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, wave as wavlib
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline
from .physics import State, wave, wave_from_coeff, phase_fix

SAMPLE_RATE=48000
MODE_CARRIER = 'Carrier (mono)'
MODE_IQ_AUDIBLE = 'Stereo I/Q — слышимое (несущая)'
MODE_IQ_RAW = 'Stereo I/Q — исходное (без несущей)'
# Version 0.3 compatibility: old mode is mathematically raw I/Q.
LEGACY_IQ = 'Stereo I/Q'
AUDIO_MODES = (MODE_CARRIER, MODE_IQ_AUDIBLE, MODE_IQ_RAW)


@dataclass
class Render:
    samples: np.ndarray
    seconds: np.ndarray
    envelope: np.ndarray
    samplerate: int
    captured_energy:float
    audio_peak:float
    audio_rms:float
    time_scale:float
    carrier_hz:float
    mode:str
    gain:float
    repeat_pulses:bool=False
    pulse_period:float=0.45


def render(state:State, *,duration:float=4.0, time_scale:float=.34,
           carrier_hz:float=440, gain:float=.6, mode:str='Carrier (mono)',
           coeff:np.ndarray|None=None, sampled_q:np.ndarray|None=None,
           sampled_psi:np.ndarray|None=None, samplerate:int=SAMPLE_RATE,
           repeat_pulses:bool=False,pulse_period:float=.45)->Render:
    state.check()
    if not .25 <= duration <= 30: raise ValueError('Длительность от 0.25 до 30 секунд')
    if not .002 <= time_scale <= 2.5: raise ValueError('Масштаб T от 0.002 до 2.5 сек')
    if repeat_pulses and not .12 <= pulse_period <= 2.0:raise ValueError('Период событий 0.12–2.0 с')
    if not 0 <= carrier_hz <= 4000 or not carrier_hz < .4*samplerate:
        raise ValueError('Несущая должна быть в допустимой полосе')
    if not 0 <= gain <= .9: raise ValueError('Громкость от 0 до 0.9')
    if mode not in (*AUDIO_MODES, LEGACY_IQ):
        raise ValueError('Неизвестный аудиорежим')
    if mode == MODE_IQ_AUDIBLE and carrier_hz < 80:
        raise ValueError('Для слышимого Stereo I/Q установите несущую ≥ 80 Гц (например, 330 Гц)')
    n=int(round(duration*samplerate))
    seconds=np.arange(n,dtype=float)/samplerate
    # Core: one complex tomographic signal. Repetition is an explicitly
    # NONUNITARY PRESENTATION LAYER, not the tomogram of a single state.
    relative_time=(np.mod(seconds,pulse_period)-.5*pulse_period if repeat_pulses
                   else seconds-.5*duration)
    q=relative_time/time_scale
    if (sampled_q is None)!=(sampled_psi is None):
        raise ValueError('Для прямого восстановления требуются и q, и ψ(q)')
    if sampled_q is not None and coeff is not None:
        raise ValueError('Нельзя одновременно передать прямое обращение и коэффициенты Фока')
    if sampled_q is not None:
        xs=np.asarray(sampled_q,float)
        ys=np.asarray(sampled_psi,complex)
        if (xs.ndim!=1 or ys.shape!=xs.shape or len(xs)<8 or
                not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)) or
                not np.all(np.diff(xs)>0)):
            raise ValueError('Некорректная комплексная волна после прямого обращения')
        # Interpolation is an *audio-discretisation* step, NOT a new
        # tomographic inversion or spectral/Fock synthesis.
        fun=CubicSpline(xs,ys,extrapolate=False)
        s=fun(np.clip(q,xs[0],xs[-1]))/np.sqrt(time_scale)
        s[(q<xs[0])|(q>xs[-1])]=0.
    else:
        s=(wave(state,q) if coeff is None else wave_from_coeff(coeff,q))/np.sqrt(time_scale)
    # The direct integral already fixes its global phase using s(t_ref)>0.
    # Do not overwrite that choice with an independent audio gauge.
    if sampled_q is None:
        s=phase_fix(s)
    if repeat_pulses:
        # Energy of ONE mathematical packet, not all repeated copies.
        core_count=min(n,int(round(pulse_period*samplerate)))
        captured=float(trapezoid(np.abs(s[:core_count])**2,dx=1/samplerate))
    else:
        captured=float(trapezoid(np.abs(s)**2,dx=1/samplerate))
    if repeat_pulses:
        k=min(n,int(round(pulse_period*samplerate)))
        peak_env=max(float(np.max(abs(s[:k]))),1e-14)
        boundary_env=max(float(abs(s[0])),float(abs(s[k-1])))/peak_env
        if boundary_env>.02:
            raise ValueError('Стыки повторяемых событий слишком велики: уменьшите T или увеличьте период')
    if captured < .05:
        raise ValueError('Почти вся энергия вне выбранного окна времени. Увеличьте длительность или уменьшите T.')
    # The finite window and chosen carrier constitute an acoustic rendering, not a unitary operation.
    if mode == MODE_CARRIER:
        a = np.real(s * np.exp(2j*np.pi*carrier_hz*seconds))
    elif mode == MODE_IQ_AUDIBLE:
        # Information-preserving complex modulation. BOTH stereo channels are
        # required for decoding: s(t) = (L+iR)*exp(-i*2*pi*fc*t)/gain_scale.
        # A common (not per-channel) volume scaling is applied below.
        z = s * np.exp(2j*np.pi*carrier_hz*seconds)
        a = np.column_stack((z.real,z.imag))
    else:
        # Raw baseband I/Q is reversible, but can have inaudible near-DC content.
        a = np.column_stack((s.real,s.imag))
    peak=float(np.max(np.abs(a)))
    # Preserve signal shape up to one global level adjustment; never per-channel normalize.
    if peak < 1e-15: raise ValueError('Нулевой аудиосигнал')
    a=np.asarray(a/peak*gain,float)
    # Short fade in/out prevents boundary discontinuities and is explicitly nonunitary.
    fade=min(int(.005*samplerate),n//8)
    if fade>1:
        env=np.linspace(0,1,fade,endpoint=False)
        a[:fade]*=env[:,None] if a.ndim==2 else env
        a[-fade:]*=env[::-1,None] if a.ndim==2 else env[::-1]
    return Render(a.astype(np.float32),seconds,s,samplerate,captured,
                  float(np.max(np.abs(a))),float(np.sqrt(np.mean(a*a))),
                  time_scale,carrier_hz,mode,gain,repeat_pulses,pulse_period)


def save_wav(path:str|Path, result:Render, state:State, *,source:str='analytic', diagnostics:dict|None=None) ->tuple[Path,Path]:
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix.lower()!='.wav': path=path.with_suffix('.wav')
    pcm=np.round(np.clip(result.samples,-1,1)*32767).astype('<i2')
    with wavlib.open(str(path),'wb') as f:
        f.setnchannels(1 if pcm.ndim==1 else pcm.shape[1])
        f.setsampwidth(2);f.setframerate(result.samplerate)
        f.writeframes(pcm.tobytes())
    info={
        'application':'TAISIA 0.5','model':'pure analytic one-mode quantum state',
        'input':state.describe(),'method':source,
        'hbar':1,'commutator':'[q,p]=i',
        'complex_signal':'direct optical tomogram inversion s(t); units from T; phase fixed at reference' ,
        'audio_mode':result.mode,'carrier_hz':result.carrier_hz,
        'sample_rate_hz':result.samplerate,'duration_seconds':len(result.samples)/result.samplerate,
        'T_seconds':result.time_scale,'captured_envelope_L2_energy':result.captured_energy,
        'peak_after_normalization':result.audio_peak,'audio_rms':result.audio_rms,
        'gain_setting':result.gain,
        'repeat_pulses_presentation_only':result.repeat_pulses,
        'pulse_period_seconds':result.pulse_period if result.repeat_pulses else None,
        'captured_energy_meaning':'L2 of one packet' if result.repeat_pulses else 'L2 of complete envelope',
        'audio_processing':'common peak normalization, 5 ms end fades, PCM16 quantization; pulse repetition if selected; not unitary',
        'global_phase':'fixed at tomographic reference time for direct method; otherwise at peak sample',
         'stereo_iq_carrier_applied':result.mode == MODE_IQ_AUDIBLE,
        'stereo_iq_note':('L=Re(s*exp(i2πfct)), R=Im(s*exp(i2πfct)); '
                          'recover s by dividing by common gain and exp(i2πfct); '
                          'do not interpret channels independently.'
                          if result.mode == MODE_IQ_AUDIBLE else
                          'Raw baseband L=Re(s), R=Im(s), possibly subaudible. '
                          'Both channels encode the complex signal before PCM16 conversion.'),
        'invertibility_qualification':'Exact in ideal continuous complex stereo; finite window, peak scaling and PCM16 introduce practical changes.' 
    }
    if diagnostics is not None: info['reconstruction_diagnostics']=diagnostics
    metadata=path.with_suffix('.json')
    metadata.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    return path,metadata
