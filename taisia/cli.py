"""Headless TAISIA synthesis; direct Manko tomogram-to-signal inversion by default."""
from __future__ import annotations
import argparse
from .physics import State,KINDS
from .reconstruction import recover
from .direct import recover_direct
from .audio import render,save_wav


def main():
    p=argparse.ArgumentParser(description='TAISIA — direct tomogram-to-audio, pure analytic states')
    p.add_argument('--kind',choices=KINDS,default=KINDS[0])
    p.add_argument('--n',type=int,default=1)
    p.add_argument('--n0',type=int,default=0)
    p.add_argument('--n1',type=int,default=1)
    p.add_argument('--ratio',type=float,default=1)
    p.add_argument('--phase',type=float,default=0)
    p.add_argument('--q0',type=float,default=1.8)
    p.add_argument('--p0',type=float,default=0)
    p.add_argument('--a',type=float,default=1)
    p.add_argument('--b',type=float,default=0)
    p.add_argument('--m',type=int,default=1)
    p.add_argument('--kerr',type=float,default=1.1)
    p.add_argument('--deformation',type=float,default=.45)
    p.add_argument('--binomial-p',type=float,default=.5,help='binomial state parameter 0..1')
    p.add_argument('--T',type=float,default=.34)
    p.add_argument('--seconds',type=float,default=4)
    p.add_argument('--carrier',type=float,default=440)
    p.add_argument('--gain',type=float,default=.6)
    p.add_argument('--mode',choices=('carrier','iq','iq_raw'),default='carrier',
                   help='iq=audible stereo with reversible carrier; iq_raw=unshifted stereo baseband')
    p.add_argument('--repeat',action='store_true',help='Repeat pulse for presentation (not unitary)')
    p.add_argument('--pulse-period',type=float,default=.43)
    p.add_argument('--method',choices=('direct','fock','analytic'),default='direct',
                   help='default: Manko direct integral optical tomogram -> complex signal')
    p.add_argument('--reconstruct',action='store_true',
                   help='deprecated alias for --method fock; retained for compatibility')
    p.add_argument('--dimension',type=int,default=16)
    p.add_argument('--wav',default='TAISIA_demo.wav')
    opts=p.parse_args()
    state=State(kind=opts.kind,n=opts.n,n0=opts.n0,n1=opts.n1,ratio=opts.ratio,
                phase=opts.phase,q0=opts.q0,p0=opts.p0,a=opts.a,b=opts.b,
                m=opts.m,kerr=opts.kerr,deformation=opts.deformation,binomial_p=opts.binomial_p)
    state.check()
    method='fock' if opts.reconstruct else opts.method
    fock=recover(state,dimension=opts.dimension) if method=='fock' else None
    direct=recover_direct(state) if method=='direct' else None
    if fock is not None and fock.tomographic_relative_error >.01:
        p.error('Fock tomography reconstruction error > 1%; increase dimension')
    if direct is not None and (abs(direct.norm-1.)>.02 or direct.density_relative_error>.01 or
                               direct.kernel_rank_consistency>.05 or direct.residual_far_boundary>.08):
        p.error('Direct tomography inversion did not pass numerical consistency checks')
    result=render(state,duration=opts.seconds,time_scale=opts.T,carrier_hz=opts.carrier,
                  gain=opts.gain,mode={'iq':'Stereo I/Q — слышимое (несущая)',
                          'iq_raw':'Stereo I/Q — исходное (без несущей)',
                          'carrier':'Carrier (mono)'}[opts.mode],
                  coeff=fock.coeff if fock else None,
                  sampled_q=direct.q if direct else None,
                  sampled_psi=direct.signal_q if direct else None,
                  repeat_pulses=opts.repeat,pulse_period=opts.pulse_period)
    diagnosis=({'kernel_rank_consistency':direct.kernel_rank_consistency,
                'fidelity_to_analytic_reference':direct.fidelity_reference_only,
                'relative_density_squared_error':direct.density_relative_error,
                'reference_q':direct.reference_q} if direct else
               {'relative_tomogram_squared_error':fock.tomographic_relative_error} if fock else None)
    wav,meta=save_wav(opts.wav,result,state,source={'direct':'Direct Manko integral w -> s(t)',
                       'fock':'Fock inversion from analytic tomograms',
                       'analytic':'Analytic wavefunction'}[method],diagnostics=diagnosis)
    print(f'WAV: {wav}\nJSON: {meta}\nMode: {method}\nCaptured energy: {result.captured_energy:.9f}')
    if direct:print(f'Independent fidelity: {direct.fidelity_reference_only:.9f}; rank residual: {direct.kernel_rank_consistency:.4g}')
    if fock:print(f'Tomographic error: {fock.tomographic_relative_error:.6g}')

if __name__=='__main__':main()
