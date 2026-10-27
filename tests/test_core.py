"""Deterministic physics, inverse tomography, and audio regression tests."""
import json
import tempfile
import unittest
import wave as wavlib
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid

from taisia.physics import State,KINDS,basis,wave,coherent_wave,tomogram,phase_fix,wave_from_coeff,symplectic_tomogram,signal_tomogram
from taisia.reconstruction import recover
from taisia.direct import recover_direct, kernel_column, DirectConfig
from taisia.forward import optical_from_wave
from unittest.mock import patch
from taisia.audio import render,save_wav

X=np.linspace(-12,12,4201)

class StateTests(unittest.TestCase):
    def test_ground_state(self):
        np.testing.assert_allclose(wave(State(),X),np.pi**(-.25)*np.exp(-X**2/2),rtol=1e-13,atol=1e-14)

    def test_fock_orthonormal(self):
        P=basis(X,13)
        overlaps=trapezoid(P[:,None]*P[None,:],x=X,axis=-1)
        np.testing.assert_allclose(overlaps,np.eye(13),atol=2e-12)

    def test_all_state_normalizations(self):
        states=[State(kind=KINDS[1],n=10),State(kind=KINDS[2],q0=-1,p0=1.1),
                State(kind=KINDS[3],a=.5,b=1.1,q0=1,p0=.5),
                State(kind=KINDS[4],q0=1.5,p0=.8),State(kind=KINDS[5],q0=1.5,p0=.8),
                State(kind=KINDS[6],n0=0,n1=10,ratio=1.4,phase=.5)]
        for st in states:
            with self.subTest(kind=st.kind):
                self.assertAlmostEqual(trapezoid(abs(wave(st,X))**2,x=X),1,places=8)

    def test_tomogram_zero_angle(self):
        for st in [State(),State(kind=KINDS[1],n=3),State(kind=KINDS[3],a=.8,b=.3),
                   State(kind=KINDS[4]),State(kind=KINDS[5]),State(kind=KINDS[6],phase=1.1)]:
            with self.subTest(kind=st.kind):
                np.testing.assert_allclose(tomogram(st,X,0),abs(wave(st,X))**2,atol=2e-14)

    def test_fresnel_tomogram_independent_quadrature(self):
        # Independent Fresnel integral derived from the X_theta eigenfunctions.
        # In particular this checks signs of phase and interference terms.
        q=np.linspace(-10,10,3601)
        xs=np.linspace(-3,3,41)
        for st in [State(kind=KINDS[3],a=.6,b=.8,q0=.7,p0=.4),
                   State(kind=KINDS[4],q0=1.6,p0=.9),
                   State(kind=KINDS[5],q0=1.6,p0=.9),
                   State(kind=KINDS[6],n0=0,n1=2,ratio=.7,phase=1.2)]:
            for theta in (.6,1.24):
                mu,nu=np.cos(theta),np.sin(theta)
                integrand=wave(st,q)*np.exp(1j*mu*q*q/(2*nu))
                amp=trapezoid(integrand[None,:]*np.exp(-1j*xs[:,None]*q[None,:]/nu),x=q,axis=1)
                fresnel=np.abs(amp)**2/(2*np.pi*abs(nu))
                np.testing.assert_allclose(fresnel,tomogram(st,xs,theta),atol=2e-9,rtol=3e-8)

    def test_tomogram_normalization_and_nonnegativity(self):
        states=[State(),State(kind=KINDS[1],n=6),State(kind=KINDS[2],q0=2,p0=-1),
                State(kind=KINDS[3],a=.6,b=.8),State(kind=KINDS[5],q0=2,p0=.7),
                State(kind=KINDS[6],n0=1,n1=3,phase=.7)]
        for st in states:
            for theta in [0,.41,1.23,np.pi/2,2.2,np.pi]:
                w=tomogram(st,X,theta)
                with self.subTest(state=st.kind,angle=theta):
                    self.assertGreaterEqual(w.min(),-1e-14)
                    self.assertAlmostEqual(trapezoid(w,x=X),1,places=8)

    def test_symplectic_homogeneity(self):
        st=State(kind=KINDS[2],q0=1.2,p0=-.6)
        x=np.linspace(-7,7,801)
        for mu,nu in ((1.,0.),(0.,1.),(.7,-.2),(-.4,1.8)):
            for lam in (-2.3,.6,2.0):
                left=symplectic_tomogram(st,lam*x,lam*mu,lam*nu)
                right=symplectic_tomogram(st,x,mu,nu)/abs(lam)
                np.testing.assert_allclose(left,right,atol=1e-14)

    def test_time_frequency_tomogram_unitary_covariance(self):
        st=State(kind=KINDS[2],q0=.9,p0=1.4)
        xs=np.linspace(-3,3,801)
        for T in (.21,.7,1.25):
            # M(X|1,0)=|s(X)|² and M(X|0,1)=T * |psi_tilde(T X)|².
            pos=signal_tomogram(st,xs,1,0,T)
            np.testing.assert_allclose(pos,np.abs(wave(st,xs/T))**2/T,atol=1e-14)
            mom=signal_tomogram(st,xs,0,1,T)
            expect=T*np.exp(-(T*xs-st.p0)**2)/np.sqrt(np.pi)
            np.testing.assert_allclose(mom,expect,atol=1e-14)

    def test_zero_symplectic_slice_rejected(self):
        with self.assertRaises(ValueError):symplectic_tomogram(State(),X,0,0)

    def test_fock_independent_theta(self):
        st=State(kind=KINDS[1],n=4)
        np.testing.assert_allclose(tomogram(st,X,.57),tomogram(st,X,0),atol=2e-15)

    def test_coherent_quadrature_displacement(self):
        st=State(kind=KINDS[2],q0=1.75,p0=-.8)
        for theta in (0,np.pi/3,np.pi/2):
            p=tomogram(st,X,theta)
            expect=st.q0*np.cos(theta)+st.p0*np.sin(theta)
            self.assertAlmostEqual(trapezoid(X*p,x=X),expect,places=7)
            self.assertAlmostEqual(trapezoid((X-expect)**2*p,x=X),.5,places=7)

    def test_squeezed_covariance(self):
        st=State(kind=KINDS[3],q0=.4,p0=.8,a=.7,b=.9)
        for theta in (0,.4,.91,np.pi/2):
            p=tomogram(st,X,theta)
            mu=st.q0*np.cos(theta)+st.p0*np.sin(theta)
            D=((np.cos(theta)+st.b*np.sin(theta))**2/st.a
               +st.a*np.sin(theta)**2)
            self.assertAlmostEqual(trapezoid((X-mu)**2*p,x=X),D/2,places=7)

    def test_cat_interference_not_mixture(self):
        plus=State(kind=KINDS[4],q0=1.0,p0=0.0)
        minus=State(kind=KINDS[5],q0=1.0,p0=0.0)
        self.assertGreater(tomogram(plus,np.array([0.]),0)[0],.1)
        self.assertAlmostEqual(tomogram(minus,np.array([0.]),0)[0],0,places=12)

    def test_fock_node_sign(self):
        st=State(kind=KINDS[1],n=1)
        s=wave(st,np.array([-1.,0.,1.]))
        self.assertAlmostEqual(s[1].real,0)
        self.assertAlmostEqual(s[0].real,-s[2].real,places=12)

    def test_phase_fix_invariance(self):
        psi=wave(State(kind=KINDS[2],p0=1.2),X)
        np.testing.assert_allclose(phase_fix(psi),phase_fix(psi*np.exp(1j*1.17)),atol=5e-15)

    def test_invalid_cat_zero_alpha(self):
        with self.assertRaises(ValueError):wave(State(kind=KINDS[5],q0=0,p0=0),X)

    def test_invalid_superposition_cancellation(self):
        with self.assertRaises(ValueError):wave(State(kind=KINDS[6],n0=1,n1=1,phase=np.pi),X)

    def test_coherent_frequency_from_q_phase(self):
        q=np.linspace(-1,1,300)
        y=coherent_wave(q,.8,2.1)
        slope=np.diff(np.unwrap(np.angle(y)))/np.diff(q)
        np.testing.assert_allclose(slope,2.1,atol=1e-12)

class TomographyReconstructionTests(unittest.TestCase):
    def _compare(self,st,d,limit):
        result=recover(st,dimension=d)
        self.assertLess(result.tomographic_relative_error,limit)
        expected=wave(st,X)
        estimate=wave_from_coeff(result.coeff,X)
        fidelity=abs(trapezoid(np.conj(expected)*estimate,x=X))**2
        self.assertGreater(fidelity,1-2*limit)

    def test_vacuum(self):self._compare(State(),12,2e-5)
    def test_fock_node(self):self._compare(State(kind=KINDS[1],n=1),12,2e-5)
    def test_superposition_phase(self):self._compare(State(kind=KINDS[6],ratio=.73,phase=1.2),12,2e-5)
    def test_coherent(self):self._compare(State(kind=KINDS[2],q0=1.7,p0=1.1),16,2e-5)
    def test_even_cat(self):self._compare(State(kind=KINDS[4],q0=1.8,p0=.5),16,2e-5)
    def test_odd_cat(self):self._compare(State(kind=KINDS[5],q0=1.8,p0=.5),16,2e-5)
    def test_squeezed(self):self._compare(State(kind=KINDS[3],a=.6,b=.7),20,6e-5)
    def test_recovery_sampling_validation(self):
        with self.assertRaises(ValueError):recover(State(),dimension=21)
        with self.assertRaises(ValueError):recover(State(),dimension=8,angle_count=7)

class AudioTests(unittest.TestCase):
    def test_time_scale_unitarity(self):
        for T,duration in [(.2,3),(.6,8)]:
            result=render(State(),time_scale=T,duration=duration,gain=.6)
            self.assertAlmostEqual(result.captured_energy,1,places=7)

    def test_audio_safe_bound(self):
        for st in [State(),State(kind=KINDS[5]),State(kind=KINDS[3],a=.5,b=.8)]:
            for mode in ('Carrier (mono)','Stereo I/Q'):
                a=render(st,mode=mode,carrier_hz=440)
                self.assertTrue(np.all(np.isfinite(a.samples)))
                self.assertLessEqual(np.max(abs(a.samples)),.600001)
                self.assertGreater(np.max(abs(a.samples)),.1)
                self.assertEqual(a.samples.ndim,1 if mode=='Carrier (mono)' else 2)

    def test_pcm16_wav_headers_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            r=render(State(kind=KINDS[1],n=1),duration=.5,mode='Stereo I/Q')
            wav,meta=save_wav(Path(folder)/'test.wav',r,State(kind=KINDS[1],n=1))
            with wavlib.open(str(wav),'rb') as fh:
                self.assertEqual(fh.getnchannels(),2)
                self.assertEqual(fh.getsampwidth(),2)
                self.assertEqual(fh.getframerate(),48000)
                self.assertEqual(fh.getnframes(),24000)
            data=json.loads(meta.read_text(encoding='utf-8'))
            self.assertEqual(data['hbar'],1)
            self.assertEqual(data['model'],'pure analytic one-mode quantum state')

    def test_wave_not_interpreted_as_periodic_wavetable(self):
        a=render(State(),duration=4,carrier_hz=440)
        # Carrier is Gaussian-gated, decays strongly near both window endpoints.
        self.assertLess(np.max(abs(a.samples[:100])),1e-4)
        self.assertLess(np.max(abs(a.samples[-100:])),1e-4)
        self.assertGreater(np.max(abs(a.samples[len(a.samples)//2-500:len(a.samples)//2+500])),.55)

    def test_invalid_duration_or_gain(self):
        with self.assertRaises(ValueError):render(State(),gain=1.0)
        with self.assertRaises(ValueError):render(State(),duration=31)

class DirectKernelTests(unittest.TestCase):
    def _compare(self, st, threshold=1e-6):
        r=recover_direct(st)
        self.assertAlmostEqual(r.norm,1.,delta=.001)
        self.assertLess(r.density_relative_error,1e-5)
        self.assertGreater(r.fidelity_reference_only,1-threshold)
        self.assertLess(r.kernel_rank_consistency,.002)
        self.assertLess(r.residual_far_boundary,.03)
        return r

    def test_direct_vacuum(self):self._compare(State())
    def test_direct_high_fock_n12(self):self._compare(State(kind=KINDS[1],n=12))
    def test_direct_fock_sign_across_node(self):
        r=self._compare(State(kind=KINDS[1],n=1))
        self.assertLess((r.psi[np.argmin(abs(r.q+1))]/r.psi[np.argmin(abs(r.q-1))]).real,-.99)
    def test_direct_coherent_displaced(self):self._compare(State(kind=KINDS[2],q0=2.4,p0=-1.9))
    def test_direct_strong_squeezed_chirped(self):self._compare(State(kind=KINDS[3],a=.35,b=3,q0=3,p0=2))
    def test_direct_even_and_odd_cats(self):
        self._compare(State(kind=KINDS[4],q0=2.2,p0=.8))
        self._compare(State(kind=KINDS[5],q0=1.9,p0=.8))
    def test_direct_complex_fock_coherences(self):
        self._compare(State(kind=KINDS[6],n0=1,n1=12,ratio=.75,phase=1.3))
    def test_direct_only_tomogram_not_known_wave_for_gaussian(self):
        st=State(kind=KINDS[3],a=.8,b=.4)
        with patch('taisia.physics.wave', side_effect=AssertionError('analytic psi must not be called')):
            r=recover_direct(st,validate_reference=False)
        self.assertAlmostEqual(r.norm,1.,places=6)
        self.assertTrue(np.isnan(r.fidelity_reference_only))

    def test_direct_kernel_matches_density_matrix_element(self):
        st=State(kind=KINDS[4],q0=1.8,p0=.65)
        ref=.3
        q=np.array([-1.3,-.2,0,.5,2.1])
        actual=kernel_column(st,q,ref)
        expected=wave(st,q)*np.conj(wave(st,np.array([ref]))[0])
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=5e-5)

    def test_forward_tomogram_from_recovered_wave(self):
        st=State(kind=KINDS[5],q0=1.6,p0=.8)
        r=recover_direct(st)
        xs=np.linspace(-3,3,150)
        for th in (.4, .8, 1.5):
            actual=optical_from_wave(r.q,r.psi,xs,th)
            np.testing.assert_allclose(actual,tomogram(st,xs,th),atol=2e-5)

    def test_audio_direct_envelope_not_wave_reference(self):
        st=State(kind=KINDS[6],n0=0,n1=2,ratio=.8,phase=.6)
        r=recover_direct(st)
        output=render(st, sampled_q=r.q, sampled_psi=r.psi,duration=2.0,time_scale=.24)
        self.assertGreater(output.captured_energy,.95)
        self.assertLessEqual(output.audio_peak,.600001)
        with self.assertRaises(ValueError):
            render(st,sampled_q=r.q, sampled_psi=r.psi,coeff=np.ones(4))

    def test_invalid_direct_integral_parameters(self):
        with self.assertRaises(ValueError):
            recover_direct(State(),DirectConfig(mu_step=.01))

if __name__=='__main__':unittest.main()
