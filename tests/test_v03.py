"""Independent checks for TAISIA 0.3: Manko direct formula + new states + scene mode."""
import unittest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch
from scipy.integrate import trapezoid
from taisia.physics import KINDS, State, wave, tomogram, fock_coefficients
from taisia.direct import signal_from_tomogram, recover_direct
from taisia.audio import render, save_wav


class DirectSignalFormulaTests(unittest.TestCase):
    def test_literal_requested_formula_t0_zero(self):
        state=State(kind=KINDS[2],q0=1.3,p0=-1.9)
        T=.063
        t=np.linspace(-.28,.28,109)
        with patch('taisia.physics.wave', side_effect=AssertionError('no ψ in inversion')):
            s=signal_from_tomogram(state,t,T,0.)
        truth=wave(state,t/T)/np.sqrt(T)
        # Phase convention s(0)>0: use inner product invariant to global phase.
        ov=trapezoid(np.conj(truth)*s,x=t)
        fidelity=abs(ov)**2/(trapezoid(abs(truth)**2,x=t)*trapezoid(abs(s)**2,x=t))
        self.assertGreater(fidelity,1-1e-9)
        self.assertGreater(s[np.argmin(abs(t))].real,0)

    def test_odd_fock_zero_reference_is_disallowed(self):
        state=State(kind=KINDS[1],n=1)
        with self.assertRaises(ValueError):
            signal_from_tomogram(state,np.array([0.,.01]),.01,0.)
        r=recover_direct(state,validate_reference=False)
        self.assertGreater(abs(r.reference_q),.1)
        self.assertLess(r.reference_density_error,1e-6)

    def test_q_deformed_exactly_coherent_at_kappa_zero(self):
        x=np.linspace(-8,8,2501)
        a=State(kind=KINDS[11],q0=2.1,p0=-.7,deformation=0.)
        b=State(kind=KINDS[2],q0=2.1,p0=-.7)
        for theta in (0.,.5,1.4):
            np.testing.assert_allclose(tomogram(a,x,theta),tomogram(b,x,theta),atol=2e-14)

    def test_kerr_zero_is_coherent(self):
        x=np.linspace(-8,8,1001)
        a=State(kind=KINDS[10],q0=2.7,p0=1.1,kerr=0.)
        b=State(kind=KINDS[2],q0=2.7,p0=1.1)
        np.testing.assert_allclose(tomogram(a,x,.79),tomogram(b,x,.79),atol=3e-14)

    def test_photon_added_zero_order_is_coherent(self):
        x=np.linspace(-8,8,1001)
        a=State(kind=KINDS[9],q0=2.7,p0=1.1,m=0)
        b=State(kind=KINDS[2],q0=2.7,p0=1.1)
        np.testing.assert_allclose(tomogram(a,x,.38),tomogram(b,x,.38),atol=3e-14)

    def test_all_new_tomograms_normalized(self):
        x=np.linspace(-13,13,5001)
        for k in KINDS[7:]:
            st=State(kind=k,q0=2.7,p0=.9,m=3,kerr=1.2,deformation=.6)
            for th in (0.,.54,1.4):
                with self.subTest(kind=k,theta=th):
                    w=tomogram(st,x,th)
                    self.assertGreaterEqual(float(np.min(w)),-1e-14)
                    self.assertAlmostEqual(trapezoid(w,x=x),1.,places=8)

    def test_fock_coefficient_normalization(self):
        for k in KINDS[9:12]:
            c=fock_coefficients(State(kind=k,q0=3.,p0=1.1,m=4,kerr=1.5,deformation=.75))
            self.assertAlmostEqual(float(np.vdot(c,c).real),1.,places=13)

    def test_direct_recovery_for_nongaussian(self):
        for kind in (KINDS[9],KINDS[11]):
            st=State(kind=kind,q0=2.2,p0=.7,m=2,deformation=.6)
            with self.subTest(kind=kind):
                recovered=recover_direct(st)
                self.assertGreater(recovered.fidelity_reference_only,.99998)
                self.assertLess(recovered.density_relative_error,1e-5)
                self.assertLess(recovered.reference_density_error,2e-3)


class StageSoundTests(unittest.TestCase):
    def test_repeat_is_distinct_from_core_single_state(self):
        st=State(kind=KINDS[4],q0=3.2,p0=0.)
        q=np.linspace(-9,9,361)
        r=recover_direct(st)
        x=render(st,duration=1.5,time_scale=.012,carrier_hz=440,
                 sampled_q=r.q,sampled_psi=r.signal_q,repeat_pulses=True,pulse_period=.5)
        self.assertAlmostEqual(x.captured_energy,1,places=4)
        self.assertLessEqual(np.max(abs(x.samples)),.60001)
        self.assertGreater(np.linalg.norm(x.samples[:24000]),0.1)
        self.assertGreater(np.linalg.norm(x.samples[24000:48000]),0.1)
        with tempfile.TemporaryDirectory() as folder:
            wav,meta=save_wav(Path(folder)/'test.wav',x,st,source='direct w->s')
            data=json.loads(meta.read_text(encoding='utf8'))
            self.assertTrue(data['repeat_pulses_presentation_only'])
            self.assertAlmostEqual(data['pulse_period_seconds'],.5)
            self.assertTrue(wav.exists())

    def test_repetition_requires_separated_events(self):
        with self.assertRaisesRegex(ValueError,'Стыки'):
            render(State(),duration=2,time_scale=.4,repeat_pulses=True,pulse_period=.2)

    def test_custom_audio_is_state_dependent(self):
        a=render(State(kind=KINDS[1],n=8),duration=2,time_scale=.009,
                 carrier_hz=440,repeat_pulses=True,pulse_period=.4)
        b=render(State(kind=KINDS[2],q0=0,p0=3),duration=2,time_scale=.009,
                 carrier_hz=440,repeat_pulses=True,pulse_period=.4)
        self.assertGreater(np.linalg.norm(a.samples-b.samples),20.)

if __name__=='__main__':unittest.main()
