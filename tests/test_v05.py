"""New states, Fresnel cross-checks, slow single-pulse rendering and profile semantics."""
import unittest
from dataclasses import replace
import numpy as np
from scipy.integrate import trapezoid
from taisia.physics import State, KINDS, wave, tomogram, fock_coefficients
from taisia.direct import DirectConfig, recover_direct
from taisia.forward import optical_from_wave
from taisia.audio import render, MODE_IQ_AUDIBLE
from app import TIMING_PROFILES, SINGLE_PULSE_PROFILES

class NewStatesTests(unittest.TestCase):
    def test_binomial_state_number_limits(self):
        for probability, expected_n in [(0.,0),(1.,7)]:
            a=State(kind=KINDS[13],n=7,binomial_p=probability,phase=.8)
            c=fock_coefficients(a)
            self.assertAlmostEqual(abs(c[expected_n]),1,places=12)
            self.assertAlmostEqual(np.linalg.norm(c),1,places=12)
            for theta in [0,.3,1.1]:
                np.testing.assert_allclose(tomogram(a,np.linspace(-8,8,301),theta),
                    tomogram(State(kind=KINDS[1],n=expected_n),np.linspace(-8,8,301),theta),atol=2e-13)

    def test_undeformed_displacements_and_squeezing_recover_fock(self):
        x=np.linspace(-8,8,500)
        f=State(kind=KINDS[1],n=5)
        for s in [State(kind=KINDS[12],n=5,a=1,b=0,q0=0,p0=0),
                  State(kind=KINDS[14],n=5,q0=0,p0=0)]:
            np.testing.assert_allclose(tomogram(s,x,.8),tomogram(f,x,.8),atol=2e-13)

    def test_quadrature_normalisation_and_fresnel(self):
        x=np.linspace(-12,12,481)
        specs=[(12,dict(n=3,q0=0,p0=.5,a=.75,b=.55)),
               (13,dict(n=7,binomial_p=.5,phase=.95)),
               (14,dict(n=4,q0=1.1,p0=-.5))]
        for index,opts in specs:
            state=State(kind=KINDS[index],**opts)
            for theta in [0,.67,1.4]:
                with self.subTest(state=index,theta=theta):
                    w=tomogram(state,x,theta)
                    self.assertGreaterEqual(float(np.min(w)),-1e-13)
                    self.assertAlmostEqual(float(trapezoid(w,x=x)),1.,places=9)
                    if theta==.67:
                        independent=optical_from_wave(x,wave(state,x),x,theta)
                        error=np.linalg.norm(independent-w)/np.linalg.norm(w)
                        self.assertLess(error,1e-6)

    def test_direct_inversion_three_new_state_families(self):
        cases=[(12,dict(n=3,q0=0,p0=0,a=.8,b=.5)),
               (13,dict(n=8,binomial_p=.5,phase=1.15)),
               (14,dict(n=4,q0=1.,p0=1.3))]
        cfg=DirectConfig(mu_step=.18,quadrature_nodes=192,q_points=181)
        for ix,opts in cases:
            with self.subTest(kind=KINDS[ix]):
                r=recover_direct(State(kind=KINDS[ix],**opts),cfg)
                self.assertGreater(r.fidelity_reference_only,.9999)
                self.assertLess(r.kernel_rank_consistency,.025)
                self.assertLess(r.density_relative_error,1e-6)

class SlowProfileTests(unittest.TestCase):
    def test_presets_are_single_nonrepeating_pulses(self):
        self.assertEqual(len(SINGLE_PULSE_PROFILES),2)
        self.assertEqual(SINGLE_PULSE_PROFILES['Очень медленный · один импульс'],('0.35','10.0'))
        self.assertEqual(SINGLE_PULSE_PROFILES['Предельно медленный · один импульс'],('0.70','20.0'))
        for name,(T,seconds) in SINGLE_PULSE_PROFILES.items():
            with self.subTest(profile=name):
                self.assertGreater(float(T),.04)
                r=render(State(kind=KINDS[13],n=6,binomial_p=.4,phase=.8),
                    duration=float(seconds),time_scale=float(T),carrier_hz=330,
                    gain=.6,mode=MODE_IQ_AUDIBLE,repeat_pulses=False)
                self.assertFalse(r.repeat_pulses)
                self.assertGreater(r.captured_energy,.995)
                self.assertEqual(len(r.samples),int(float(seconds)*48000))
                self.assertGreater(float(np.std(r.samples)),.01)

    def test_duration_outside_new_limit_rejected(self):
        with self.assertRaises(ValueError):render(State(),duration=30.1)

if __name__=='__main__':unittest.main()
