"""A few fast sanity checks for startup; run full tests separately."""
import unittest
import numpy as np
from scipy.integrate import trapezoid
from taisia.physics import State,KINDS,tomogram
from taisia.direct import signal_from_tomogram
from taisia.audio import render

class StartupSmokeTests(unittest.TestCase):
    def test_manko_direct_vacuum(self):
        st=State()
        x=np.linspace(-3.,3.,31)
        s=signal_from_tomogram(st,x,.2,0.) # q-range -15..15, but formula stable
        expected=np.pi**(-.25)*np.exp(-.5*(x/.2)**2)/np.sqrt(.2)
        np.testing.assert_allclose(s,expected,atol=1e-7,rtol=1e-5)
    def test_nongaussian_tomogram_positive_and_normalized(self):
        st=State(kind=KINDS[11],q0=2.,p0=.3,deformation=.5)
        x=np.linspace(-10,10,1000)
        y=tomogram(st,x,.42)
        self.assertGreaterEqual(float(np.min(y)),0.)
        self.assertAlmostEqual(trapezoid(y,x=x),1.,places=8)
    def test_pulse_audio_safe(self):
        st=State(kind=KINDS[1],n=8)
        r=render(st,time_scale=.009,repeat_pulses=True,pulse_period=.45,duration=.8)
        self.assertLessEqual(float(np.max(abs(r.samples))),.90001)

if __name__=='__main__':unittest.main()
