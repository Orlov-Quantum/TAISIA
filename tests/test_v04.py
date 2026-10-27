"""TAISIA 0.4 regression tests for audible reversible I/Q and timing profiles."""
import json
import tempfile
import unittest
import wave
from pathlib import Path
from scipy.signal import butter, sosfilt
import numpy as np
from taisia.audio import (render, save_wav, MODE_CARRIER, MODE_IQ_AUDIBLE,
                          MODE_IQ_RAW, LEGACY_IQ)
from taisia.physics import State, KINDS


class IQMonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st=State(kind=KINDS[0])
        kwargs=dict(duration=3.,time_scale=.025,carrier_hz=330,gain=.6,
                    repeat_pulses=True,pulse_period=1.0)
        cls.raw=render(cls.st,mode=MODE_IQ_RAW,**kwargs)
        cls.audible=render(cls.st,mode=MODE_IQ_AUDIBLE,**kwargs)

    def test_audible_has_two_nonempty_channels(self):
        x=self.audible.samples
        self.assertEqual(x.ndim,2)
        self.assertEqual(x.shape[1],2)
        self.assertGreater(float(np.std(x[:,0])),.02)
        self.assertGreater(float(np.std(x[:,1])),.02)
        self.assertTrue(np.isfinite(x).all())

    def test_audible_mode_carries_audio_band_energy(self):
        # Highpass below 80 Hz: slow baseband has practically no musical pitch.
        filt=butter(5,80,fs=48000,btype='highpass',output='sos')
        ratio=lambda x:float(np.linalg.norm(sosfilt(filt,x,axis=0))/np.linalg.norm(x))
        self.assertLess(ratio(self.raw.samples),.22)
        self.assertGreater(ratio(self.audible.samples),.90)

    def test_complex_carrier_is_reversible(self):
        base=self.raw.samples[:,0].astype(float)+1j*self.raw.samples[:,1]
        shifted=self.audible.samples[:,0].astype(float)+1j*self.audible.samples[:,1]
        demod=shifted*np.exp(-2j*np.pi*330*self.audible.seconds)
        # Each rendering has one positive real gain factor. Up to that scalar,
        # both encode the SAME complex envelope, with relative phase preserved.
        scale=(np.vdot(base,demod)/np.vdot(base,base))
        err=np.linalg.norm(demod-scale*base)/np.linalg.norm(demod)
        self.assertLess(float(err),2e-6)
        self.assertAlmostEqual(scale.imag,0,places=6)
        self.assertGreater(scale.real,0)

    def test_legacy_raw_remains_raw_for_old_clients(self):
        legacy=render(self.st,duration=.5,time_scale=.025,mode=LEGACY_IQ)
        raw=render(self.st,duration=.5,time_scale=.025,mode=MODE_IQ_RAW)
        np.testing.assert_array_equal(legacy.samples,raw.samples)

    def test_audio_monitor_carrier_zero_rejected(self):
        with self.assertRaisesRegex(ValueError,'несущую'):
            render(self.st,mode=MODE_IQ_AUDIBLE,carrier_hz=0.)

    def test_wav_export_channel_count_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            wav_path,meta=save_wav(Path(td)/'iq.wav',self.audible,self.st)
            with wave.open(str(wav_path),'rb') as f:
                self.assertEqual(f.getnchannels(),2)
                self.assertEqual(f.getframerate(),48000)
                raw=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').reshape(-1,2)
            self.assertGreater(np.max(np.abs(raw)),1000)
            self.assertGreater(float(np.std(raw[:,1])),100)
            m=json.loads(meta.read_text(encoding='utf-8'))
            self.assertEqual(m['application'],'TAISIA 0.5')
            self.assertTrue(m['stereo_iq_carrier_applied'])
            self.assertIn('exp',m['stereo_iq_note'])

class TimingProfilesTests(unittest.TestCase):
    def test_profiles_duration_windows_are_physically_valid(self):
        # No GUI required, but checks actual preset parameters used by GUI.
        from app import TIMING_PROFILES
        self.assertEqual(len(TIMING_PROFILES),3)
        self.assertEqual(TIMING_PROFILES['Протяжённые пакеты — рекомендую'],('0.025','1.0'))
        for name,(T,p) in TIMING_PROFILES.items():
            with self.subTest(profile=name):
                x=render(State(),duration=3,time_scale=float(T),pulse_period=float(p),
                         repeat_pulses=True,mode=MODE_IQ_AUDIBLE,carrier_hz=330)
                self.assertGreater(x.captured_energy,.98)
                self.assertGreater(x.audio_rms,.01)

if __name__=='__main__':unittest.main()
