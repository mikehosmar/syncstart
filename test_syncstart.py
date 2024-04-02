#!/usr/bin/env python

"""
Tests for `syncstart` package.
"""

import pytest

from syncstart import *
from scipy.io import wavfile
import numpy as np

def to_files(sr,s1,s2,p):
  wav1 = p / "wav1.wav"
  wav2 = p / "wav2.wav"
  wavfile.write(wav1, sr, s1.astype(np.int16))
  wavfile.write(wav2, sr, s2.astype(np.int16))
  return str(wav1),str(wav2)

@pytest.fixture
def rndwav(tmp_path):
  rng = np.random.RandomState(0)
  sr = 32000
  s1 = rng.standard_normal(sr)
  s2 = np.concatenate([rng.standard_normal(sr//2), s1])
  return to_files(sr,s1,s2,tmp_path)

#import tempfile
#import pathlib
#tmp_path = pathlib.Path(tempfile.mkdtemp())
#rndwav = rndwav(tmp_path)

def test_rnd0(rndwav):
  file,offset = file_offset(
     in1=rndwav[1]
    ,in2=rndwav[0]
    ,take=1.5
    ,normalize=False
    ,denoise=False
    ,lowpass=0
    ,show=False
  )
  assert offset==0.5 and file==rndwav[1]

def test_rnd1(rndwav):
  file,offset = file_offset(
     in1=rndwav[0]
    ,in2=rndwav[1]
    ,take=1.5
    ,normalize=False
    ,denoise=False
    ,lowpass=0
    ,show=False
  )
  assert offset==0.5 and file==rndwav[1]

def test_rnd00(rndwav):
  file,offset = file_offset(
     in1=rndwav[0]
    ,in2=rndwav[0]
    ,take=1.5
    ,normalize=False
    ,denoise=False
    ,lowpass=0
    ,show=False
  )
  assert offset==0.0 and file==rndwav[0]

def test_rnd11(rndwav):
  file,offset = file_offset(
     in1=rndwav[1]
    ,in2=rndwav[1]
    ,take=1.5
    ,normalize=False
    ,denoise=False
    ,lowpass=0
    ,show=False
  )
  assert offset==0.0 and file==rndwav[1]

test_wav_offset = 2.3113832199546485
@pytest.fixture
def tstwav(tmp_path):
  sr,s = wavfile.read('test.wav')
  lens2 = len(s)//2
  #lens2/sr # 2.3113832199546485
  s1 = s
  s2 = np.concatenate([s1[lens2:,:],s1])
  #wavfile.write('test2.wav', sr, s2.astype(np.int16))
  #mono:
  #wavfile.write('test1.wav', sr, s[:,1].astype(np.int16))
  return to_files(sr,s1,s2,tmp_path)

# testwav = testwav(tmp_path)

def test_tst0(tstwav):
  file,offset = file_offset(
     in1=tstwav[0]
    ,in2=tstwav[1]
    ,show=False
  )
  assert abs(offset-test_wav_offset)<0.001 and file==tstwav[1]

def test_tst1(tstwav):
  file,offset = file_offset(
     in1=tstwav[0]
    ,in2=tstwav[1]
    ,show=False
  )
  assert abs(offset-test_wav_offset)<0.001 and file==tstwav[1]

