#!/usr/bin/env python3

"""
The steps taken by ``syncstart``:

- get the maximum audio sample frequency or video frame rate among the inputs using ffprobe
- process and extract sample audio/video clips using ffmpeg with some default and optional filters
- read the two clips into a 1D array and apply optional z-score normalization
- compute offset via correlation using scipy ifft/fft
- print and return result and optionally show in diagrams

Requirements:

- ffmpeg and ffprobe installed
- Python3 with tk (tk is separate on Ubuntu: python3-tk)

References:

- https://ffmpeg.org/ffmpeg-all.html
- https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar
- https://dsp.stackexchange.com/questions/18846/map-time-difference-between-two-similar-videos

Within Python:

from syncstart import file_offset

"""

import matplotlib
matplotlib.use('TkAgg')
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy import fft
from scipy.io import wavfile
import tempfile
import os
import pathlib
import sys
import subprocess

__version__ = '1.1.0'
__author__ = """Roland Puntaier, drolex2"""
__email__ = 'roland.puntaier@gmail.com'

#global
ax = None
video = False
begin = 0
take = 20
normalize = False
denoise = False
lowpass = 0
crop = False
quiet = False
loglevel = 32

ffmpegvideo = 'ffmpeg -loglevel %s -hwaccel auto -ss %s -i "{}" %s -map 0:v -c:v mjpeg -q 1 -f mjpeg "{}"'
ffmpegwav = 'ffmpeg -loglevel %s -ss %s -i "{}" %s -map 0:a -c:a pcm_s16le -ac 1 -f wav "{}"'

audio_filters = {
  'default': 'atrim=0:%s,aresample=%s',
  'lowpass': 'lowpass=f=%s',
  'denoise': 'afftdn=nr=24:nf=-25'
}

video_filters = {
  'default': 'trim=0:%s,fps=%s,format=gray,scale=-1:300',
  'crop': 'crop=400:300',
  'denoise': 'hqdn3d=3:3:2:2'
}

def z_score_normalization(array):
  mean = np.mean(array)
  std_dev = np.std(array)
  normalized_array = (array - mean) / std_dev
  return normalized_array

def get_max_rate(in1,in2):
  probe_audio = 'ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1'.split()
  probe_video = 'ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1'.split()
  command = probe_video if video else probe_audio
  rates = []
  for file in [in1,in2]:
    result = subprocess.run(command+[file], capture_output=True, text=True)
    rates.append( eval(result.stdout.split('=')[1]) )
  return max(rates)

def read_video(input_video):
  # Open input video
  cap = cv2.VideoCapture(str(input_video))
  # Check if the video file was opened successfully
  if not cap.isOpened():
    print('Error: Could not open the video file.')
    return None
  # Get video properties
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  # Initialize list to store difference in average brightness between the left and right halves of each video frame
  brightdiff = []
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    # Split the frame into left and right halves
    left_half = frame[:, :width // 2]
    right_half = frame[:, width // 2:]
    # Calculate the difference in average brightness between the left and right halves
    brightdiff.append( np.mean(right_half) - np.mean(left_half) )
  # Release the video capture object
  cap.release()
  return brightdiff

def in_out(command,infile,outfile):
  if not quiet: #default
    hdr = '-'*len(command)
    print('%s\n%s\n%s'%(hdr,command,hdr))
  ret = os.system(command.format(infile,outfile))
  if 0 != ret:
    sys.exit(ret)

def get_sample(infile,rate):
  outname = pathlib.Path(infile).stem + '_sample'
  with tempfile.TemporaryDirectory() as tempdir:
    outfile = pathlib.Path(tempdir)/(outname)
    if video: #compare video
      filters = [video_filters['default']%(take,rate)]
      if crop:
        filters.append(video_filters['crop'])
      if denoise:
        filters.append(video_filters['denoise'])
      filter_string = '-vf "' + ','.join(filters) + '"'
      in_out(ffmpegvideo%(loglevel,begin,filter_string),infile,outfile)
      s = read_video(outfile)
    else: #compare audio
      filters = [audio_filters['default']%(take,rate)]
      if int(lowpass):
        filters.append(audio_filters['lowpass']%lowpass)
      if denoise:
        filters.append(audio_filters['denoise'])
      filter_string = '-af "' + ','.join(filters) + '"'
      in_out(ffmpegwav%(loglevel,begin,filter_string),infile,outfile)
      r,s = wavfile.read(outfile)
    return s

def fig1(title=None):
  fig = plt.figure(1)
  plt.margins(0, 0.1)
  plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
  plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
  plt.title(title or 'Signal')
  plt.xlabel('Time [seconds]')
  plt.ylabel('Amplitude')
  axs = fig.get_axes()
  global ax
  ax = axs[0]

def show1(fs, s, color=None, title=None, v=None):
  if not color: fig1(title)
  if ax and v: ax.axvline(x=v,color='green')
  plt.plot(np.arange(len(s))/fs, s, color or 'black')
  if not color: plt.show()

def show2(fs,s1,s2,title=None):
  fig1(title)
  show1(fs,s1,'blue')
  show1(fs,s2,'red')
  plt.show()

def corrabs(s1,s2):
  ls1 = len(s1)
  ls2 = len(s2)
  padsize = ls1+ls2+1
  padsize = 2**(int(np.log(padsize)/np.log(2))+1)
  s1pad = np.zeros(padsize)
  s1pad[:ls1] = s1
  s2pad = np.zeros(padsize)
  s2pad[:ls2] = s2
  corr = fft.ifft(fft.fft(s1pad)*np.conj(fft.fft(s2pad)))
  ca = np.absolute(corr)
  xmax = np.argmax(ca)
  return ls1,ls2,padsize,xmax,ca

def cli_parser(**ka):
  import argparse
  parser = argparse.ArgumentParser(
    prog='syncstart',
    description=file_offset.__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--version', action='version', version = __version__)

  if 'in1' not in ka:
    parser.add_argument(
      'in1',
      help='First media file to sync with second.')
  if 'in2' not in ka:
    parser.add_argument(
      'in2',
      help='Second media file to sync with first.')
  if 'video' not in ka:
    parser.add_argument(
      '-v','--video',
      dest='video',
      action='store_true',
      default=False,
      help='Compare video streams. (audio is default)')
  if 'begin' not in ka:
    parser.add_argument(
      '-b','--begin',
      dest='begin',
      action='store',
      default=0,
      help='Begin comparison X seconds into the inputs. (default: 0)')
  if 'take' not in ka:
    parser.add_argument(
      '-t','--take',
      dest='take',
      action='store',
      default=20,
      help='Take X seconds of the inputs to look at. (default: 20)')
  if 'normalize' not in ka:
    parser.add_argument(
      '-n','--normalize',
      dest='normalize',
      action='store_true',
      default=False,
      help='Normalizes audio/video values from each stream.')
  if 'denoise' not in ka:
    parser.add_argument(
      '-d','--denoise',
      dest='denoise',
      action='store_true',
      default=False,
      help='Reduces audio/video noise in each stream.')
  if 'lowpass' not in ka:
    parser.add_argument(
      '-l','--lowpass',
      dest='lowpass',
      action='store',
      default=0,
      help="Audio option: Discards frequencies above the specified Hz,\
      e.g., 300. 0 == off (default)")
  if 'crop' not in ka:
    parser.add_argument(
      '-c','--crop',
      dest='crop',
      action='store_true',
      default=False,
      help='Video option: Crop to 4:3. Helpful when aspect ratios differ.')
  if 'show' not in ka:
    parser.add_argument(
      '-s','--show',
      dest='show',
      action='store_false',
      default=True,
      help='Turn off "show diagrams", in case you are confident.')
  if 'quiet' not in ka:
    parser.add_argument(
      '-q','--quiet',
      dest='quiet',
      action='store_true',
      default=False,
      help='Suppresses standard output except for the CSV result.\
      Output will be: file_to_advance,seconds_to_advance')
  return parser

def file_offset(**ka):
  """CLI interface to sync two media files using their audio or video streams.
  ffmpeg needs to be available.
  """

  parser = cli_parser(**ka)
  args = parser.parse_args().__dict__
  ka.update(args)

  global video,begin,take,normalize,denoise,lowpass,crop,quiet,loglevel
  in1,in2,begin,take = ka['in1'],ka['in2'],ka['begin'],ka['take']
  video,crop,quiet,show = ka['video'],ka['crop'],ka['quiet'],ka['show']
  normalize,denoise,lowpass = ka['normalize'],ka['denoise'],ka['lowpass']
  loglevel = 16 if quiet else 32
  sample_rate = get_max_rate(in1,in2)
  s1,s2 = get_sample(in1,sample_rate),get_sample(in2,sample_rate)
  if normalize:
    s1,s2 = z_score_normalization(s1),z_score_normalization(s2)
  ls1,ls2,padsize,xmax,ca = corrabs(s1,s2)
  if show: show1(sample_rate,ca,title='Correlation',v=xmax/sample_rate)
  sync_text = """
==============================================================================
%s needs 'ffmpeg -ss %s' cut to get in sync
==============================================================================
"""
  if xmax > padsize // 2:
    if show: show2(sample_rate,s1,s2[padsize-xmax:],title='Sample Matchup')
    file,offset = in2,(padsize-xmax)/sample_rate
  else:
    if show: show2(sample_rate,s1[xmax:],s2,title='Sample Matchup')
    file,offset = in1,xmax/sample_rate
  if not quiet: #default
    print(sync_text%(file,offset))
  else: #quiet
    ## print csv: file_to_advance,seconds_to_advance
    print("%s,%s"%(file,offset))
  return file,offset

main = file_offset
if __name__ == '__main__':
    main()
