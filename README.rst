=======================================
syncstart(1) Version 1.1.0 \| syncstart
=======================================

SYNOPSIS
========

Command line help::

    usage: syncstart [-h] [--version] [-v] [-b BEGIN] [-t TAKE] [-n] [-d] [-l LOWPASS] [-c] [-s] [-q] in1 in2
    
    CLI interface to sync two media files using their audio or video streams.
      ffmpeg needs to be available.
      
    
    positional arguments:
      in1                   First media file to sync with second.
      in2                   Second media file to sync with first.
    
    options:
      -h, --help            show this help message and exit
      --version             show program's version number and exit
      -v, --video           Compare video streams. (audio is default)
      -b BEGIN, --begin BEGIN
                            Begin comparison X seconds into the inputs. (default: 0)
      -t TAKE, --take TAKE  Take X seconds of the inputs to look at. (default: 20)
      -n, --normalize       Normalizes audio/video values from each stream.
      -d, --denoise         Reduces audio/video noise in each stream.
      -l LOWPASS, --lowpass LOWPASS
                            Audio option: Discards frequencies above the specified Hz, e.g., 300. 0 == off (default)
      -c, --crop            Video option: Crop to 4:3. Helpful when aspect ratios differ.
      -s, --show            Suppress "show diagrams", in case you are confident.
      -q, --quiet           Suppresses standard output except for the CSV result. Output will be:
                            file_to_advance,seconds_to_advance


DESCRIPTION
===========


The steps taken by ``syncstart``:

- get the maximum audio sample frequency or video frame rate among the inputs using ffprobe
- process and extract sample audio/video clips using ffmpeg with some default and optional filters
- read the two clips into a 1D array and apply optional z-score normalization
- compute offset via correlation using scipy ifft/fft
- print ffmpeg/ffprobe output or optionally quiet that
- show diagrams to allow MANUAL correction using ZOOM or optionally suppress that
- print result

Requirements:

- ffmpeg and ffprobe installed
- Python3 with tk (tk is separate on Ubuntu: python3-tk)

References:

- https://ffmpeg.org/ffmpeg-all.html
- https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar
- https://dsp.stackexchange.com/questions/18846/map-time-difference-between-two-similar-videos

Within Python:

from syncstart import file_offset




INSTALLATION
============

To install for user only, do::

   pip install --user syncstart

EXAMPLES
--------

::

  # compute audio offset with default settings:
  syncstart from_s10.m4a from_gopro.m4p

  # compute audio offset using first 10 seconds with denoising, normalization and a 300 Hz lowpass filter:
  syncstart video1.mp4 video2.mkv -t 10 -dnl 300

  # compute video offset using first 20 seconds, don't show plots, only output final result:
  syncstart video1.mp4 video2.mkv -vsq

  # compute video offset using seconds 15 to 25 with denoising, cropping and normalization:
  syncstart video1.mp4 video2.mkv -b 15 -t 10 -vdcn

License
-------

MIT

