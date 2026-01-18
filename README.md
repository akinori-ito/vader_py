# vader.py
A voice activity detetion function for Python

This package provides the voice acitivity detection function for a waveform.The return value is a vector of logical, each value of which indicates that the frame (10 msec by default) is the voice part or not.
This program is translated from vadeR, a VAD package for R.

## Example
```Python:
import vader

v = voice_activity("example.wav")
# v is a numpy array of logical
seg = voice_segment(v)
# seg is a DataFrame where each row is a beginning and ending frame of a vocal region
#   begin end
# 1    28 131
# 2   156 291
```


