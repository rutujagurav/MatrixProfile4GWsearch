import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from gwpy.signal import filter_design
# def preprocess_ts(data, bandpass_freq=(35, 350), notch_freqs=(60, 120, 180)):
#   print("Bandppass: ", bandpass_freq)
#   bp = filter_design.bandpass(bandpass_freq[0], bandpass_freq[1], data.sample_rate)
#   print(f"Notches: {notch_freqs}")
#   notches = [filter_design.notch(line, data.sample_rate) for
#            line in notch_freqs]
#   zpk = filter_design.concatenate_zpks(bp, *notches)
#   filt_data = data.filter(zpk, filtfilt=True)
#   filt_data = filt_data.crop(*filt_data.span.contract(1))
#   return filt_data

def preprocess_ts(data, bandpass_freq=(35, 350), notch_freqs=(60, 120, 180)):
  filters = []
  # print("Bandppass: ", bandpass_freq)
  if bandpass_freq: 
    bp = filter_design.bandpass(bandpass_freq[0], bandpass_freq[1], data.sample_rate)
    filters.append(bp)
  if notch_freqs:
    # print(f"Notches: {notch_freqs}")
    for line in notch_freqs:
      filters.append(filter_design.notch(line, data.sample_rate))
  zpk = filter_design.concatenate_zpks(*filters)
  filt_data = data.filter(zpk, filtfilt=True)
  filt_data = filt_data.crop(*filt_data.span.contract(1))
  return filt_data

def znorm_ts(data):
  z_normed_vals = (data.value - np.mean(data.value)) / data.value.std()
  return TimeSeries(z_normed_vals, t0=data.t0, dt=data.dt, name=data.name)