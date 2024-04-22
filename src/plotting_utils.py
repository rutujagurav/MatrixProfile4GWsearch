import matplotlib.pyplot as plt
from gwpy.plot import Plot
from preprocess import preprocess_ts

def plot_event_ts(hdata, ldata, event_gps, preprocess_params=None, titlestr=None):

    if preprocess_params:
      hdata = preprocess_ts(hdata, 
                            bandpass_freq = preprocess_params['bandpass_freq'], 
                            notch_freqs = preprocess_params['notch_freqs'])
      ldata = preprocess_ts(ldata, 
                            bandpass_freq = preprocess_params['bandpass_freq'], 
                            notch_freqs = preprocess_params['notch_freqs'])
    ## Plot timeseries
    plot = Plot(figsize=[12, 6])
    ax = plot.gca()
    ax.plot(hdata, label='LIGO-Hanford', color='gwpy:ligo-hanford')
    ax.plot(ldata, label='LIGO-Livingston', color='gwpy:ligo-livingston')
    ax.set_xscale('seconds', epoch=event_gps)
    ax.set_ylabel('Amplitude [strain]')
    ax.set_title(titlestr, loc='right')
    # if preprocess:
    #   ax.text(1.0, 1.01, r'Whitened data, 30-350 Hz bandpass, notches at 60, 120, 180 Hz',
    #      transform=ax.transAxes, ha='right')
    # else:
    #   ax.text(1.0, 1.01, 'Whitened data',
    #      transform=ax.transAxes, ha='right')
    ax.legend()
    plt.tight_layout()

def plot_event_asd(hdata, ldata, preprocess_params=None, titlestr=None):
    
    if preprocess_params:
      hdata = preprocess_ts(hdata, 
                            bandpass_freq = preprocess_params['bandpass_freq'], 
                            notch_freqs = preprocess_params['notch_freqs'])
      ldata = preprocess_ts(ldata, 
                            bandpass_freq = preprocess_params['bandpass_freq'], 
                            notch_freqs = preprocess_params['notch_freqs'])
    
    ## Plot ASD
    hdata_asd = hdata.asd(2,1)
    ldata_asd = ldata.asd(2,1)
    
    plot = Plot(figsize=[5, 5])
    ax = plot.gca()
    ax.plot(hdata_asd, label='LIGO-Hanford', color='gwpy:ligo-hanford')
    ax.plot(ldata_asd, label='LIGO-Livingston', color='gwpy:ligo-livingston')
    # ax.set_xlim(10, 2000)
    # ax.set_ylim(1e-25, 1e-18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'GW strain ASD [strain$/\sqrt{\mathrm{Hz}}$]')
    ax.set_title(titlestr, loc='right')
    # ax.legend(frameon=False, bbox_to_anchor=(1., 1.), loc='lower right', ncol=2)
    ax.legend()
    plt.tight_layout()

def plot_event_qtransform(hdata, ldata, event_gps, delta_outseg=(0.5,0.5), freq_range=(30,350), qrange=(4,64), whiten=False, clim={'H1': None, 'L1': None}, titlestr=None):
    outseg_start_gps, outseg_end_gps = event_gps-delta_outseg[0], event_gps+delta_outseg[1]
    # print("data",from_gps(hdata.span[0]), from_gps(hdata.span[-1]))
    # print("outseg", from_gps(outseg_start_gps), from_gps(outseg_end_gps))

    h_qspecgram = hdata.q_transform(outseg=(outseg_start_gps, outseg_end_gps), frange=freq_range, qrange=qrange, whiten=whiten)

    plot = h_qspecgram.plot(figsize=[8, 4])
    ax = plot.gca()
    ax.set_xscale('seconds', epoch=event_gps)
    ax.set_yscale('log')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title("LIGO Hanford "+titlestr)
    ax.grid(True, axis='y', which='both')
    ax.colorbar(cmap='viridis', label='Normalized energy')
    if clim['H1']:
      plot.colorbars[0].mappable.set_clim(*clim['H1'])
      plot.refresh()

    l_qspecgram = ldata.q_transform(outseg=(outseg_start_gps, outseg_end_gps), frange=freq_range, qrange=qrange, whiten=whiten)

    plot = l_qspecgram.plot(figsize=[8, 4])
    ax = plot.gca()
    ax.set_xscale('seconds', epoch=event_gps)
    ax.set_yscale('log')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title("LIGO Livingston "+titlestr)
    ax.grid(True, axis='y', which='both')
    ax.colorbar(cmap='viridis', label='Normalized energy')
    if clim['L1']:
      plot.colorbars[0].mappable.set_clim(*clim['L1'])
      plot.refresh()

