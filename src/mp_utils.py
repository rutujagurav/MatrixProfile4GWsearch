import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime

from gwpy.timeseries import TimeSeries, TimeSeriesDict
from gwpy.time import to_gps, from_gps
from gwpy.plot import Plot

from scipy.signal import find_peaks

import stumpy
from stumpy.floss import _cac
import pyscamp


## Plotting
def plot_regimes(data, cac, regime_locations, n_regimes=3):
    cac = np.concatenate([cac, np.ones(len(data.value)-len(cac))*[np.nan]])
    cac = TimeSeries(cac, times=list(data.times), name='corrected arc curve')
    
    plot = Plot(data, cac, separate=True, sharex=True)
    
    ax = plot.gca()
    for i in range(n_regimes-1):
        x_loc = data.times[regime_locations[i]].value
        ax.axvline(x=x_loc, color='red', linestyle='--')
    # ax.set_title(data.name)
    
    return ax

def plot_distance_profile(data, dp, log_yscale=False):
    og_dp_len = len(dp)
    padding_entries = np.ones(len(data.value)-len(dp))*[np.nan]
    dp = np.concatenate([dp, padding_entries])
    dp = TimeSeries(dp, times=list(data.times), name='Distance Profile')
    plot = Plot(data, dp, separate=True, sharex=True)
    for i, ax in enumerate(plot.axes):
        if i==0:
            ax.set_ylabel("Data")
            ax.set_title(data.name)
        else:
            ax.set_ylabel("Distance Profile")
            # ## Find peaks in the dp timeseries
            peak_height = np.mean(dp.value)-1*np.std(-dp.value)
            peaks, _ = find_peaks(-dp.value, height=-peak_height)
            ax.plot(data.times[peaks], dp.value[peaks], "x")

    ax = plot.gca()
    if log_yscale:
        ax.set_yscale('log')

    return ax

def plot_matrix_profile(data, event_gps=None, mp=None, motif_len=0, log_yscale=False, dqf=False, titlestr=None):
    '''
    mp[:,0] : matrix profile values
    mp[:,1] : matrix profile indices
    mp[:,2] : right indices
    mp[:,3] : left indices
    '''
    if isinstance(data, list):
        data_len = len(data[0].value)
        times = data[0].times
    else:
        data_len = len(data.value)
        times = data.times

    mp_ = np.concatenate([mp[:,0], np.ones(data_len-len(mp))*[np.nan]])
    mp_ts = TimeSeries(mp_, times=times, name='Matrix Profile')
    
    if isinstance(data, list): plot = Plot(*data, mp_ts, separate=True, sharex=True, title='')
    else: plot = Plot(data, mp_ts, separate=True, sharex=True, title='')
    if dqf: plot.add_segments_bar(dqf)
    # print(len(plot.axes))

    for i, ax in enumerate(plot.axes):
        if isinstance(data, list):
            if i == 0 or i ==1:
                ax.set_ylabel(f"Data {i+1}")
                # ax.set_title(data[i].name)
                if log_yscale: ax.set_yscale('log')
                if event_gps: ax.set_xscale('seconds', epoch=event_gps)
                
                # motif_idx = np.argsort(mp[:, 0])[0] # top-1 motif
                # nearest_neighbor_idx = mp[motif_idx, 1]
                # print(f"Top-1 Motif Index: {motif_idx}, Nearest Neighbor Index: {nearest_neighbor_idx}")
                # ax.plot(data.times.value[motif_idx:motif_idx+motif_len], data.value[motif_idx:motif_idx+motif_len], color='red', label='Motif')
                # ax.plot(data.times.value[nearest_neighbor_idx:nearest_neighbor_idx+motif_len], data.value[nearest_neighbor_idx:nearest_neighbor_idx+motif_len], color='red', label='Nearest Neighbor')

                # colors = matplotlib.cm.tab20(range(10))
                # for k in range(10):
                #     motif_idx = np.argsort(mp[:, 0])[k] # top k-th motif
                #     nearest_neighbor_idx = mp[motif_idx, 1]
                #     print(f"Top-{k+1} Motif Index: {motif_idx}, Nearest Neighbor Index: {nearest_neighbor_idx}")
                #     ax.plot(data.times.value[motif_idx:motif_idx+motif_len], data.value[motif_idx:motif_idx+motif_len], color=colors[k], label='Motif')
                #     ax.plot(data.times.value[nearest_neighbor_idx:nearest_neighbor_idx+motif_len], data.value[nearest_neighbor_idx:nearest_neighbor_idx+motif_len], color=colors[k], label='Nearest Neighbor')
            
            if i == 2:
                ax.set_ylabel("Matrix Profile\n(AB-Join)")
                if event_gps: ax.set_xscale('seconds', epoch=event_gps)
                # if log_yscale: ax.set_yscale('log')
            
        else:
            if i == 0:
                ax.set_ylabel("Data")
                # ax.set_title(data.name)
                if log_yscale: ax.set_yscale('log')
                if event_gps: ax.set_xscale('seconds', epoch=event_gps)
            elif i==1:
                ax.set_ylabel("Matrix Profile\n(Self-Join)")
                if event_gps: ax.set_xscale('seconds', epoch=event_gps)
                # if log_yscale: ax.set_yscale('log')
    
    plot.suptitle(titlestr)

    return plot

def plot_multidim_matrix_profile(data, mdmp):
    '''
    mdmp[0] : multi-dimensional matrix values
    mdmp[1] : multi-dimensional matrix profile index
    '''
    num_ts = mdmp[0].shape[0]
    data_ts_len = len(data[list(data.keys())[0]])
    mdmp_ts_len = mdmp[0].shape[1]
    match_with_zeros_arr = np.ones((num_ts, data_ts_len-mdmp_ts_len))*[np.nan]

    mdmp_0 = np.concatenate((mdmp[0], match_with_zeros_arr), axis=1)

    mdmp_tsdict = TimeSeriesDict()
    # i = 0
    for i, (ts_name, ts_val) in enumerate(data.items()):
        mdmp_tsdict[ts_name] = TimeSeries(list(mdmp_0[i,:]), times=ts_val.times, name='Matrix Profile')   

    plot1 = data.plot()
    # plot1 = Plot(*list(data.values()), separate=True, sharex=True, figsize=(20,50))
    ax1 = plot1.gca()
    
    plot2 = mdmp_tsdict.plot()
    # plot2 = Plot(*list(mdmp_tsdict.values()), separate=True, sharex=True, figsize=(20,50))
    ax2 = plot2.gca()
    
    return ax1, ax2

def zoom_to_regimes(data, cac, regime_location, zoom_window=2*60*60):
    cac = np.concatenate([cac, np.ones(len(data.value)-len(cac))])
    cac = TimeSeries(cac, times=list(data.times), name='Corrected Arc Curve')
    
    plot = Plot(data, cac, separate=True, sharex=True)
    
    ax = plot.gca()
    x_loc = data.times[regime_location].value
    ax.axvline(x=x_loc, color='red', linestyle='--')
    ax.set_xlim(x_loc-(zoom_window//2), x_loc+(zoom_window//2))
    return ax

def plot_pan_matrix_profile(data, pan_mp, log_yscale=False):
    pass



## Computation
def compute_distance_profile(query=None, data=None):
    print("Computing the Distance Profile...\n\t(timeseries length = {}, query length = {})...".format(len(data), len(query)))
    tic = datetime.datetime.now()
    dp = stumpy.mass(query, data)
    print("DONE.")
    print(f"Time taken: {datetime.datetime.now()-tic}")
    return dp

# def compute_matrix_profile(data=None, m=100, T_B=None, approx=False, num_updates=1, use_gpu=False):
#     if approx:
#         print("Computing the Approximate Matrix Profile...\n\t(timeseries length = {}, subsequence length = {})...".format(len(data), m))
#         tic = datetime.datetime.now()
#         mp = stumpy.scrump(data, T_B=T_B, m=m)
#         for i in range(num_updates): mp.update()
#         print("DONE.")
#         print(f"Time taken: {datetime.datetime.now()-tic}")
#         mp = np.array([mp.P_, mp.I_]).T
#     else:
#         print("Computing the Exact Matrix Profile...\n\t(timeseries length = {}, subsequence length = {})...".format(len(data), m))
#         tic = datetime.datetime.now()
#         if use_gpu:
#             mp = stumpy.gpu_stump(data, T_B=T_B, m=m)
#         else:
#             mp = stumpy.stump(data, T_B=T_B, m=m)
#         print("DONE.")
#         print(f"Time taken: {datetime.datetime.now()-tic}")
#     # mp = stumpy.gpu_stump(data, m=m)
#     return mp

def compute_matrix_profile(data=None, m=100, T_B=None, 
                            algo='scrimp', num_updates=1, pre_scrump=False,
                            compute_correlation=False
                          ):
    '''
    data  : Numpy array holding the timeseries
    m     : subsequence/window/motif length
    T_B   : Second timeseries if doing an AB-join
    algo  : 'stump' if doing exact MP
            'stump-gpu' if doing exact MP using GPU
            'scrimp' if doing an approximate MP, also specify num_updates and pre_scrump options.
            'scamp' if doing exact MP with a faster algo.
    '''
    
    if algo=='scamp':
      if isinstance(T_B, np.ndarray):
        print("Computing the Exact Matrix Profile AB-join using SCAMP...\n\t(timeseries length = {}, subsequence length = {})...".format(len(data), m))
        tic = datetime.datetime.now()
        profile, index = pyscamp.abjoin(data, T_B, m, pearson=compute_correlation)
        print("DONE.")
        elapsed_time = datetime.datetime.now()-tic
        print(f"Time taken: {elapsed_time}")
        mp = np.array([profile, index]).T
      else:
        print("Computing the Exact Matrix Profile self-join using SCAMP...\n\t(timeseries length = {}, subsequence length = {})...".format(len(data), m))
        tic = datetime.datetime.now()
        if isinstance(data, np.ndarray): data = list(data)
        profile, index = pyscamp.selfjoin(data, m, pearson=compute_correlation)
        print("DONE.")
        elapsed_time = datetime.datetime.now()-tic
        print(f"Time taken: {elapsed_time}")
        mp = np.array([profile, index]).T

    elif algo=='scrimp':
      print("Computing the Approximate Matrix Profile using SCRIMP...\n\t(timeseries length = {}, subsequence length = {})...".format(len(data), m))
      tic = datetime.datetime.now()
      mp = stumpy.scrump(data, T_B=T_B, m=m, pre_scrump=pre_scrump)
      for i in range(num_updates): mp.update()
      print("DONE.")
      elapsed_time = datetime.datetime.now()-tic
      print(f"Time taken: {elapsed_time}")
      mp = np.array([mp.P_, mp.I_]).T
    elif algo=='stump':
      print("Computing the Exact Matrix Profile using STUMP...\n\t(timeseries length = {}, subsequence length = {})...".format(len(data), m))
      tic = datetime.datetime.now()
      mp = stumpy.stump(data, T_B=T_B, m=m)
      print("DONE.")
      elapsed_time = datetime.datetime.now()-tic
      print(f"Time taken: {elapsed_time}")
    elif algo=='stump-gpu':
      print("Computing the Exact Matrix Profile using STUMP with GPU...\n\t(timeseries length = {}, subsequence length = {})...".format(len(data), m))
      tic = datetime.datetime.now()
      mp = stumpy.gpu_stump(data, T_B=T_B, m=m)
      print("DONE.")
      elapsed_time = datetime.datetime.now()-tic
      print(f"Time taken: {elapsed_time}")
    
    print(f"\tmin(MP)/mean(MP) = {np.min(mp[:,0])/np.mean(mp[:,0])}")
    return mp

def compute_multidim_matrix_profile(data=None, m=100):
    print("Computing the Multi-dimensional Matrix Profile...")
    tic = datetime.datetime.now()
    mdmp = stumpy.mstump(data,m=m)
    print("DONE.")
    print(f"Time taken: {datetime.datetime.now()-tic}")
    return mdmp

def compute_cac(data=None, m=100, L=100, n_regimes=3):
    mp = compute_matrix_profile(data=data, m=m)
    print("Computing the Corrected Arc Curve...")
    tic = datetime.datetime.now()
    cac, regime_locations = stumpy.fluss(mp[:, 1], L=L, n_regimes=n_regimes, excl_factor=1)
    print("DONE.")
    print(f"Time taken: {datetime.datetime.now()-tic}")
    return cac, regime_locations

def compute_pan_matrix_profile(data=None, min_m=60*60, max_m=5*60*60, step=60*60):
    print("Computing the Pan Matrix Profile...")
    tic = datetime.datetime.now()
    pan_mp = stumpy.stimp(data, min_m=min_m, max_m=max_m, step=step)
    print("DONE.")
    print(f"Time taken: {datetime.datetime.now()-tic}")
    return pan_mp