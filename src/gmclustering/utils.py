#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
import pkg_resources
from tqdm import tqdm




def find_crossings_with_times(times,
                              predictions,
                              window_size = None,
                              max_deltaT  = None):
    """
    Finds changes in classification (a "crossing") in a time series. Only
    crossings with half a window length before all belonging to one class and
    half a window length ahead all belonging to another, and with a maximum
    time gap between prediction changes of max_deltaT are kept.
    
    Crossings are returned as part of a dict with tuples as keys, e.g.
        { (1,2) : [0, 50, 100],
          (2,1) : [25, 75] }
    means that crossings going from cluster 1 to 2 were found at row indices
    0, 50, and 100 and crossings going from cluster 2 to 1 were found at
    row indices 25 and 75.

    Parameters
    ----------
    times : Pandas series of Timestamps
        Series of times of measurements
    predictions : 1d numpy integer array
        Array of integer predictions
    window_size : Pandas Timedelta
        Window length used to check predictions
    max_deltaT : Pandas Timedelta
        Max time allowed between two measurements in when considering a
        change in predictoins to be a crossing.

    Returns
    -------
    dict( (2-element int tuples) -> row index of point just before crossing )
    """
    if window_size is None:
        window_size = (times[1:] - times[:-1]).median() * 5
    if max_deltaT is None:
        max_deltaT = (times[1:] - times[:-1]).median() * 3
    
    crossings = []
    cross_dict = {}
    
    # Calculate consecutive differences in predictions
    diff_predictions = np.diff(predictions)
    
    # Find where consecutive differences indicate a crossing
    crossing_indices = np.where(diff_predictions != 0)[0]
    
    # Convert crossing indices to corresponding timestamps
    for idx in tqdm(crossing_indices):
        
        # find window start / end with binary search
        window_start = np.searchsorted(times, times[idx] - window_size/2, side='right')
        window_end = np.searchsorted(times, times[idx+1] + window_size/2, side='right')
        
        # confirm half window length before all belongs to one cluster and
        # half window length ahead belongs to another
        pre_crossing_all_same = np.all( predictions[window_start:idx+1] == predictions[idx] )
        post_crossing_all_same = np.all( predictions[idx+1:window_end+1] == predictions[idx+1] )
        
        # check that adjacent crossing points have time diff <= max time allowed
        no_large_time_skip = (times[idx+1] - times[idx]) <= max_deltaT
        
        if pre_crossing_all_same and post_crossing_all_same and no_large_time_skip:
            
            class_tuple = (predictions[idx], predictions[idx+1])
            
            if class_tuple not in cross_dict:
                cross_dict[class_tuple] = [ idx ]
            else:
                cross_dict[class_tuple].append( idx )
    
    return cross_dict




def remap_array(arr, mapping_dict):
    """
    Maps cluster ints to different ints. Supports re-mapping either as just a
    permutation (e.g. [0,1,2] where 0->4, 1->8, 2->3 ==> [4,8,3]) or a
    reduction in clusters (e.g. [0,1,2,3] where (0,1)->5, 2->7, 3->9 ==> [5,5,7,9]).

    Example
    -------
    The following code will take an array of 4-class predictions (with labels
    0, 1, 2, and 3) and remap them such that 0 and 1 map to 0, 2 maps to 1,
    and 3 maps to 2.
    remap_array(preds_arr, {0:[0,1], 1:[2], 2:[3]})

    Parameters
    ----------
    arr : 1d numpy array of ints
        Int array representing cluster affiliation
    mapping_dict : dict(int->list of ints)
        dict where keys are new ints and values are old ints that should
        be replaced

    Returns
    -------
    new_arr : 1d numpy array of ints
        Int array with new cluster affiliations
    """
    new_arr = np.zeros(arr.shape[0], dtype=np.int32)
    for new_value, old_values in mapping_dict.items():
        for old_value in old_values:
            new_arr[arr == old_value] = new_value
    return new_arr






def compare_data_against_clusters(df,
                                  preds,
                                  sub_df,
                                  logx      = None,
                                  bins      = None, 
                                  hist_vars = None,
                                  clusters  = None,
                                  fig_kws   = None,
                                  hist_kws  = None):
    """
    Compares the distribution of data in sub_df against all data of df
    belonging to the specified clusters. Can be used for comparing
    the data of individual nodes against the full cluster distributions.

    Required Parameters
    -------------------
    df : pandas dataframe
        dataframe used to create histograms
    preds : 1d numpy array of ints
        Array containing the predictions of each point in df
    sub_df : pandas dataframe
        Another dataframe that will be binned against data in df
    
    Optional Parameters
    -------------------
    logx : list of strs or bool (default False)
        Used to determine if log10 of feature should be binned. Can specify
        everything as logx using 'logx = True' or particular feature using
        'logx = [ my_feature ]'
    bins : int (default 50)
        Number of bins for the histogram
    hist_vars : list of strs, optional (default is first column of df)
        Features of df that will be plotted
    clusters : int / list of ints (default 0)
        The clusters over which data should be separately plotted
    fig_kws : kwargs dict (default is empty dict)
        keywords for plt.subplots()
    hist_kws : kwargs dict (default is {'alpha':0.5, 'density':True})
        keywords for plt.hist()

    Returns
    -------
    2-element tuple of created (fig, axes)
    """
    
    # Pick first var as hist_var if none given
    if hist_vars is None: hist_vars = [ list(df)[0] ]
    
    # default to 0th cluster if none given
    if clusters is None: clusters = 0
    if np.issubdtype(type(clusters), np.number): clusters = [ clusters ]
        
    # default fig_kws to empty dict
    if fig_kws is None: fig_kws = {}
    
    # default hist_kws to dict
    if hist_kws is None:
        hist_kws = {'alpha'   : 0.5,
                    'density' : True}
    
    # default bins
    if bins is None: bins = 50
    
    # Default logx to False; make logx list of hist_vars if set to True
    if logx is None: logx = False
    if isinstance(logx, bool):
        if logx:
            logx = hist_vars
        else:
            logx = []
    
    fig, axes = plt.subplots(len(clusters), len(hist_vars), **fig_kws)
    axes_2d = axes.reshape( (len(clusters), len(hist_vars)) )
    
    # Enumerate over clusters first ...
    for ax_x, clust_int in enumerate(clusters):
        cluster_df = df[preds == clust_int]
        
        # ... then each var given for a cluster
        for ax_y, var in enumerate(hist_vars):
            cluster_var_data = cluster_df[var].values
            sub_var_data = sub_df[var].values
            
            if var in logx:
                cluster_var_data = np.log10(cluster_var_data)
                sub_var_data = np.log10(sub_var_data)
            
            bin_max = max( [ cluster_var_data.max(), sub_var_data.max() ] )
            bin_min = min( [ cluster_var_data.min(), sub_var_data.min() ] )
            var_bins = np.linspace(bin_min, bin_max, num=bins)
            
            ax = axes_2d[ax_x,ax_y]
            ax.hist( cluster_var_data, bins=var_bins, **hist_kws)
            ax.hist( sub_var_data, bins=var_bins, **hist_kws)
            
            # set ylabel if first column of axes
            if ax_y == 0:
                ax.set_ylabel(clust_int)
            
            # set xlabel if last row of axes
            if ax_x == len(clusters) - 1:
                ax.set_xlabel(var)
    
    return fig, axes






def modify_xy_bounds(xy, margin=None):
    if margin is None: margin = 0.05
    
    mins = np.min(xy, axis=0)
    maxs = np.max(xy, axis=0)
    
    new_mins = mins - (maxs - mins) * margin
    new_maxs = (maxs - mins) * margin + maxs
    
    return new_mins, new_maxs





def nrows_ncols(num_plots):
    
    nr, nc = None, None
    if num_plots <= 4:
        nr, nc = (2,2)
    elif num_plots <= 6:
        nr, nc = (2,3)
    elif num_plots <= 9:
        nr,nc = (3,3)
    elif num_plots <= 12:
        nr,nc = (3,4)
    elif num_plots <= 16:
        nr, nc = (4,4)
    elif num_plots <= 20:
        nr, nc = (4,5)
    else: # <= 25
        nr, nc = (5,5)
        
    return nr, nc





def plot_xy_preds_multimission(df, preds, fig_kws=None, title=None, xy=None):
    if xy is None: xy = ['X','Y']
    
    from matplotlib.ticker import AutoMinorLocator
    
    if fig_kws is None: fig_kws = {'figsize':(8,8)}
    
    num_clust = len( np.unique(preds) )
    nc = num_clust + 1
    nr = 3
    #nr, nc = nrows_ncols(num_axes)
    
    fig, axes = plt.subplots(nr, nc, **fig_kws)
    
    # setup consistent binds
    mins, maxs = modify_xy_bounds(df[xy].values)
    bins = [ np.linspace(mins[i], maxs[i], num=50) for i in range(2) ]
    
    for axes_row, sc in zip(axes,['all','themis','mms']):
        
        axes1d = axes_row.flatten()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        #colors = plt.get_cmap('brg')( np.linspace(0, 1, num_clust) )
        
        if sc == 'all':
            sc_mask = np.full(df.shape[0], True)
        else:
            sc_mask = get_particular_sc_mission(df, mission=sc)
        
        subdf = df[sc_mask]
        
        
        #axes1d[0].scatter( df['X'], df['Y'], s=2.5, c=preds )
        for i in range(num_clust):
            inds = np.where(preds[sc_mask] == i)[0]
            sns.histplot( subdf.iloc[inds], ax=axes1d[0],
                          x=xy[0], y=xy[1], alpha=0.5, color=colors[i], bins=bins )
        
        for i in range(num_clust):
            inds = np.where(preds[sc_mask] == i)[0]
            subdf_cluster = subdf.iloc[inds]
            #axes1d[i+1].scatter( subdf['X'], subdf['Y'], s=2.5, c=colors[i])
            sns.histplot(subdf_cluster, x=xy[0], y=xy[1],
                         ax=axes1d[i+1], color=colors[i], bins=bins)
            
        # get consistent xy bounds and turn grid on
        #mins, maxs = modify_xy_bounds(subdf[xy].values)
        for i in range(len(axes1d)):
            axes1d[i].set_xlim(mins[0], maxs[0])
            axes1d[i].set_ylim(mins[1], maxs[1])
            axes1d[i].xaxis.set_minor_locator(AutoMinorLocator())
            axes1d[i].yaxis.set_minor_locator(AutoMinorLocator())
            axes1d[i].grid(True, which='major', ls='solid', c='grey', alpha=0.5, lw=3)
            axes1d[i].grid(True, which='minor', ls='solid', c='grey', alpha=0.3, lw=1)
            
        
    if title is not None:
        fig.suptitle(title, fontsize=16)
    #fig.tight_layout()
    
    for ax in axes.flatten():
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    axes[0,0].set_ylabel('THEMIS+MMS')
    axes[1,0].set_ylabel('THEMIS')
    axes[2,0].set_ylabel('MMS')
    
    axes[0,0].set_title('C-ALL')
    axes[0,1].set_title('C0')
    axes[0,2].set_title('C1')
    axes[0,3].set_title('C2')
    axes[0,4].set_title('C3')
    
    #plt.show()
    
    return fig, axes








def plot_xy_preds(df, preds, fig_kws=None, title=None, xy=None):
    if xy is None: xy = ['X','Y']
    
    from matplotlib.ticker import AutoMinorLocator
    
    if fig_kws is None: fig_kws = {'figsize':(8,8)}
    
    num_clust = len( np.unique(preds) )
    num_axes = num_clust + 1
    nr, nc = nrows_ncols(num_axes)
    
    fig, axes = plt.subplots(nr, nc, **fig_kws)
    axes1d = axes.flatten()
    
    # setup consistent binds
    mins, maxs = modify_xy_bounds(df[xy].values)
    bins = [ np.linspace(mins[i], maxs[i], num=50) for i in range(2) ]
    
    #colors = color_per_cluster(np.unique(preds))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    #axes1d[0].scatter( df['X'], df['Y'], s=2.5, c=preds )
    for i in range(num_clust):
        inds = np.where(preds == i)[0]
        sns.histplot( df.iloc[inds], ax=axes1d[0],
                      x=xy[0], y=xy[1], alpha=0.5, color=colors[i], bins=bins )
    
    for i in range(num_clust):
        inds = np.where(preds == i)[0]
        subdf = df.iloc[inds]
        #axes1d[i+1].scatter( subdf['X'], subdf['Y'], s=2.5, c=colors[i])
        sns.histplot(subdf, x=xy[0], y=xy[1], ax=axes1d[i+1], color=colors[i], bins=bins)
        
    # get consistent xy bounds and turn grid on
    mins, maxs = modify_xy_bounds(df[['X','Y']].values)
    for i in range(len(axes1d)):
        axes1d[i].set_xlim(mins[0], maxs[0])
        axes1d[i].set_ylim(mins[1], maxs[1])
        axes1d[i].xaxis.set_minor_locator(AutoMinorLocator())
        axes1d[i].yaxis.set_minor_locator(AutoMinorLocator())
        axes1d[i].grid(True, which='major', ls='solid', c='grey', alpha=0.5, lw=3)
        axes1d[i].grid(True, which='minor', ls='solid', c='grey', alpha=0.3, lw=1)
        
    for i in range(len(axes1d)):
        if i > num_clust:
            axes1d[i].axis('off')
    
    if title is not None:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    
    plt.show()
    
    return fig, axes





def color_per_cluster(cluster_ints, cmap=None):
    """
    Return dict of cluster ints -> colors to be used with each cluster

    Parameters
    ----------
    cluster_ints : container of *unique* cluster ints

    Returns
    -------
    dict[int:cluster color (str)]
    """
    if cmap is None: cmap ='brg'
    
    n = len(cluster_ints)
    if -1 in cluster_ints:
        colors = ['black'] + plt.get_cmap(cmap)(np.linspace(0, 1, n - 1)).tolist()
    else:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, n)).tolist()
    
    colors_dict = {}
    for c_int, c_int_color in zip(np.sort(cluster_ints), colors):
        colors_dict[c_int] = c_int_color
    
    return colors_dict




def time_series(dat, preds, trange=None, sc=None,
                fig_kws=None, node_dict=None,
                identify_nodes=None, show_posit=None):
    if node_dict is None: node_dict = {}
    if identify_nodes is None: identify_nodes = []
    if show_posit is None: show_posit = False
    
    if sc is None: sc = 'tha'
    sc_mask = dat['spacecraft'] == sc
    
    if trange is None:
        start = pd.to_datetime( dat[sc_mask]['time'].iloc[0] )
        trange = [ start, start + pd.Timedelta('24h') ]
    else:
        trange = [ pd.to_datetime(elem) for elem in trange ]
    trange_mask = ( pd.to_datetime( dat[sc_mask]['time'] ) >= trange[0] ) \
                  & ( pd.to_datetime( dat[sc_mask]['time'] ) <= trange[1] )
    
    if fig_kws is None:
        fig_kws = {'figsize':(8,10)}
    # Create figure and subplots
    fig = plt.figure(**fig_kws)

    # Define grid for subplots
    grid_shape = (5, 4) if show_posit else (5, 2)
    ax1 = plt.subplot2grid(grid_shape, (0, 0), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid(grid_shape, (1, 0), rowspan=1, colspan=2)
    ax3 = plt.subplot2grid(grid_shape, (2, 0), rowspan=1, colspan=2)
    ax4 = plt.subplot2grid(grid_shape, (3, 0), rowspan=1, colspan=2)
    ax5 = plt.subplot2grid(grid_shape, (4, 0), rowspan=1, colspan=2)

    # Scatter plots
    if show_posit:
        ax6 = plt.subplot2grid(grid_shape, (0, 2), rowspan=2, colspan=2)
        ax7 = plt.subplot2grid(grid_shape, (2, 2), rowspan=2, colspan=2)
    
    subdat = dat[sc_mask][trange_mask]
    subpreds = preds[sc_mask][trange_mask]
    times = pd.to_datetime( subdat['time'] )
    
    
    # sort based on times just in case data is shuffled
    sort_inds = np.argsort(times)
    times = times.iloc[sort_inds]
    subdat = subdat.iloc[sort_inds]
    subpreds = subpreds[sort_inds]
    
    ax1.plot( times, subdat['BX'], c='red', lw=1.5, label='BX' )
    ax1.plot( times, subdat['BY'], c='green', lw=1.5, label='BY' )
    ax1.plot( times, subdat['BZ'], c='blue', lw=1.5, label='BZ' )
    ax1.set_ylabel('B (nT)')
    ax1.legend(framealpha=0.5)
    
    ax2.plot( times, subdat['VX'], c='red', lw=1.5, label='VX' )
    ax2.plot( times, subdat['VY'], c='green', lw=1.5, label='VY' )
    ax2.plot( times, subdat['VZ'], c='blue', lw=1.5, label='VZ' )
    ax2.set_ylabel('V (km/s)')
    ax2.legend(framealpha=0.5)
    
    ax3.plot( times, np.log10(subdat['T']) )
    ax3.set_ylabel('T (eV)')
    
    ax4.plot( times, np.log10(subdat['n']) )
    ax4.set_ylabel('n (#/cc)')
    
    # Create horizontal color bars to better show classifications
    # time series
    pred_types = np.unique(preds).tolist()
    if -1 in pred_types: pred_types.remove(-1)
    pred_type_colors = [ *plt.rcParams['axes.prop_cycle'].by_key()['color'] ]
    pred_type_colors = pred_type_colors[:len(pred_types)]
    for val in np.arange(len(pred_types)):
        color = pred_type_colors[val]
        axis_val = val / ( len(pred_types)-1 )
        axvspan_shift = 0.1 / ( len(pred_types)-1 )
        if val == 0:
            ymin = axis_val
            ymax = axis_val+1.75*axvspan_shift
        elif val == max(pred_types):
            ymin = axis_val-1.75*axvspan_shift
            ymax = axis_val
        else:
            ymin = axis_val-axvspan_shift
            ymax = axis_val+axvspan_shift
        ax5.axvspan(times.iloc[0],
                        times.iloc[-1],
                        ymin = ymin,
                        ymax = ymax,
                        facecolor = color)
        
        # also scatter plot here to get colors?
        if show_posit:
            subpreds_mask = subpreds == val
            ax6.scatter( subdat['X'][subpreds_mask],
                         subdat['Y'][subpreds_mask],
                         s=2, c=color )
            ax7.scatter( subdat['X'][subpreds_mask],
                         subdat['Z'][subpreds_mask],
                         s=2, c=color )
    
    # plot actual time series of classifications as black line
    ax5.plot( times, subpreds, c='black', lw=1.5 )
    ax5.set_ylabel('class')
    ax5.set_ylim( min(pred_types)-0.1, max(pred_types)+0.1 )


    # Identify nodes in time series by drawing vertical lines at point
    # belonging to given nodes
    if len(identify_nodes) > 0:
        
        # check that node_dict was given
        if node_dict is None:
            raise ValueError('Must supply node_dict to identify nodes in'
                             + ' time series (see function GMC.som_activations)')
        
        # identify points belonging to node that are within time frame ...
        for node in identify_nodes:
            inds = node_dict[node]
            
            node_mask = np.isin(inds, subdat.index.values)            
            node_times = times.loc[ inds[node_mask] ]
            
            # ... then plot them on each axis
            if node_mask.any():
                for ax in [ax1, ax2, ax3, ax4, ax5]:
                    for single_time in node_times:
                        ax.axvline( single_time, lw=1, alpha=0.3 )
    
    ## determine date formatter
    # hide time labels of all time series besides bottom plot
    for ax in [ax1,ax2,ax3,ax4]:
        ax.set_xticklabels([])
    # if covers, multiple days then include day ...
    if times.iloc[0].day != times.iloc[-1].day:
        date_fmt = '%d %H:%M'
    else:
        date_fmt = '%H:%M'
    # format dates
    date_formatter = mdates.DateFormatter(date_fmt)
    ax5.xaxis.set_major_formatter(date_formatter)
    # rotate time labels
    ax5.tick_params(axis='x', rotation=45)
    
    # Extra work if having to show position plots
    if show_posit:
        
        # hide x axis labels of x-y plot
        ax6.set_xticklabels([])
        
        # show start and end of scatter plot
        ax6.text( subdat['X'].iloc[0], subdat['Y'].iloc[0], '1')
        ax6.text( subdat['X'].iloc[-1], subdat['Y'].iloc[-1], '2')
        ax7.text( subdat['X'].iloc[0], subdat['Z'].iloc[0], '1')
        ax7.text( subdat['X'].iloc[-1], subdat['Z'].iloc[-1], '2')
    
        # also show markerx on x-y and y-z plots corresponding to time markers
        ## Now THIS was tricky... matplotlib saves the tick info as the number
        ## of *DAYS* since 1970.. so for each tick, need to ...
        for tick in ax5.get_xticks():
            # .. convert it to datetime object ..
            tick_time = pd.to_datetime(tick, unit='d')
            # .. find closest point in data ..
            idx = np.argmin( np.abs(times - tick_time) )
            # .. and plot marker for x-y ..
            ax6.scatter( subdat['X'].iloc[idx],
                         subdat['Y'].iloc[idx],
                         facecolors='none',
                         marker='o',
                         edgecolors='black')
            # .. and y-z
            ax7.scatter( subdat['X'].iloc[idx],
                         subdat['Z'].iloc[idx],
                         facecolors='none',
                         marker='o',
                         edgecolors='black')

    axes = [ax1, ax2, ax3, ax4, ax5]
    if show_posit:
        axes.extend( [ax6, ax7] )
    
    return subdat, (fig, axes)




def get_particular_sc_mission(dat, mission=None):
    if mission is None:
        mission = 'themis'
        
    sc = []
    if mission == 'themis':
        sc = ['tha','thb','thc','thd','the']
    elif mission == 'mms':
        sc = ['mms1','mms2','mms3','mms4']
    else:
        raise ValueError('Unrecognized mission: '+mission)
        
    return np.isin(dat['spacecraft'],sc)






def hist1d_vars(dat, preds, logx=None, hist_vars=None, fig_kws=None):
    if logx is None: logx = []
    if hist_vars is None: hist_vars = 'VX'
    if isinstance(hist_vars,str): hist_vars = [ hist_vars ]
    if isinstance(logx,bool) and logx: logx = hist_vars
    if fig_kws is None: fig_kws = {}
    
    fig, axes = plt.subplots(len(hist_vars), 1, **fig_kws)

    def plot_var(data, var, preds, ax):
        
        bg_dat = data[var]
        if var in logx: bg_dat = np.log10( bg_dat )
        bins = np.linspace( np.min(bg_dat), np.max(bg_dat), 100 )
        ax.hist( bg_dat, bins=bins, alpha=0.5, histtype='step', color='black' )
        ax.set_title(var, fontsize=16)
        #color_dict = {0:0, 1:3, 2:1, 3:2}
        for i in np.unique(preds):
            mask = preds == i
            fg_dat = data[mask][var]
            if var in logx: fg_dat = np.log10( fg_dat )
            #color = plt.rcParams['axes.prop_cycle'].by_key()['color'][ color_dict[int(i)] ]
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][ int(i) ]
            ax.hist( fg_dat, bins=bins, alpha=0.5, color=color )

    for var, ax in zip(hist_vars,axes):
        plot_var(dat, var, preds, ax)

    return fig, axes




def load_example_data():
    stream = pkg_resources.resource_stream(__name__, 'data/mms1_2017.csv')
    return pd.read_csv(stream)
    
