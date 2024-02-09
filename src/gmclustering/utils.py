#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pkg_resources



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




def time_series(dat, preds, trange=None, sc=None, fig_kws=None):
    if sc is None: sc = 'tha'    
    sc_mask = dat['spacecraft'] == sc
    
    if fig_kws is None: fig_kws = {'figsize':(8,8)}
    
    if trange is None:
        start = pd.to_datetime( dat[sc_mask]['time'].iloc[0] )
        trange = [ start, start + pd.Timedelta('24h') ]
    else:
        trange = [ pd.to_datetime(elem) for elem in trange ]
    trange_mask = ( pd.to_datetime( dat[sc_mask]['time'] ) >= trange[0] ) \
                  & ( pd.to_datetime( dat[sc_mask]['time'] ) <= trange[1] )
    
    #fig, axes = plt.subplots(6, 2, figsize=(8,10))
    fig = plt.figure(**fig_kws)
    grid_size = (5,1)
    
    subdat = dat[sc_mask][trange_mask]
    subpreds = preds[sc_mask][trange_mask]
    times = pd.to_datetime( subdat['time'] )
    
    
    # sort based on times just in case data is shuffled
    sort_inds = np.argsort(times)
    times = times.iloc[sort_inds]
    subdat = subdat.iloc[sort_inds]
    subpreds = subpreds[sort_inds]
    
    
    #ax0 = plt.subplot2grid(grid_size, (0,0), colspan=1)
    #ax0.set_title(sc + f' ({times.shape[0]} pts)')
    #ax0.plot( times, subdat['BX'], c='red', lw=1.5, label='BX' )
    #ax0.plot( times, subdat['BY'], c='green', lw=1.5, label='BY' )
    #ax0.plot( times, subdat['BZ'], c='blue', lw=1.5, label='BZ' )
    #ax0.legend()
    
    ax0 = plt.subplot2grid(grid_size, (0,0), colspan=1)
    #ax0.set_title(sc + f' ({times.shape[0]} pts)')
    ax0.plot( times, subdat['VX'], c='red', lw=1.5, label='VX' )
    ax0.plot( times, subdat['VY'], c='green', lw=1.5, label='VY' )
    ax0.plot( times, subdat['VZ'], c='blue', lw=1.5, label='VZ' )
    ax0.legend()
    
    ax1 = plt.subplot2grid(grid_size, (1,0), colspan=1)
    ax1.plot( times, np.log10(subdat['T']) )
    ax1.set_ylabel('T')
    
    ax2 = plt.subplot2grid(grid_size, (2,0), colspan=1)
    ax2.plot( times, np.log10(subdat['n']) )
    ax2.set_ylabel('n')
    
    ax3 = plt.subplot2grid(grid_size, (3,0), colspan=1)
    ax3.plot( times, np.log10(subdat['MA']) )
    ax3.set_ylabel('MA')
    
    ax4 = plt.subplot2grid(grid_size, (4,0), colspan=1)
    ax4.plot( times, subpreds )
    ax4.set_ylabel('prediction')
    
    """
    ax4 = plt.subplot2grid(grid_size, (0,1), rowspan=2)
    ax4.scatter( subdat['X'], subdat['Y'], s=50, c=np.arange(times.shape[0]),
                 cmap='viridis')#, edgecolors='black' )
    ax4.set_xlim(-25,25)
    ax4.set_ylim(-25,25)
    ax4_xticks = ax4.get_xticks()
    ax4_xlabels = ax4.get_xticklabels()
    ax4.set_title('X-Y (purple-to-yellow)')
    
    
    ax5 = plt.subplot2grid(grid_size, (2,1), rowspan=2)
    ax5.scatter( subdat['X'], subdat['Z'], s=50, c=np.arange(times.shape[0]),
                 cmap='viridis' )
    ax5.set_title('X-Z')
    ax5.set_xlim( *ax4.get_xlim() )
    ax5.set_ylim( -10, 10 )
    """
    
    fig.autofmt_xdate()
    fig.tight_layout()
    
    #ax5.set_xticks(ax4_xticks, [elem for elem in ax4_xticks])#ax6_xlabels)
    
    return fig




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
    
