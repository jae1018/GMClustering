#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pkg_resources



def merge_clusters(merge_list, preds, nc):
    
    clusters_to_merge = [ item for sublist in merge_list for item in sublist ]
    clusters_to_keep = np.sort( np.setdiff1d( np.arange(nc), clusters_to_merge ) ).tolist()
    
    num_clusters_after = len(merge_list) + len(clusters_to_keep)
    
    new_preds = np.full(preds.shape[0], -1)
    cluster_counter = 0
    for single_merge_list in merge_list:
        merged_inds = np.hstack( [ np.where(preds == c)[0] for c in single_merge_list ] )
        new_preds[merged_inds] = cluster_counter
        cluster_counter += 1
    
    for c_kept in clusters_to_keep:
        inds = np.where(preds == c_kept)[0]
        new_preds[inds] = cluster_counter
        cluster_counter += 1
        
    return new_preds







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
        #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = plt.get_cmap('brg')( np.linspace(0, 1, num_clust) )
        
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
    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = plt.get_cmap('brg')( np.linspace(0, 1, num_clust) )
    
    # setup consistent binds
    mins, maxs = modify_xy_bounds(df[xy].values)
    bins = [ np.linspace(mins[i], maxs[i], num=50) for i in range(2) ]
    
    
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
        color_dict = {0:0, 1:3, 2:1, 3:2}
        for i in np.unique(preds):
            mask = preds == i
            fg_dat = data[mask][var]
            if var in logx: fg_dat = np.log10( fg_dat )
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][ color_dict[int(i)] ]
            ax.hist( fg_dat, bins=bins, alpha=0.5, color=color )

    for var, ax in zip(hist_vars,axes):
        plot_var(dat, var, preds, ax)

    return fig, axes




def load_example_data():
    stream = pkg_resources.resource_stream(__name__, 'data/mms1_2017.csv')
    return pd.read_csv(stream)
    
