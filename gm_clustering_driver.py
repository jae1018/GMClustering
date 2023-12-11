#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gmclustering import GMClustering, utils
import matplotlib.pyplot as plt
import numpy as np










if __name__ == "__main__":
    
    # get data
    dat = utils.load_example_data()
    #dat = pd.read_csv('/home/jedmond/Documents/ML_Research/Source/gmclustering/GMClustering/src/gmclustering/data/mms1_2017.csv')
    
    # prepare clustering model
    gmc = GMClustering()
    
    # the SOM nodes are static and do not require data to
    # analyze in-of-themselves... so we can go ahead and inspect
    # the clustering solution on the SOM nodes
    gmc.prepare_aggclust(dist=1.65)
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    gmc.plot_som_clust(ax=axes[0])
    dendrogram_kws = {'truncate_mode':'level', 'p':5}
    gmc.plot_dendrogram(ax             = axes[1],
                        dendrogram_kws = dendrogram_kws)
    
    # make predictions for mms sample data
    #  .. can process in batches if enough RAM isn't available all at once
    preds = gmc.predict(dat)
    
    # visualize predictions with a histogram over the clusters
    utils.plot_xy_preds(dat, preds,
                        xy = ['X','Y'],
                        fig_kws = {'figsize' : (10,8)})
    
    # visualize clusters as subsets of 1d histograms
    fig, axes = utils.hist1d_vars(dat, preds,
                                  hist_vars = ['n','T','VY','VX'],
                                  logx      = ['n','T'],
                                  fig_kws = {'figsize':(8,10)})
    fig.tight_layout()
    
    
    
    
    
    
    # ---------- Subcluster Analysis ----------
    
    # Subclusters of the original hierarchical clustering solution can
    # be analyzed by invoking prepare_aggclust and specifying
    # the original distance used and cluster to analyze as a tuple
    # Analyzing the magsphere cluster (cluster 0) as an example
    sub_cluster = 0
    original_distance = GMClustering.default_aggclust_dist
    subset = (original_distance, sub_cluster)
    gmc.prepare_aggclust(subset=subset)
    
    # Plot the dendrogram for the subclustering solution
    # to inform where to pick the new threshold distance
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    gmc.plot_som_clust(ax=axes[0])
    gmc.plot_dendrogram(ax             = axes[1],
                   dendrogram_kws = dendrogram_kws)
    
    # 1.2 seems reasonable
    sub_dist = 1.2
    gmc.prepare_aggclust(subset=subset, dist=sub_dist)
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    gmc.plot_som_clust(ax=axes[0])
    gmc.plot_dendrogram(ax             = axes[1],
                        dendrogram_kws = dendrogram_kws)
    
    # Get predictions for data based on new clustering solution
    subpreds = gmc.predict(dat)
    
    # filter out data that was assigned -1
    subdat = dat[subpreds != -1]
    subpreds = subpreds[subpreds != -1]
    
    # get counts of predictions
    print( np.unique(subpreds, return_counts=True) )
    
    # Show predictions for subcluster
    utils.plot_xy_preds(subdat, subpreds,
                        xy = ['X','Y'])
    fig, axes = utils.hist1d_vars(subdat, subpreds,
                                  hist_vars = ['n','T','VY','VX'],
                                  logx      = ['n','T'],
                                  fig_kws = {'figsize':(8,10)})
    fig.tight_layout()
    