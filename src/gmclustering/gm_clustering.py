#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic packages
import pickle
import os
import pkg_resources

# computational packages
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

# visualization packages
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors







class GMClustering:
    
    #yj_vars = [ 'BX', 'BY', 'BZ',
    #            'VX', 'VY', 'VZ' ]
    init_vector_vars = [ 'BX', 'BY', 'BZ',
                         'VX', 'VY', 'VZ', ]
    init_scalar_vars = [ 'n', 'T' ]
    #derived_vars = [ 'p', 'beta', 'MA' ]
    derived_vars = ['mom_X', 'mom_Y', 'mom_Z', 'B', 'V']
    init_vars = [ *init_vector_vars, *init_scalar_vars ]
    log_vars = [ *init_scalar_vars, 'mom_X', 'mom_Y', 'mom_Z' ]
    
    default_aggclust_kws = {'linkage'    : 'ward',
                            'n_clusters' : None}
    
    default_aggclust_dist = 1.65
    
    def __init__(self):
        
        stream = pkg_resources.resource_filename(__name__, 'models')
        self.model_folder = stream
        self._load_models()
        self.node_mask = np.full( self.som_weights_2d().shape[0], True )
        self.current_dist = GMClustering.default_aggclust_dist
        self.prepare_aggclust()
        
        
    
    
    
    def _load_model(self, filename):
        file = open( os.path.join(self.model_folder,filename), 'rb' )
        return pickle.load(file)
    
    
    
    
    
    def _load_models(self):
        """
        Load all model components needed for overall model
        """
        #self.yj_trans = self._load_model('yj_trans.pkl')
        self.init_scaler = self._load_model('init_scaler.pkl')
        self.pca = self._load_model('pca.pkl')
        self.post_pca_scaler = self._load_model('postpca_scaler.pkl')
        self.som = self._load_model('som.pkl')
        
        
        
        
        
    def _calculate_derived_params(df):
        
        """
        Calculates the additional variables needed for clustering
        from the original variables
        
        Input
        -----
        df: pandas dataframe with original variables
        
        Returns
        -------
        dataframe deep-copy with including derived variables
        """
        
        # make deep copy
        new_df = df.copy(deep=True)
        
        # values from df
        density_numPcc = new_df['n'].values
        b_field_magnitude_nT = np.sqrt( (new_df[['BX','BY','BZ']]**2).sum(axis=1) )
        new_df['B'] = b_field_magnitude_nT
        speed_kmPs = np.sqrt( (new_df[['VX','VY','VZ']]**2).sum(axis=1) )
        new_df['V'] = speed_kmPs
        
        # calculate momentum density
        new_df['mom_X'] = density_numPcc * np.abs( new_df['VX'].values )
        new_df['mom_Y'] = density_numPcc * np.abs( new_df['VY'].values )
        new_df['mom_Z'] = density_numPcc * np.abs( new_df['VZ'].values )
        
        return new_df



    
    
    def _prepare_data_for_pca(self, df):
        """
        Prepares df for pca transform by
        -- log10 scaling scalar variables
        -- yj-transforming the log10 of square of vector components
        -- MinMax scaling the results

        Parameters
        ----------
        df : Pandas dataframe containing derived vars

        Returns
        -------
        dataframe to be fed to pca
        """
        scaled_df = df.copy(deep=True)
        # log10 scaling of scalar data
        log_vars = GMClustering.log_vars
        scaled_df[log_vars] = np.log10( scaled_df[log_vars] )
        # ensure proper order of columns
        orig_var_order = self.init_scaler.feature_names_in_
        return pd.DataFrame(
                    self.init_scaler.transform( scaled_df[orig_var_order] ),
                    columns = list(orig_var_order)
                            )
        
        
    
    
    def _som_cluster_mapper(self, data_for_som,
                                  cmodel_preds):
        """
        Propagate cluster predictions of SOM nodes to data

        Parameters
        ----------
        data_for_som : numpy array
            array of data already prepared for SOM
        cmodel_preds : 1d numpy array
            cluster predictions of SOM nodes

         Returns
         -------
         1d numpy array of cluster predictions of data
         """
        
        # Get dict of som nodes to data indices
        som_preds = self.som.labels_map(data_for_som,
                                        np.arange(data_for_som.shape[0]))
        newdict = {}
        for key in som_preds:
            newdict[key] = list( som_preds[key].keys() )
           
        # setup final preds array
        if len(cmodel_preds.shape) == 1:
            final_preds = np.zeros( data_for_som.shape[0] )
        else:
            final_preds = np.zeros( (data_for_som.shape[0], cmodel_preds.shape[1]) )
       
        # assign preds from node to point  
        for node_tuple in newdict:
            flat_ind = np.ravel_multi_index( node_tuple, self.som._weights.shape[:2] )
            final_preds[ newdict[node_tuple] ] = cmodel_preds[flat_ind]
           
        return final_preds
    
    
    
    
    
    def _apply_pca(self, df_for_pca):
        """
        Applies PCA (and post-PCA rescaling) to data
        
        Return
        ------
        numpy array
        """
        # make sure to transform feature in same order as original fitting
        orig_var_order = self.pca.feature_names_in_
        # rescale and pca-transform data
        return self.post_pca_scaler.transform(
                            self.pca.transform( df_for_pca[orig_var_order] )
                                              )
        
    
    
    
    
    def som_weights_2d(self):
        """
        Get weights of SOM as 2d matrix where rows are nodes and columns
        are their features

        Returns
        -------
        2d numpy array
        """
        num_neurons = np.prod( self.som._weights.shape[:2] )
        shape_2d = (num_neurons, self.som._weights.shape[2] )
        return self.som._weights.reshape(shape_2d)
    
    
    
    
    
    def _aggclust_predictions_on_som_nodes(self, dist = None):
        """
        Get hierarchical clustering predictions of SOM nodes

        Parameters
        ----------
        dist : float, optional
            Distance used for hierarchical clustering partitioning
            If None, default dist is used 

        Returns
        -------
        1d numpy array of integers representing cluster predictions
        (nodes that are masked out are assigned a cluster int of -1)
        """
        if dist is None: dist = self.aggclust.distance_threshold
        
        aggclust_kws = { **GMClustering.default_aggclust_kws,
                         **{'distance_threshold' : dist} }
        aggclust = AgglomerativeClustering( **aggclust_kws )
        
        # Get prediction over subset (as determined by node mask)
        weights_2d = self.som_weights_2d()
        sub_preds = aggclust.fit_predict( weights_2d[self.node_mask] )
        
        # Predict non-relevant nodes as -1
        full_preds = np.full(weights_2d.shape[0], -1)
        full_preds[self.node_mask] = sub_preds
        
        return full_preds
    
    
    
    
    
    def plot_dendrogram(self, dendrogram_kws = None,
                              fig_kws        = None,
                              ax             = None):
        """
        Create a dendrogram showing the merge-order of the hierarhical clusters
        of SOM nodes. If any nodes have been masked out using prepare_aggclust,
        they will not be factored into the dendrogram.

        Parameters
        ----------
        dendrogram_kws : dict of keywords, optional
            keywords for scipy.cluster.hierarchy.dendrogram function
        fig_kws : dict of keywords, optional
            keywords used when creating figure object containing plot
        ax : matplotlib axis instance
            If None, figure and axis will be created

        Returns
        -------
        (fig, ax) tuple
        fig and ax are figure and axis instance used to create plot
        """
        
        if dendrogram_kws is None: dendrogram_kws = {}
        default_dendrogram_kws = {'color_threshold':0}
        dendrogram_kws = { **default_dendrogram_kws, **dendrogram_kws }
        
        if fig_kws is None: fig_kws = {'figsize':(6,6)}
        
        # Create figure if axis is None
        if ax is None:
            fig, ax = plt.subplots(1,1,**fig_kws)
        else:
            fig = plt.gcf()
        
        # Get linkage matrix of masked-in SOM nodes
        linkage_matrix = linkage(
                    self.som_weights_2d()[self.node_mask],
                    method = GMClustering.default_aggclust_kws['linkage']
                                )
        max_dist = self.aggclust.distances_.max()
        
        # Show dendrogram with grid lines
        dendrogram(linkage_matrix, ax=ax, **dendrogram_kws)
        for vert_val in np.arange(0.5, int(max_dist)+0.5, step=0.5):
            ax.axhline(vert_val, ls='solid', c='grey', alpha=0.5)
        ax.xaxis.set_ticklabels([])
        
        # Show distance threshold used for clustering solution
        dist = self.aggclust.distance_threshold
        ax.axhline(dist, ls='dashed', c='black')
        
        return fig, ax
    
    
    
    
    
    def _prepare_data_for_som(self, data):
        """
        Prepares data for feeding into trained self-organizing map

        Parameters
        ----------
        data : pandas dataframe containing the original variables

        Raises
        ------
        ValueError: If data does not contain the needed variables

        Returns
        -------
        postpca_data : numpy array
            Array of PCA-projected data
        """
        
        # Check that data contains necessary vars
        num_init_vars = np.sum( np.isin(list(data), GMClustering.init_vars) )
        if num_init_vars != len(GMClustering.init_vars):
            raise ValueError('data does not contain all of the necessary '
                             + 'vars: ' + str(GMClustering.init_vars))
        
        # calculate derived params
        init_vars = GMClustering.init_vars
        data_with_derived_vars = GMClustering._calculate_derived_params(data[init_vars])
        
        # prepare data for pca by applying transforms and rescalings
        scaled_data = self._prepare_data_for_pca(data_with_derived_vars)
        
        # apply pca and rescale post-pca results
        postpca_data = self._apply_pca(scaled_data)
        
        return postpca_data
        
        
    
    
    
    def plot_som_hits(self, data,
                            ax         = None,
                            pcolor_kws = None):
        """
        Plot the number of hits per node as a greyscale matrix

        Parameters
        ----------
        data : dataframe to compute hits for
        ax : matplotlib axis instance
            If None, figure and axis will be created
        pcolor_kws : Dict of keywords for matplotlib.pcolor

        Returns
        -------
        (fig, ax) tuple
        fig and ax are figure and axis instance used to create plot
        """
        
        if pcolor_kws is None: pcolor_kws = {}
        default_pcolor_kws = {'cmap':'gray'}
        pcolor_kws = { **default_pcolor_kws, **pcolor_kws }
        
        som_dat = self._prepare_data_for_som(data)
        
        ind_map = self.som.labels_map(som_dat, np.arange(som_dat.shape[0]))
        # massage into standard dict form, avoiding python Counter objects
        for key in ind_map:
            ind_map[key] = np.array(list( ind_map[key].elements() )).astype( np.int64 )
            
        som_shape_ = self.som_shape()
        hitarr = np.zeros(som_shape_).astype( np.int64 )
        for key in ind_map:
            hitarr[key] = len(ind_map[key])
        
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = plt.gcf()
            
        ref = ax.pcolor(hitarr.T, **pcolor_kws)
        fig.colorbar(ref, ax=ax)
        
        return fig, ax
    
    
    
    
    
    def som_shape(self):
        """
        Returns 2d shape of SOM as 2-element integer tuple
        """
        return ( self.som._neigx.max()+1, self.som._neigy.max()+1 )





    def plot_som_clust(self, dist = None,
                             ax   = None):
        """
        Create grid of SOM nodes colored according to clustering solution

        Parameters
        ----------
        dist : float, optional
            Distance used for hierarchical clustering partitioning
            If None, default dist is used 
        ax : matplotlib axis instance, optional
            If None, figure and axis will be created

        Returns
        -------
        (fig, ax) tuple
        fig and ax are figure and axis instance used to create plot
        """
        
        cmodel_preds = \
            self._aggclust_predictions_on_som_nodes(dist = dist)
        
        # Setup fig if not given
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = plt.gcf()
    
        # establish consistent colors
        colors = [ *plt.rcParams['axes.prop_cycle'].by_key()['color'] ]
        if -1 in np.unique(cmodel_preds):
            colors.insert( 0, 'black')
        colors = colors[:len(np.unique(cmodel_preds))]
        cmap = mcolors.ListedColormap(colors)
        
        som_shape = self.som_shape()
        ref = ax.imshow( cmodel_preds.reshape(som_shape).T,
                         cmap   = cmap,
                         origin = 'lower')
        
        for i in range(som_shape[0]):
            for q in range(som_shape[1]):
                ax.axvline(i+0.5, color='black', lw=0.01)
                ax.axhline(i+0.5, color='black', lw=0.01)
        
        fig.colorbar(ref, ax=ax, ticks=np.unique(cmodel_preds), shrink=0.75)
        
        return fig, ax
        
        
     
        
     
    def prepare_aggclust(self, dist   = None,
                               subset = None):
        """
        Prepare the hierarchical clustering model for clustering SOM nodes
        
        To get sub-clustering solutions, use subset keyword. Data that does
        not belong to the sub-clustering of interest will be masked out
        in future calculations and plots.
        
        Sub-Clustering Example
        ----------------------
        If interested in analyzing, for example, the 3rd cluster (where
        cluster counting starts from 0) after a truncation distance of 2.5
        was used, then subset would be:
          subset = (2.5, 3), or subset = [ (2.5, 3) ]
        If you wanted to analysize further subgroups beyond that, say the
        2nd cluster after a distance of 1, then the full subset keyword 
        would be:
          subset = [ (2.5, 3), (1, 2) ]

        Parameters
        ----------
        dist : float, optional
            Distance used to partition nodes of SOM
        subset : tuple or list of tuples, optional
            Used to analyze sub-clusters of the resulting clustering

        Returns
        -------
        None.
        """
        
        if dist is None: dist = GMClustering.default_aggclust_dist
        if subset is None: subset = []
        if isinstance(subset,tuple): subset = [ subset ]
        
        subset_tuples = [ elem for elem in subset ]
        subset_tuples.append( (dist,None) )
        
        nodes2d = self.som_weights_2d()
        
        inds = np.arange(nodes2d.shape[0])
        for i in range(len(subset_tuples)):
            dist, cluster = subset_tuples[i]
            
            sub_aggclust_kws = { **GMClustering.default_aggclust_kws,
                                 **{'distance_threshold':dist} }
            aggclust = AgglomerativeClustering( **sub_aggclust_kws )
            aggclust.fit( nodes2d[inds] )
            self.aggclust = aggclust
            preds = aggclust.fit_predict( nodes2d[inds] )
            
            if cluster is not None:
                sub_inds = np.where( preds == cluster )[0]
                inds = inds[sub_inds]
        
        self.node_mask = np.isin(np.arange(nodes2d.shape[0]), inds)
    
    
    
    
    
    def predict(self, data,
                      dist = None):
        """
        Make predictions of data. The hierarchical clustering predictions of
        SOM nodes are propagated to the data that the nodes represent.
        
        If prepare_aggclust was used to investigate sub-clusters, data
        not belonging to the subgroup of interest will be assigned
        a cluster value of -1.

        Parameters
        ----------
        data : pandas dataframe containing necessary vars for clustering
        dist : float, optional
            distance threshold used to partition SOM nodes.
            If None, default value is used

        Returns
        -------
        1d numpy array of integers representing cluster predictions
        """
        # Transform data into form usable for SOM
        scaled_embeds = self._prepare_data_for_som( data )
        # get AggClust predictions of SOM nodes 
        cmodel_preds = self._aggclust_predictions_on_som_nodes(dist = dist)
        # Propagate clustering som predictions to data
        return self._som_cluster_mapper(scaled_embeds, cmodel_preds).astype(np.int32)
    
    
    
    
    
    def predict_magsphere_magsheath_solarwind(self, data):
        """
        Make predictions for magnetosheath, magnetosphere, and solar wind.
        This 3-class model accuracy is about 99.4% on the olshevsky
        labeled dataset.

        Parameters
        ----------
        data : pandas dataframe containing necessary vars for clustering

        Returns
        -------
        2-element tuple where
          0: 1d numpy array of integers representing cluster predictions
          1: dict(int->str) indicating cluster affiliation
        """
        
        # Reset model
        self.prepare_aggclust()
        # Predictions from default model correspond to 3-class predictions
        return ( self.predict(data),
                 { 0 : 'magnetosphere',
                   1 : 'magnetosheath',
                   2 : 'solar_wind' } )
        
    
    
    
    
    
    def predict_magsphere_magsheath_foreshock_pristinesolarwind(self, data):
        """
        Make predictions for magnetosheath, magnetosphere, pristine solar wind,
        and ion foreshock
        
        Note that this 4-class model accuracy is about 86% (on the olshevsky
        labeled dataset), so use caution with ion foreshock vs pristine
        solar wind predictions.

        Parameters
        ----------
        data : pandas dataframe containing necessary vars for clustering

        Returns
        -------
        2-element tuple where
          0: 1d numpy array of integers representing cluster predictions
          1: dict(int->str) indicating cluster affiliation
        """
        
        # Get 3-class predictions (model will be reset)
        class3_preds, _ = self.predict_magsphere_magsheath_solarwind(data)
        
        # Unpack solar wind cluster into two sub-clusters
        init_sw_cluster_int = 2
        subset = (self.default_aggclust_dist, init_sw_cluster_int)
        self.prepare_aggclust(subset=subset)
        sub_sw_preds = self.predict(data, dist=0.5)
        
        # mask out non-solar wind predictions
        sw_mask = sub_sw_preds != -1
        
        # increment sub-sw preds from (0,1) by 2 to (2,3), then assign to
        # original preds
        new_preds = np.zeros(data.shape[0], np.int32)
        # keep original magsphere and magsheath (0 and 1) predictions ...
        new_preds[~sw_mask] = class3_preds[~sw_mask]
        # ... but split sw predictions (old 2) to both pristine solar
        # wind (3) and ion foreshock (new 2)
        new_preds[sw_mask] = sub_sw_preds[sw_mask] + 2
        
        return ( new_preds,
                 { 0 : 'magnetosphere',
                   1 : 'magnetosheath',
                   2 : 'ion_foreshock', 
                   3 : 'pristine_solar_wind' } )
    
    
    
    
    
    def som_activations(self, data):
        """
        Creates a dict of node position to index of data, e.g. if the data
        point at index 5 belongs to node (3,5), then the returned
        dict will contain something like { (3,5): [5] }. If a
        node is absent from the dict, then no points mapped to it.

        Parameters
        ----------
        data : pandas dataframe containing necessary vars for clustering

        Returns
        -------
        dict( 2-element int tuple -> 1d integer numpy array )
        """
        
        # Prepare data for the som
        dat_for_som = self._prepare_data_for_som( data )
        
        # Get dict of som nodes to data indices
        som_preds = self.som.labels_map( dat_for_som,
                                         np.arange(dat_for_som.shape[0]) )
        
        # Convert from original format to arrays as values
        node_dict = {}
        for key in som_preds:
            node_dict[key] = np.array( list( som_preds[key].keys() ) )
        
        return node_dict
        
        
        
        
