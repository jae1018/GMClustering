#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic packages
import pickle
import os
import itertools
import pkg_resources

# computational packages
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

# visualization packages
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# etc
from tqdm import tqdm





def engineer_features(df,
                      scalar_labels = None,
                      log_scalar    = None):
    
    
    
    def build_duo_scalar_combos(df_, vars_, log_scalar=None):
        if log_scalar is None: log_scalar = True
        
        duo_combos = list(itertools.combinations(vars_,2))
        for combo in duo_combos:
            comp1, comp2 = combo
            
            func = (lambda x: x) if not log_scalar else np.log10
            
            # product (X*Y)
            df_[comp1+'*'+comp2] = func( df_[comp1] * df_[comp2] )
            # ratio (X/Y)
            df_[comp1+'/'+comp2] = func( df_[comp1] / df_[comp2] )

    
    new_df = df.copy(deep=True)
        
    build_duo_scalar_combos(new_df, scalar_labels)
    #build_triple_scalar_combos(new_df, scalar_labels)
        
    
    return new_df
        







class GMClustering:
    
    init_vars = [ 'BX', 'BY', 'BZ',
                  'VX', 'VY', 'VZ',
                  'T', 'n' ]
    
    derived_vars = ['p', 'beta', 'MA']
    
    log_vars = [ 'T', 'n', 'p', 'beta', 'MA' ]
    
    default_aggclust_kws = {'linkage'    : 'ward',
                            'n_clusters' : None}
    default_aggclust_dist = 3
    
    def __init__(self):
        
        stream = pkg_resources.resource_filename(__name__, 'models')
        self.model_folder = stream
        self.load_models()
        self.node_mask = np.full( self.som_weights_2d().shape[0], True )
        self.current_dist = GMClustering.default_aggclust_dist
        self.prepare_aggclust()
        
        
    
    def _load_model(self, filename):
        file = open( os.path.join(self.model_folder,filename), 'rb' )
        return pickle.load(file)
    
    
    
    def load_models(self):
        
        self.init_scaler = self._load_model('init_scaler.pkl')
        self.init_pca = self._load_model('pca.pkl')
        self.post_pca_scaler = self._load_model('post_pca_scaler.pkl')
        self.som = self._load_model('som.pkl')
        
        
        
    def calculate_derived_params(df):
        
        new_df = df.copy(deep=True)
        
        # constants
        m_ion = 1.67 * 10**-27
        mu_0 = 4 * np.pi * 10**(-7)
        boltz_k = 1.380649 * 10**(-23)   # boltzmann constant in J/K
        kelvins_per_eV = 11604.51812
        # values from df
        density_numPcc = new_df['n'].values
        density_numPm3 = density_numPcc * (100**3)   # N/(cm^3) --> N/(m^3)
        temp_eV = new_df['T'].values
        b_field_magnitude_nT = np.sqrt( (new_df[['BX','BY','BZ']]**2).sum(axis=1) )
        b_field_magnitude_T = b_field_magnitude_nT * (10**-9)    # nT --> Tesla
        speed_kmPs = np.sqrt( (new_df[['VX','VY','VZ']]**2).sum(axis=1) )
        # calculate pressure (nPa)
        pressure_nPa = ( density_numPm3
                         * boltz_k
                         * temp_eV
                         * kelvins_per_eV   # Kelvins / eV
                         * 10**9 )   # Pascals --> nanoPascals
        new_df['p'] = pressure_nPa
        # calculate plasma beta
        new_df['beta'] = (
            ( pressure_nPa * 10**-9 )   # nPa --> Pa
                /
            (  b_field_magnitude_T**2 / (2 * mu_0)  )
                            )
        # calculate alfven speed (m/s)
        alfven_speed_mps = (
                b_field_magnitude_T
                        /
                np.sqrt( mu_0 * m_ion * density_numPm3 )
                            )
        #new_df['VA'] = alfven_speed_mps
        # calculate alfven mach number
        new_df['MA'] = speed_kmPs * 1000 / alfven_speed_mps
        
        return new_df
    
    
    def dimensional_expansion(df):
        
        scalar_labels = [ 'T','n','beta','p','MA' ]
        model_df = engineer_features(df,
                                     log_scalar = True,
                                     scalar_labels = scalar_labels)
        del model_df['T*n']   # pressure
        del model_df['T/p']   # 1/(k*density)
        del model_df['n/p']   # 1/(k*temp)
        for var in GMClustering.log_vars:
            model_df[var] = np.log10( model_df[var] )
        
        return model_df
    
    
    def som_cluster_mapper(self, data_for_som, cmodel_preds):
       
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
    
    
    def scaled_embeddings(self, dim_exp_data):
        # rescale and pca transform data      
        trimmed_scaled_data = self.post_pca_scaler.transform(
                                self.init_pca.transform(
                                    pd.DataFrame(
                                        self.init_scaler.transform(dim_exp_data),
                                        columns = list(dim_exp_data)
                                                )
                                                        )
                                                                    )
        return trimmed_scaled_data
        
    
    
    def som_weights_2d(self):
        num_neurons = np.prod( self.som._weights.shape[:2] )
        shape_2d = (num_neurons, self.som._weights.shape[2] )
        return self.som._weights.reshape(shape_2d)
    
    
    
    def aggclust_predictions_on_som_nodes(self, dist = None):
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
    
    
    
    def dendrogram(self, dendrogram_kws = None,
                         fig_kws        = None,
                         ax             = None):
        if dendrogram_kws is None: dendrogram_kws = {}
        default_dendrogram_kws = {'color_threshold':0}
        dendrogram_kws = { **default_dendrogram_kws, **dendrogram_kws }
        
        if fig_kws is None: fig_kws = {'figsize':(6,6)}
        
        if ax is None:
            fig, ax = plt.subplots(1,1,**fig_kws)
        else:
            fig = plt.gcf()
        
        linkage_matrix = linkage(
                    self.som_weights_2d()[self.node_mask],
                    method = GMClustering.default_aggclust_kws['linkage']
                                )
        max_dist = self.aggclust.distances_.max()
        dendrogram(linkage_matrix, ax=ax, **dendrogram_kws)
        for vert_val in np.arange(0.5, int(max_dist)+0.5, step=0.5):
            ax.axhline(vert_val, ls='solid', c='grey', alpha=0.5)
        ax.xaxis.set_ticklabels([])
        
        dist = self.aggclust.distance_threshold
        ax.axhline(dist, ls='dashed', c='black')
        
        return fig, ax
    
    
    
    def prepare_data_for_som(self, data, rows_per_batch = None):
        
        # Check that data contains necessary vars
        num_init_vars = np.sum( np.isin(list(data), GMClustering.init_vars) )
        if num_init_vars != len(GMClustering.init_vars):
            raise ValueError('data does not contain all of the necessary '
                             + 'vars: ' + str(GMClustering.init_vars))
        
        if rows_per_batch is None: rows_per_batch = data.shape[0]
        num_batches = int( data.shape[0] / rows_per_batch )
        if num_batches == 0: num_batches += 1
        
        processed_data = []
        for chunk in tqdm( np.array_split(data[GMClustering.init_vars],
                                          num_batches,
                                          axis=0) ):
        
            # calculate extra params before dim expansion
            df_with_derived_params = GMClustering.calculate_derived_params(chunk)
            
            # do dimensional expansion to capture deeper correlations
            dim_exp_data = \
                GMClustering.dimensional_expansion( df_with_derived_params )
            dim_exp_data = dim_exp_data[ self.init_scaler.feature_names_in_ ]
        
            # get (scaled) embeddings from autoencoder
            processed_data.append( self.scaled_embeddings( dim_exp_data ) )
        
        # get AggClust predictions of SOM nodes 
        return np.vstack(processed_data)
    
    
    
    def som_hits(self, data,
                       ax             = None,
                       rows_per_batch = None,
                       pcolor_kws     = None):
        
        if pcolor_kws is None: pcolor_kws = {}
        default_pcolor_kws = {'cmap':'gray'}
        pcolor_kws = { **default_pcolor_kws, **pcolor_kws }
        
        som_dat = self.prepare_data_for_som(data,
                                            rows_per_batch = rows_per_batch)
        
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
        return ( self.som._neigx.max()+1, self.som._neigy.max()+1 )



    def som_clust(self, dist       = None,
                        ax         = None):
        
        cmodel_preds = \
            self.aggclust_predictions_on_som_nodes(dist = dist)
        
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
    
        #ax.pcolor( cmodel_preds.reshape(self.som_shape()).T,
        #           cmap       = cmap,
        #           **pcolor_kws)
        
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
        Subset is tuple / list of tuples of (distance_threshold, cluster)
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
                      dist           = None,
                      rows_per_batch = None):
        
        scaled_embeds = self.prepare_data_for_som(
                                data,
                                rows_per_batch = rows_per_batch
                                                 )
        
        # get AggClust predictions of SOM nodes 
        cmodel_preds = \
            self.aggclust_predictions_on_som_nodes(dist = dist)
            
        return self.som_cluster_mapper(scaled_embeds, cmodel_preds)
        
        
        
