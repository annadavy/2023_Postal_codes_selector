# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:49:31 2023

@author: Anna Davy
"""

import pandas as pd
import matplotlib.pyplot as plt
import geopandas
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class CodesSelector:    
    
    """
    Class for finding user defined number of zip codes/geopoints that are evenly spread on the territory
    of any country. To run the code it is necessary to have the list of postal codes for the particular
    country (text file) with the geocoordinates (latitude and longitude) of each postal code.
    To use the plot function it is necessary to have the shape files for particular country
    (code was tested on the shape files of Poland). The plot is presenting the cluster spread on the territory 
    of the country depending on the number of codes given as parameter by the user.
    
    """
    
    
    def __init__(self,codes_file,shape_files):
        
        self.codes_file = codes_file
        self.shape_files = shape_files
        self.codes, self.country = self.get_data()
        
        
    def get_data(self):
        
        codes = pd.read_csv(self.codes_file).reset_index(drop=True).drop(columns='Unnamed: 0',axis=1)
        country = geopandas.read_file(self.shape_files)
        
        return codes, country
        
        
    def k_means(self,n_codes):
        
        self.model = KMeans(n_clusters=n_codes).fit(self.codes.drop(['Code'],axis=1))
        self.labels = self.model.predict(self.codes.drop(['Code'],axis=1))
        self.closest, _ = pairwise_distances_argmin_min(self.model.cluster_centers_,\
                                                        self.codes.drop(['Code'],axis=1))
            
        return self.labels, self.closest

    def plot(self,figsize, n_codes):
        
        # figsize is a tuple of plot sizes e.g.(15,25)
        
        self.labels, _ = self.k_means(n_codes)

        plt.figure(figsize=figsize)
        self.country.plot(color='white', edgecolor='black',figsize=figsize)
        plt.scatter(self.codes.Longitude.values, self.codes.Latitude.values, c=self.labels,cmap='prism')
        plt.show()
        
    def get_codes(self, n_codes):
        
        _, self.closest = self.k_means(n_codes)
        
        return pd.DataFrame(self.codes.iloc[self.closest].sort_values(by='Code'))
    
    
