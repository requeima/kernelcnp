import torch
import torch.nn as nn
from torch.distributions import Normal,MultivariateNormal
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from utils import *


class GermanyRandomExperimentDataset(Dataset):
  
  def __init__(self, 
               val_size,
               test_inds,
               train_inds,
               train=True):

        super().__init__()

        self.DATA_DIR = "./data/"

        self.x_context = np.load(
            self.DATA_DIR+"context/x_context.npy").transpose(1,0,2)

        self.x_target = np.load(
            self.DATA_DIR+"target/tmax_all_x_target.npy"
            )
        self.y_target = np.load(
            self.DATA_DIR+"target/tmax_all_y_target.npy"
            )
        self.elev = np.load(
            self.DATA_DIR+"elevation/elev_tmax_all.npy"
            )
        

        if train:
            self.y_context = np.memmap(
                self.DATA_DIR+"context/y_context_training_mmap.dat", 
                dtype='float32', 
                mode='r', 
                shape=(8766, 25, 87, 50))
            self.subset_to_germany()
            print(self.x_target.shape)
            self.x_target = self.x_target[train_inds,:]
            self.y_target = self.y_target[:8766, train_inds]
            self.elev = self.elev[train_inds,:]

        else:
            self.y_context = np.memmap(
                self.DATA_DIR+"context/y_context_val_mmap.dat", 
                dtype='float32', 
                mode='r', 
                shape=(2192, 25, 87, 50))
            self.subset_to_germany()
            self.x_target = self.x_target[test_inds,:]
            self.y_target = self.y_target[8766:, test_inds]
            self.elev = self.elev[test_inds,:]
            

        # Calculate dists and scale
        self.dists = get_dists(self.x_target, 
                                self.x_context[...,0], 
                                self.x_context[...,1])
        x_scaler = MinMaxScaler().fit(self.dists.reshape(-1,1))
        self.dists = x_scaler.transform(self.dists.reshape(-1,1)).reshape(self.dists.shape)

  def subset_to_germany(self):
        # Subset everything to 6->16 lon and 47->55 lat
        LON_MIN_IND = 41
        LON_MAX_IND = 55
        LAT_MIN_IND = 23
        LAT_MAX_IND = 34

        # Subset target points to Germany
        inc_inds = ((6<self.x_target[:,0]) & 
           (16>self.x_target[:,0]) & 
           (47<self.x_target[:,1]) & 
           (self.x_target[:,1]<55))
        
        self.y_context = self.y_context[...,LON_MIN_IND:LON_MAX_IND, LAT_MIN_IND:LAT_MAX_IND]
        self.x_context = self.x_context[LON_MIN_IND:LON_MAX_IND, LAT_MIN_IND:LAT_MAX_IND,:]
        self.x_target = self.x_target[inc_inds,:]
        self.y_target = self.y_target[:,inc_inds]
        self.elev = self.elev[inc_inds, :]

  def __len__(self):
        'Denotes the total number of samples'
        return self.y_context.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        y_context = torch.from_numpy(self.y_context[index,...]).float()
        y_target = torch.from_numpy(self.y_target[index,...]).float()
        dists = torch.from_numpy(self.dists).float()
        elev = torch.from_numpy(self.elev).float()
        x_target = torch.from_numpy(self.x_target).float()

        return {"y_context":y_context.cuda(), 
                "y_target":y_target.cuda(), 
                "x_target":x_target.cuda(),
                "dists":dists.cuda(),
                "elev":elev.cuda()}


class ValueExperimentDataset86(Dataset):
  
  def __init__(self, 
               train=True):

        super().__init__()

        self.DATA_DIR = "./data/"

        self.x_context = np.load(
            self.DATA_DIR+"context/x_context_coarse_final.npy")
        self.y_context = np.load(
            self.DATA_DIR+"context/y_context_coarse_final.npy", mmap_mode="r")
        

        if train:
            self.y_context = self.y_context[:8766,...]

            self.x_target = np.load(
                self.DATA_DIR+"target/value_x_target.npy"
                )
            self.y_target = np.load(
                self.DATA_DIR+"target/tmax_value_y_target.npy"
                )[:8766,:]
            self.elev = np.load(
                self.DATA_DIR+"elevation/elev_value.npy"
                )

            #self.x_target = np.load(
            #    self.DATA_DIR+"target/tmax_all_x_target.npy"
          #      )
            #self.y_target = np.load(
            #    self.DATA_DIR+"target/tmax_all_y_target.npy"
            #    )[:8766,:]
            #self.elev = np.load(
            #    self.DATA_DIR+"elevation/elev_tmax_all.npy"
            #    )

        else:
            self.y_context = self.y_context[8766:,...]

            self.x_target = np.load(
                self.DATA_DIR+"target/value_x_target.npy"
                )
            self.y_target = np.load(
                self.DATA_DIR+"target/tmax_value_y_target.npy"
                )[8766:,:]
            self.elev = np.load(
                self.DATA_DIR+"elevation/elev_value.npy"
                )

        # Calculate dists and scale
        self.dists = get_dists(self.x_target, 
                                self.x_context[...,0], 
                                self.x_context[...,1])
        x_scaler = MinMaxScaler().fit(self.dists.reshape(-1,1))
        self.dists = x_scaler.transform(self.dists.reshape(-1,1)).reshape(self.dists.shape)


  def plot_target_locs(self):
        df = pd.DataFrame(
            np.stack([self.x_target[...,0], self.x_target[...,1], self.elev[...,0]], axis=-1),
             columns = ['lon', 'lat','elev'])
    
        fig = px.scatter_mapbox(df,
                                lat='lat',
                                lon='lon',
                                color = 'elev',
                                zoom=2)
        fig.show()

  def __len__(self):
        'Denotes the total number of samples'
        return self.y_context.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        y_context = torch.from_numpy(self.y_context[index,...]).float()
        y_target = torch.from_numpy(self.y_target[index,...]).float()
        dists = torch.from_numpy(self.dists).float()
        elev = torch.from_numpy(self.elev).float()
        x_target = torch.from_numpy(self.x_target).float()

        return {"y_context":y_context.cuda(), 
                "y_target":y_target.cuda(), 
                "x_target":x_target.cuda(),
                "dists":dists.cuda(),
                "elev":elev.cuda()}

class ValueExperimentDatasetAll(Dataset):
  
  def __init__(self, 
               train=True):

        super().__init__()

        self.DATA_DIR = "./data/"

        self.x_context = np.load(
            self.DATA_DIR+"context/x_context_coarse_final.npy")
        self.y_context = np.load(
            self.DATA_DIR+"context/y_context_coarse_final.npy", mmap_mode="r")
        

        if train:
            self.y_context = self.y_context[:8766,...]

            self.x_target = np.load(
                self.DATA_DIR+"target/tmax_all_x_target.npy"
                )
            self.y_target = np.load(
                self.DATA_DIR+"target/tmax_all_y_target.npy"
                )[:8766,:]
            self.elev = np.load(
                self.DATA_DIR+"elevation/elev_tmax_all.npy"
                )

        else:
            self.y_context = self.y_context[8766:,...]

            self.x_target = np.load(
                self.DATA_DIR+"target/value_x_target.npy"
                )
            self.y_target = np.load(
                self.DATA_DIR+"target/tmax_value_y_target.npy"
                )[8766:,:]
            self.elev = np.load(
                self.DATA_DIR+"elevation/elev_value.npy"
                )

        # Calculate dists and scale
        self.dists = get_dists(self.x_target, 
                                self.x_context[...,0], 
                                self.x_context[...,1])
        x_scaler = MinMaxScaler().fit(self.dists.reshape(-1,1))
        self.dists = x_scaler.transform(self.dists.reshape(-1,1)).reshape(self.dists.shape)


  def plot_target_locs(self):
        df = pd.DataFrame(
            np.stack([self.x_target[...,0], self.x_target[...,1], self.elev[...,0]], axis=-1),
             columns = ['lon', 'lat','elev'])
    
        fig = px.scatter_mapbox(df,
                                lat='lat',
                                lon='lon',
                                color = 'elev',
                                zoom=2)
        fig.show()

  def __len__(self):
        'Denotes the total number of samples'
        return self.y_context.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        y_context = torch.from_numpy(self.y_context[index,...]).float()
        y_target = torch.from_numpy(self.y_target[index,...]).float()
        dists = torch.from_numpy(self.dists).float()
        elev = torch.from_numpy(self.elev).float()
        x_target = torch.from_numpy(self.x_target).float()

        return {"y_context":y_context.cuda(), 
                "y_target":y_target.cuda(), 
                "x_target":x_target.cuda(),
                "dists":dists.cuda(),
                "elev":elev.cuda()}