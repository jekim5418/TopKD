import numpy as np

import gudhi as gd
from gudhi.representations import DiagramSelector
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from gudhi.representations.vector_methods import PersistenceImage


def get_barcode(point, h_dim=0): 
    RipsM = gd.RipsComplex(points=point)
    RipsM_tree = RipsM.create_simplex_tree(max_dimension = h_dim+1)
    RipsM_tree.persistence()

    if h_dim == 0:
        BarCodes_Rips0 = RipsM_tree.persistence_intervals_in_dimension(0)
        return BarCodes_Rips0
        
    elif h_dim == 1:
        BarCodes_Rips0 = RipsM_tree.persistence_intervals_in_dimension(0)
        BarCodes_Rips1 = RipsM_tree.persistence_intervals_in_dimension(1)
        return BarCodes_Rips0, BarCodes_Rips1

def dataset_to_dg(dataset, h_dim=1):
    PD_train0 = []
    PD_train1 =[]
    if h_dim == 1:
        for X in tqdm(dataset):
            dg0, dg1 = get_barcode(point=X, h_dim=h_dim)
            if len(dg0) == 0:
                dg0 = np.empty([0,2])
            if len(dg1) == 0:
                dg1 = np.empty([0,2])
            PD_train0.append(dg0)
            PD_train1.append(dg1)
        return PD_train0, PD_train1
    else:
        for X in tqdm(dataset):
            dg0 = get_barcode(point=X, h_dim=h_dim)
            if len(dg0) == 0:
                dg0 = np.empty([0,2])
            PD_train0.append(dg0)
        return PD_train0


def dg_to_PI(PD_train):
        
    pds_train = DiagramSelector(use=True).fit_transform(PD_train)

    vpdtr = np.vstack(pds_train)
    pers = vpdtr[:,1]-vpdtr[:,0]
    bps_pairs = pairwise_distances(np.hstack([vpdtr[:,0:1],vpdtr[:,1:2]-vpdtr[:,0:1]])[:200]).flatten()
    ppers = bps_pairs[np.argwhere(bps_pairs > 1e-5).ravel()]
    sigma = np.quantile(ppers, .2)
    im_bnds = [np.quantile(vpdtr[:,0],0.), np.quantile(vpdtr[:,0],1.), np.quantile(pers,0.), np.quantile(pers,1.)]

    PI_params = {'bandwidth': sigma, 'weight': lambda x: 10*np.tanh(x[1])+10*np.log(100*x[0]+1), 
             'resolution': [20,20], 'im_range': im_bnds}
    PI_train = PersistenceImage(**PI_params).fit(pds_train).transform(pds_train)
    MPI = np.max(PI_train)
    PI_train /= MPI

    return PI_train









