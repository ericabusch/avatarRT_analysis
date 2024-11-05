# this script is being written on 10/31/24 to try to debug why there are missing values in the results
import numpy as np
import pandas as pd
import os, sys, glob, shutil,argparse,pickle,inspect, subprocess
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
sys.path.insert(0,'../../..')
from mpi4py import MPI
import matplotlib.pyplot as plt
import avatarRT_utils as utils
from numpy import linalg
import nibabel as nib
from scipy.stats import spearmanr, pearsonr,zscore,f_oneway, spearmanr
from nilearn.maskers import NiftiMasker
from nilearn import plotting, image  
from nibabel import Nifti1Image
from brainiak.searchlight.searchlight import Searchlight
from analysis_utils import shift_timing, load_location_labels, list_by_trial
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LinearRegression
from config import *
np.random.seed(SEED)