import sys
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import run_sim
import pandas as pd
import run_sim as rs

importlib.reload(run_sim)

# Verify GPU status with nvidia-smi prior to activating GPU
#cp.cuda.Device(0).use()

# Name of geometry to draw data from
sim_dir_name = "thin_wall"

num_LP = 4 # Number of laser profiles to run
for itr in range(0, num_LP):
    sim_itr = rs.FeaModel(geom_dir=sim_dir_name, laserpowerfile=("LP_" + str(itr+1)), outputstep = 0.002)
    sim_itr.run()