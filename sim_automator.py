import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import run_sim
import pandas as pd

# Verify GPU status with nvidia-smi prior to activating GPU
#cp.cuda.Device(0).use()

# Name of geometry to draw data from
sim_dir_name = "thin_wall"

num_LP = 10 # Number of laser profiles to run
for itr in range(0, num_LP):
    laser_power_seq = pd.read_csv("laser_inputs/" + sim_dir_name + "/LP_" + str(itr+1) + ".csv")
