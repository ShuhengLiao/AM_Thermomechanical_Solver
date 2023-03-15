import os
import subprocess
import importlib
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import run_sim
import pandas as pd
import run_sim as rs
import time

importlib.reload(run_sim)

# Verify GPU status with nvidia-smi prior to activating GPU
# todo: configure 
MAGI_0 = cp.cuda.Device(0)
MELCHIOR_1 = cp.cuda.Device(1)
BALTHASAR_2 = cp.cuda.Device(2)
CASPAR_3 = cp.cuda.Device(3)

# Name of geometry to draw data from
sim_dir_name = "thin_wall"

# rclone directory - change for your device
rclone_stream = "ONEDRIVE-NU:"
dest_dir = os.path.join("DED-DT - IDEAS Lab", "08-Technical", "data-gamma")

num_LP = 5
with MELCHIOR_1.use():
    for itr in range(0, num_LP):
        # Laser file iteration
        laser_file = "LP_" + str(itr+1)
        # Create simulation object
        sim_itr = rs.FeaModel(geom_dir=sim_dir_name, laserpowerfile=laser_file, outputstep = 1, outputVtkFiles=True)
        # Run simulation
        sim_itr.run()
        # upload output files
        sim_itr.OneDriveUpload(rclone_stream=rclone_stream, destination=dest_dir)