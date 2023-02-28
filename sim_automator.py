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
cp.cuda.Device(0).use()

# Name of geometry to draw data from
sim_dir_name = "thin_wall"

# rclone directory - change for your device
rclone_stream = "ONEDRIVE-NU:"

num_LP = 5 # Number of laser profiles to run
for itr in range(0, num_LP):
    # Laser file iteration
    laser_file = "LP_" + str(itr+1)

    # Directory of output
    output_dir = os.path.join(sim_dir_name, laser_file)

    # Create simulation object
    sim_itr = rs.FeaModel(geom_dir=sim_dir_name, laserpowerfile=laser_file, outputstep = 1)
    # Run simulation
    sim_itr.run()
    

    ## Uploading
    ## TODO - write this section into a subroutine, so the next sim can run concurrently with the uploading process.
    zarpth = os.path.join("./zarr_output", output_dir) + ".zarr"
    sendpath = os.path.join(rclone_stream, "DED-DT - IDEAS Lab", "08-Technical", "data-gamma")

    new_outpath = os.path.join(sendpath, zarpth)

    # Zip .zarr file
    subprocess.run('tar -czf "' + sim_dir_name +"_" + laser_file + '.tar.gz" "' + zarpth + '"', shell=True, executable='/bin/bash')

    # Upload to cloud using rsync
    subprocess.run('rclone sync "' + sim_dir_name + '_' + laser_file + '.tar.gz" "' + new_outpath + '" -vv ', shell=True, executable='/bin/bash')