import os
import subprocess
import importlib
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import run_sim
import pandas as pd
import run_sim as rs
from multiprocessing import Process
import time
importlib.reload(run_sim)

def CallRunSim(GPUse, SimSet, StartWallTime):
    with cp.cuda.Device(GPUse).use():

        # Set unlimited memory for mempool
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=999999999999)
        for itr in range(0, len(SimSet)):
            # Laser file iteration
            laser_file = prefix + str(SimSet[itr]+1)
            
            # Create simulation object
            sim_itr = rs.FeaModel(geom_dir=sim_dir_name,
                                laserpowerfile=laser_file,
                                VtkOutputStep = 1,
                                ZarrOutputStep = 0.02,
                                outputVtkFiles=True,
                                verbose=True)
            
            # Run simulation
            sim_itr.run()
            EndWallTime = time.perf_counter()
            WallElapsedTime = EndWallTime - StartWallTime

            print(f"Elased Time for Simulation {(SimSet[itr]+1):.0f}: {WallElapsedTime:0.4f}")
            # upload output files
            sim_itr.OneDriveUpload(rclone_stream=rclone_stream, destination=dest_dir, BashLoc='/bin/bash')

# Verify GPU status with nvidia-smi prior to activating GPU
# Name of geometry to draw data from
sim_dir_name = "thin_wall"

# rclone directory - change for your device
rclone_stream = "ONEDRIVE-NU:"
dest_dir = os.path.join("DED-DT - IDEAS Lab", "08-Technical", "data-gamma")

# prefix for laser power signals
prefix = "NLP_"

# Simulations to run
sim_list = []
sim_list.append(range(0, 25))
sim_list.append(range(25, 50))
sim_list.append(range(50, 75))
sim_list.append(range(75, 100))

#sim_list.append(range(0, 1))
#sim_list.append(range(1, 2))
#sim_list.append(range(2, 3))
#sim_list.append(range(3, 4))
#sim_list.append(range(4, 5))
NumGPUs = len(sim_list)

# GPU assignments
GPULIST = np.array([0, 0, 0, 0], dtype=int)

if len(GPULIST) < NumGPUs:
   Exception("Error! More GPUs requested than assigned.")

processes = []
for jtr in range(0, NumGPUs):
    StartWallTime = time.perf_counter()
    processes.append(Process(target=CallRunSim, args=(GPULIST[jtr], sim_list[jtr], StartWallTime)))
    processes[jtr].start()

# Run script by running from terminal: $ nohup python3 -u sim_automator.py &