import sys
import importlib
import os
from includes.preprocessor import write_keywords, write_birth, write_parameters
from includes.gamma import domain_mgr, heat_solve_mgr, load_toolpath, get_toolpath
import cupy as cp
import numpy as np
import pyvista as pv
import vtk
import pandas as pd
import warnings
# import h5py
import zarr as z
import subprocess
import time
import datetime
import numba

# For debugging gamma.py or preprocessor, uncomment
importlib.reload(sys.modules['includes.gamma'])
importlib.reload(sys.modules['includes.preprocessor'])

class FeaModel():
    def __init__(self, geom_dir, laserpowerfile, VtkOutputStep=1, ZarrOutputStep=0.02, outputVtkFiles=True, verbose=True):

        ## Setting up resources
        # output
        self.verbose = verbose

        # laserpowerfile: profile of laser power w.r.t time
        self.laserpowerfile = laserpowerfile

        # geom_dir: directory containing .k input file and toolpath.crs file
        self.geom_dir = geom_dir

        # Location of geometry and laser power sequence
        self.geometry_file = os.path.join("geometries-toolpaths", self.geom_dir, "inp.k")
        self.toolpath_file = os.path.join("geometries-toolpaths", self.geom_dir, "toolpath.crs")

        # Start heat_solver and simulation domain
        self.domain = domain_mgr(filename=self.geometry_file, toolpathdir=self.toolpath_file, verbose=self.verbose)
        self.heat_solver = heat_solve_mgr(self.domain)
        
        # Read laser power input and timestep-sync file
        inp = pd.read_csv(os.path.join("laser_inputs", self.geom_dir, self.laserpowerfile) + ".csv").to_numpy()
        self.laser_power_seq = inp[:, 0]
        self.timesteps = inp[:, 1]
        self.las_max_itr = len(self.timesteps)

        # las_max_itr: length of laser input signal
        # def_max_itr: time for original simulation to run to completion
        self.def_max_itr = int(self.domain.end_time/self.domain.dt)
        self.max_itr = min(self.las_max_itr, self.def_max_itr)

        # VTK output_times: vector containing expected times at which a vtk file is outputted.
        self.VtkOutputStep = VtkOutputStep  # Time step between iterations
        self.VtkOutputTimes = np.linspace(0, self.VtkOutputStep*self.max_itr, self.max_itr+1)
        self.VtkOutputTimes = [x for x in self.VtkOutputTimes if x <= self.domain.end_time]
        exp_vtk_num = len(self.VtkOutputTimes)

        # Zarr output steps: vector containing expected times at which a zarr file is generated
        self.ZarrOutputStep = ZarrOutputStep
        self.ZarrOutputTimes = np.linspace(0, self.ZarrOutputStep*self.max_itr, self.max_itr+1)
        self.ZarrOutputTimes = [x for x in self.ZarrOutputTimes if x <= self.domain.end_time]
        exp_zarr_len = len(self.ZarrOutputTimes)

        ### Initialization of outputs
        # Start datarecorder object to save pointwise data
        self.zarr_stream = DataRecorder(nnodes=self.domain.nodes.shape[0],
                                        nele=self.domain.elements.shape[0],
                                        outputFolderPath=(os.path.join("./zarr_output",
                                                                      self.geom_dir,
                                                                      self.laserpowerfile) +".zarr"),
                                        ExpOutputSteps=exp_zarr_len)

        # Record nodes and nodal locations 
        self.zarr_stream.nodelocs = self.domain.nodes
        self.zarr_stream.ele = self.domain.elements

        # VtkFileNum: .vtk output iteration
        # ZarrFileNum
        self.VtkFileNum = 0
        self.ZarrFileNum = 0

        # Save initial state as vtk file
        self.outputVtkFiles = outputVtkFiles
        if self.outputVtkFiles:
            filename = os.path.join('vtk_files', self.geom_dir, self.laserpowerfile, 'u{:05d}.vtk'.format(self.VtkFileNum))
            self.save_vtk(filename)
        self.VtkFileNum = self.VtkFileNum + 1
    
    def run(self):
        ''' Run the simulation. '''

        # Time loop
        while self.domain.current_time < self.domain.end_time - 1e-8 and self.heat_solver.current_step < self.max_itr :
            # Load the current step of the laser profile, and multiply by the absortivity
            self.heat_solver.q_in = self.laser_power_seq[self.heat_solver.current_step]*self.domain.absortivity
            
            # Check that the time steps agree
            if np.abs(self.domain.current_time - self.timesteps[self.heat_solver.current_step]) / self.domain.dt > 0.01:
                # Check if the current domain is correct
                # In the future, probably best to just check this once at the beginning instead of every iteration
                warnings.warn("Warning! Time steps of LP input are not well aligned with simulation steps")

            # Run the solver
            self.heat_solver.time_integration()

            # Save timestamped zarr file
            if self.domain.current_time >= (self.ZarrOutputTimes[self.ZarrFileNum] - (self.domain.dt/10)):
                self.ZarrFileNum = self.ZarrFileNum + 1
                self.RecordToZarr()

            # save .vtk file if the current time is greater than an expected output time
            # offset time by dt/10 due to floating point error
            # honestly this whole thing should really be done with integers
            if self.domain.current_time >= (self.VtkOutputTimes[self.VtkFileNum] - (self.domain.dt/10)):

                # Print time and completion status to terminal
                if self.verbose:
                    print("Current time:  {} s, Percentage done:  {}%".format(
                        self.domain.current_time, 100 * self.domain.current_time / self.domain.end_time))
                
                # vtk file filename and save
                if self.outputVtkFiles:
                    filename = os.path.join('vtk_files', self.geom_dir, self.laserpowerfile, 'u{:05d}.vtk'.format(self.VtkFileNum))
                    self.save_vtk(filename)
                    
                # iterate file number
                self.VtkFileNum = self.VtkFileNum + 1
                self.output_time = self.domain.current_time
        # Post Simulation Tasks
        # Write to disk

        #StoreLoc = z.DirectoryStore(outputFolderPath)
        #self.zarr_stream.out_root.store(StoreLoc)

    def OneDriveUpload(self, rclone_stream, destination, BashLoc):
        # Directory of output
        output_dir = os.path.join(self.geom_dir, self.laserpowerfile)

        ## Uploading
        # Todo
        zarpth = os.path.join("./zarr_output", output_dir) + ".zarr"
        vtkpth = os.path.join("./vtk_files", output_dir)
        sendpath = os.path.join(rclone_stream, destination)
        new_outpath = os.path.join(sendpath, output_dir)

        # Zip .zarr file
        TarZarrCmd = 'tar -czf "' + self.geom_dir +"_" + self.laserpowerfile + '_zarr' + '.tar.gz" "' + zarpth + '"'
        # Upload zarr targz
        UploadZarrTarCmd = 'rclone copy "' + self.geom_dir + '_' + self.laserpowerfile + '_zarr' + '.tar.gz" "' + new_outpath + '" -v'
        # Delete zarr targz
        DelZarrTarCmd = 'rm -rf "' + self.geom_dir + '_' + self.laserpowerfile + '_zarr' + '.tar.gz"'
        # Delete original data from drive
        DelZarrOrigCmd = 'rm -rf "' + zarpth + '"'

        # Zip vtk
        TarVTKCmd = 'tar -czf "' + self.geom_dir +"_" + self.laserpowerfile + '_vtk' + '.tar.gz" "' + vtkpth + '"'
        # Upload vtk targz
        UploadVTKTarCmd = 'rclone copy "' + self.geom_dir + '_' + self.laserpowerfile + '_vtk' + '.tar.gz" "' + new_outpath + '" -v'
        # Delete vtk targz
        DelVTKTarCmd = 'rm -rf "' + self.geom_dir + '_' + self.laserpowerfile + '_vtk' +'.tar.gz"'
        # Delete vtk originals from drive
        DelVTKOrigCmd = 'rm -rf "' + vtkpth + '"'

        # Run zarr commands subsequently to upload zarr files to drive
        subprocess.Popen(TarZarrCmd + " && " + DelZarrOrigCmd + " && " + UploadZarrTarCmd  + " && "+ DelZarrTarCmd, shell=True, executable=BashLoc)
        
        # Run commands to upload vtk to drive
        subprocess.Popen(TarVTKCmd + " && " + DelVTKOrigCmd + " && " + UploadVTKTarCmd + " && " + DelVTKTarCmd, shell=True, executable=BashLoc)

    def RecordToZarr(self, outputmode="structured"):
        '''Records a single data point to a zarr file'''

        timestep = np.expand_dims(self.domain.current_time, axis=0)
        pos_x = np.expand_dims(self.heat_solver.laser_loc[0].get(), axis=0)
        pos_y = np.expand_dims(self.heat_solver.laser_loc[1].get(), axis=0)
        pos_z = np.expand_dims(self.heat_solver.laser_loc[2].get(), axis=0)
        laser_power = np.expand_dims(self.heat_solver.q_in, axis=0)
        active_nodes = self.domain.active_nodes.astype('i1')
        ff_temperature = self.heat_solver.temperature.get()
        active_elements = self.domain.active_elements.astype('i1')

        if outputmode == "structured":
            # For each of the data streams, append the data for the current time step
            # expanding dimensions as needed to match
            self.zarr_stream.streamobj["timestamp"][self.ZarrFileNum] = timestep
            self.zarr_stream.streamobj["pos_x"][self.ZarrFileNum] = pos_x
            self.zarr_stream.streamobj["pos_y"][self.ZarrFileNum] = pos_y
            self.zarr_stream.streamobj["pos_z"][self.ZarrFileNum] = pos_z
            self.zarr_stream.streamobj["laser_power"][self.ZarrFileNum] = laser_power
            self.zarr_stream.streamobj["active_nodes"][self.ZarrFileNum] = active_nodes
            self.zarr_stream.streamobj["ff_temperature"][self.ZarrFileNum] = ff_temperature
            self.zarr_stream.streamobj["active_elements"][self.ZarrFileNum] = active_elements

        elif outputmode == "bulked":
            new_row = np.zeros([1, (5+self.domain.nodes.shape[0])])
            new_row[0, 1] = timestep[0, 0]
            new_row[0, 2] = pos_x[0, 0]
            new_row[0, 3] = pos_y[0, 0]
            new_row[0, 4] = pos_z[0, 0]
            new_row[0, 5] = laser_power[0, 0]
            new_row[0, 6:(6+self.domain.nodes.shape[0])] = laser_power[0]
            self.zarr_stream.streamobj["all_floats"].append(new_row, axis=0)
            self.zarr_stream.streamobj["active_nodes"].append(active_nodes, axis=0)

        else:
            raise Exception("Error! Invalid zarr output type!")
    
    ## DEFINE SAVE VTK FILE FUNCTION
    def save_vtk(self, filename):
        active_elements = self.domain.elements[self.domain.active_elements].tolist()
        active_cells = np.array([item for sublist in active_elements for item in [8] + sublist])
        active_cell_type = np.array([vtk.VTK_HEXAHEDRON] * len(active_elements))
        points = self.domain.nodes.get()
        active_grid = pv.UnstructuredGrid(active_cells, active_cell_type, points)
        active_grid.point_data['temp'] = self.heat_solver.temperature.get()
        try:
            os.makedirs(os.path.dirname(filename))
            active_grid.save(filename)
        except:
            active_grid.save(filename)


class DataRecorder():
    def __init__(self,
        nnodes,
        nele,
        ExpOutputSteps,
        outputFolderPath,
        outputmode = "structured"
    ):
        
        # Location to save file
        if outputmode == "structured":
            self.outputFolderPath = outputFolderPath
            # Types of data being captured
            self.dataStreams = [
                "timestamp",
                "pos_x",
                "pos_y",
                "pos_z",
                "laser_power",
                "active_nodes",
                "ff_temperature",
                "active_elements"
            ]

            # Dimension of one time-step of each data stream
            dims = [1, 1, 1, 1, 1, nnodes, nnodes, nele]
            # Type of each data stream
            types = ['f8', 'f8', 'f8', 'f8', 'f8', 'i1', 'f8', 'i1']

        elif outputmode == "bulked":
            self.outputFolderPath = outputFolderPath
            self.dataStreams = ["all_floats", "active_nodes", "active_elements"]
            dims = [5 + nnodes, nnodes, nele]
            types = ['f8', 'i1', 'i1']
        else:
            raise Exception("Error! Invalid zarr output type!")

        self.dimsdict = {self.dataStreams[itr]:dims[itr] for itr in range(0, len(self.dataStreams))}
        self.typedict = {self.dataStreams[itr]:types[itr] for itr in range(0, len(self.dataStreams))}

        # dict containing the data streams themselves
        self.streamobj = dict.fromkeys(self.dataStreams)
        
        # Create zarr datasets for each data stream with length 1
        self.out_root = z.group(self.outputFolderPath)
        for stream in self.dataStreams:
            try:
                self.streamobj[stream] = self.out_root.create_dataset(stream,
                                                                      shape=(ExpOutputSteps, self.dimsdict[stream]),
                                                                      chunks=(1, self.dimsdict[stream]),
                                                                      dtype=self.typedict[stream])
            except:
                # Fails if directory already exists
                # todo: ideally, this will ask the user if they want to overwrite the output files
                # and do so with confirmation
                self.streamobj[stream] = self.out_root.create_dataset(stream,
                                                                      shape=(ExpOutputSteps, self.dimsdict[stream]),
                                                                      chunks=(1, self.dimsdict[stream]),
                                                                      dtype=self.typedict[stream],
                                                                      overwrite=True)
                #raise Exception("Error! Base directory not empty!"
        
        # Zarr datasets containing elements, node locations
        self.nodelocs = self.out_root.create_dataset("node_coords", shape=(nnodes, 3), dtype='f8', overwrite=True)
        self.ele = self.out_root.create_dataset("elements", shape=nele, dtype='i8', overwrite=True)

if __name__ == "__main__":
    tic = time.perf_counter()
    model = FeaModel('thin_wall', 'LP_1', ZarrOutputStep=0.2)
    toc1 = time.perf_counter()
    model.run()
    toc2 = time.perf_counter()
    print(f"Time to Simulate: {toc2-toc1:0.4f}")
    print(f"Total time to run: {toc2-tic:0.4f}")

    simtime = toc2-toc1
    projectedtime = (model.domain.end_time/12.854) * simtime

    print(f"End Time: {projectedtime:0.4f}")