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

# For debugging gamma.py or preprocessor, uncomment
importlib.reload(sys.modules['includes.gamma'])
importlib.reload(sys.modules['includes.preprocessor'])

class FeaModel():
    def __init__(self, geom_dir, laserpowerfile, outputstep = 1, outputVtkFiles = False):

        ## Setting up resources
        # laserpowerfile: profile of laser power w.r.t time
        self.laserpowerfile = laserpowerfile

        # geom_dir: directory containing .k input file and toolpath.crs file
        self.geom_dir = geom_dir

        # Location of geometry and laser power sequence
        self.geometry_file = os.path.join("geometries-toolpaths", self.geom_dir, "inp.k")
        self.toolpath_file = os.path.join("geometries-toolpaths", self.geom_dir, "toolpath.crs")

        # Start heat_solver and simulation domain
        self.domain = domain_mgr(filename=self.geometry_file, toolpathdir=self.toolpath_file)
        self.heat_solver = heat_solve_mgr(self.domain)
        
        # Read laser power input and timestep-sync file
        inp = pd.read_csv(os.path.join("laser_inputs", self.geom_dir, self.laserpowerfile) + ".csv").to_numpy()
        self.laser_power_seq = inp[:, 0]
        self.timesteps = inp[:, 1]
        self.max_itr = len(self.timesteps)

        ### Initialization of outputs
        self.zarr_stream = DataRecorder(nnodes=self.domain.nodes.shape[0], outputFolderPath=os.path.join("./zarr_output", self.geom_dir, self.laserpowerfile)+".zarr")

        # file_num: .vtk output iteration
        self.file_num = 0

        # output_times: vector containing expected times at which a vtk file is outputted.
        self.output_step = outputstep  # Time step between iterations
        self.output_times = np.linspace(0, self.output_step*(self.max_itr), (self.max_itr+1))

        # Save initial state as vtk file
        self.outputVtkFiles = outputVtkFiles
        if self.outputVtkFiles:
            filename = os.path.join('vtk_files', self.geom_dir, self.laserpowerfile, 'u{:05d}.vtk'.format(self.file_num))
            self.save_vtk(filename)
        self.file_num = self.file_num + 1
    
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
            self.RecordToZarr()

            # save .vtk file if the current time is greater than an expected output time
            # offset by dt/10 due to floating point error
            # honestly this whole thing should really be done with integers
            if self.domain.current_time >= (self.output_times[self.file_num] - (self.domain.dt/10)):

                # Print time and completion status to terminal
                print("Current time:  {} s, Percentage done:  {}%".format(
                    self.domain.current_time, 100 * self.domain.current_time / self.domain.end_time))
                
                # vtk file filename and save
                if self.outputVtkFiles:
                    filename = os.path.join('vtk_files', self.geom_dir, self.laserpowerfile, 'u{:05d}.vtk'.format(self.file_num))
                    self.save_vtk(filename)
                    
                # iterate file number
                self.file_num = self.file_num + 1
                self.output_time = self.domain.current_time

                # save other data to csv files, for training
                # unindent to save at every time step.
                # WARNING: can generate a lot of data in a very short amount of time if unindented!
                # Ensure the drive this is run on has enough storage
                #self.OldRecordDataPoint()
        # Post-simulation tasks here

    # def OldRecordDataPoint(self):
    #     ''' Record a single datapoint at the current simulation timestep. '''

    #     # Open file stream
    #     output_obj = DataRecorder(outputFolderPath=os.path.join("./output", self.geom_dir, self.laserpowerfile))

    #     # Write outputs to file
    #     self.heat_solver.laser_loc[0].tofile(output_obj.files["pos_x"], sep=',', format='%.10E')
    #     self.heat_solver.laser_loc[1].tofile(output_obj.files["pos_y"], sep=',', format='%.10E')
    #     self.heat_solver.laser_loc[2].tofile(output_obj.files["pos_z"], sep=',', format='%.10E')
    #     self.heat_solver.q_in.tofile(output_obj.files["laser_power"], sep=',', format='%.10E')
    #     self.heat_solver.temperature.tofile(output_obj.files["ff_temperature"], sep=',', format='%.10E')
    #     self.domain.active_nodes.tofile(output_obj.files["active_nodes"], sep=',', format='%d')
    #     np.array([self.domain.current_time], dtype=np.float64).tofile(output_obj.files["timestamp"], sep=',', format='%.10E')

    #     # Write line breaks
    #     for datastream in output_obj.files:
    #         output_obj.files[datastream].write("\r\n")

    #     # Delete output object, closing file streams
    #     del output_obj
    #     pass
    
    def RecordToZarr(self):
        '''Records a single data point to a zarr file'''
        
        # For each of the data streams, append the data for the current time step
        # expanding dimensions as needed to match
        self.zarr_stream.streamobj["timestamp"].append(np.expand_dims(np.expand_dims(self.domain.current_time, axis=0), axis=1), axis=0)
        self.zarr_stream.streamobj["pos_x"].append(np.expand_dims(np.expand_dims(self.heat_solver.laser_loc[0].get(), axis=0), axis=1), axis=0)
        self.zarr_stream.streamobj["pos_y"].append(np.expand_dims(np.expand_dims(self.heat_solver.laser_loc[1].get(), axis=0), axis=1), axis=0)
        self.zarr_stream.streamobj["pos_z"].append(np.expand_dims(np.expand_dims(self.heat_solver.laser_loc[2].get(), axis=0), axis=1), axis=0)
        self.zarr_stream.streamobj["laser_power"].append(np.expand_dims(np.expand_dims(self.heat_solver.q_in, axis=0), axis=1), axis=0)
        self.zarr_stream.streamobj["active_nodes"].append(np.expand_dims(self.domain.active_nodes.astype('i1'), axis=0), axis=0)
        self.zarr_stream.streamobj["ff_temperature"].append(np.expand_dims(self.heat_solver.temperature.get(), axis=0), axis=0)

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
        outputFolderPath = "ouput",
    ):
        
        # Location to save file
        self.outputFolderPath = outputFolderPath

        # Types of data being captured.
        self.dataStreams = [
            "timestamp",
            "pos_x",
            "pos_y",
            "pos_z",
            "laser_power",
            "active_nodes",
            "ff_temperature"
        ]

        # Dimension of one time-step of each data stream
        dims = [1, 1, 1, 1, 1, nnodes, nnodes]

        # Type of each data stream
        types = ['f8', 'f8', 'f8', 'f8', 'f8', 'i1', 'f8']
        self.dimsdict = {self.dataStreams[itr]:dims[itr] for itr in range(0, len(self.dataStreams))}
        self.typedict = {self.dataStreams[itr]:types[itr] for itr in range(0, len(self.dataStreams))}

        # dict containing the data streams themselves
        self.streamobj = dict.fromkeys(self.dataStreams)

        # Create zarr arrays for each data stream with length 1
        self.out_root = z.group(outputFolderPath)
        for stream in self.dataStreams:
            try:
                self.streamobj[stream] = self.out_root.create_dataset(stream, shape=(1, self.dimsdict[stream]), dtype=self.typedict[stream])
            except:
                # Fails if directory already exists
                raise Exception("Error! Base directory not empty!")
                pass

if __name__ == "__main__":
    model = FeaModel('thin_wall', 'LP_1')
    model.run()