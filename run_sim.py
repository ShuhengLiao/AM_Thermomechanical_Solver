import sys
import importlib
import os
from includes.preprocessor import write_keywords,write_birth,write_parameters
from includes.gamma import domain_mgr, heat_solve_mgr,load_toolpath,get_toolpath
import cupy as cp
import numpy as np
import pyvista as pv
import vtk
import pandas as pd

# For debugging gamma.py or preprocessor, uncomment
importlib.reload(sys.modules['includes.gamma'])
importlib.reload(sys.modules['includes.preprocessor'])

class FeaModel():

    def __init__(self, geom_dir, laserpowerfile, outputstep = 2):

        ## ACTIVATE DOMAIN
        self.geometry_file = "geometries-toolpaths/" + geom_dir + "/inp.k"
        self.toolpath_file = "geometries-toolpaths/" + geom_dir + "/toolpath.crs"
        self.domain = domain_mgr(filename=self.geometry_file, toolpathdir=self.toolpath_file)

        self.heat_solver = heat_solve_mgr(self.domain)
        
        inp = pd.read_csv("laser_inputs/" + geom_dir + "/" + laserpowerfile).to_numpy()

        self.laser_power_seq = inp[:, 0]
        self.timesteps = inp[:, 1]
        self.max_itr = len(self.timesteps)

        ## RUN SIMULATION
        self.output_step = outputstep  # output time step

        # Initialization
        self.file_num = 0
        self.time_itr = 0

        # save file
        # filename = 'vtk/u{:05d}.vtk'.format(self.file_num)
        # save_vtk(filename)
        self.file_num = self.file_num + 1
        output_time = self.domain.current_time

    def run(self):
        ''' Run the simulation. '''

        # time loop
        while self.domain.current_time < self.domain.end_time - 1e-8 and self.time_itr < self.max_itr :
            # Load the current step of the laser profile
            self.heat_solver.q_in = self.laser_power_seq[self.time_itr]
            
            # Check that the time steps agree
            if np.abs(self.domain.current_time - self.timesteps[self.time_itr]) / self.domain.dt > 0.1:
                raise Exception("Error! Time steps of LP input are not well aligned with simulation steps")

            # Run the solver
            self.heat_solver.time_integration()

            # #save .vtk file
            if self.domain.current_time >= output_time + self.output_step:
                print("Current time:  {}, Percentage done:  {}%".format(
                    self.domain.current_time, 100 * self.domain.current_time / self.domain.end_time))
                # filename = 'vtk/u{:05d}.vtk'.format(self.file_num)
                # save_vtk(filename)
                self.file_num = self.file_num + 1
                output_time = self.domain.current_time

            # Save data
            self.recordDataPoint()

            # Iteratate
            self.time_itr = self.time_itr + 1
    
    def recordDataPoint(self):
        ''' Record a single datapoint at the current simulation timestep. '''
        pass

    ## DEFINE SAVE VTK FILE FUNCTION
    def save_vtk(filename):
        active_elements = self.domain.elements[self.domain.active_elements].tolist()
        active_cells = np.array([item for sublist in active_elements for item in [8] + sublist])
        active_cell_type = np.array([vtk.VTK_HEXAHEDRON] * len(active_elements))
        points = self.domain.nodes.get()
        active_grid = pv.UnstructuredGrid(active_cells, active_cell_type, points)
        active_grid.point_data['temp'] = self.heat_solver.temperature.get()
        active_grid.save(filename)


class DataRecorder():
    def __init__(self,
        outputFolderPath = "./ouput",
        dataStreams = [
            "pos_x",
            "pos_y",
            "pos_z",
            "laser_power",
            "active_nodes",
            "timestamp"
        ]
    ):
        self.outputFolderPath = outputFolderPath
        self.dataStreams = dataStreams

        self.files = {}
        for streamName in dataStreams:
            self.files[streamName] = open(os.path.join(outputFolderPath, streamName.csv))
    
    def __del__(self):
        for _, f in self.files.items():
            f.close()

if __name__ == "__main__":
    model = FeaModel('thin_wall', 'LP_1.csv')
    #model.run()