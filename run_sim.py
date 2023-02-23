import sys
sys.path.append('../../includes/')
import os
from preprocessor import write_keywords,write_birth,write_parameters
from gamma import domain_mgr, heat_solve_mgr,load_toolpath,get_toolpath
import cupy as cp
import numpy as np
import pyvista as pv
import vtk
cp.cuda.Device(0).use()


class FeaModel():

    def __init__(self):

        ## ACTIVATE DOMAIN
        self.geometry_name = "wall.k"
        self.domain = domain_mgr(filename=self.geometry_name)
        self.heat_solver = heat_solve_mgr(domain)

        ## RUN SIMULATION
        self.output_step = 2  # output time step

        # initialization
        self.file_num = 0
        if not os.path.exists('./vtk'):
            os.mkdir('./vtk')

        # save file
        # filename = 'vtk/u{:05d}.vtk'.format(self.file_num)
        # save_vtk(filename)
        self.file_num = self.file_num + 1
        output_time = self.domain.current_time

        # time loop
        while self.domain.current_time < domain.end_time - 1e-8:
            self.heat_solver.time_integration()
            
            # save file
            if self.domain.current_time >= output_time + output_step:
                print("Current time:  {}, Percentage done:  {}%".format(
                    self.domain.current_time, 100 * self.domain.current_time / self.domain.end_time))
                # filename = 'vtk/u{:05d}.vtk'.format(self.file_num)
                # save_vtk(filename)
                self.file_num = self.file_num + 1
                output_time = self.domain.current_time

    ## DEFINE SAVE VTK FILE FUNCTION
    def save_vtk(filename):
        active_elements = domain.elements[domain.active_elements].tolist()
        active_cells = np.array([item for sublist in active_elements for item in [8] + sublist])
        active_cell_type = np.array([vtk.VTK_HEXAHEDRON] * len(active_elements))
        points = domain.nodes.get()
        active_grid = pv.UnstructuredGrid(active_cells, active_cell_type, points)
        active_grid.point_data['temp'] = heat_solver.temperature.get()
        active_grid.save(filename)