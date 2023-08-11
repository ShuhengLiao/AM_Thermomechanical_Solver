import os
import subprocess
import time
import warnings
import cupy as cp
import numpy as np
import pandas as pd
import pyvista as pv
import vtk
import zarr as z

from gamma.simulator.gamma import domain_mgr, heat_solve_mgr

class FeaModel():
    ''' This class manages the FEA simulation. Use this as the primary interface to the simulation. '''

    def __init__(self, input_data_dir, geom_dir, laserpowerfile, timestep_override=-1, VtkOutputStep=1, ZarrOutputStep=0.02, outputVtkFiles=True, verbose=True, CalcNodeSurfDist=False):
        
        self.timestep_override = timestep_override
        self.outputVtkFiles = outputVtkFiles
        self.CalcNodeSurfDist = CalcNodeSurfDist
        ## Setting up resources
        # output
        self.verbose = verbose

        # geom_dir: directory containing .k input file and toolpath.crs file
        self.geom_dir = geom_dir

        # Location of geometry and laser power sequence
        self.geometry_file = os.path.join(input_data_dir, "geometries-toolpaths", self.geom_dir, "inp.k")
        self.toolpath_file = os.path.join(input_data_dir, "geometries-toolpaths", self.geom_dir, "toolpath.crs")

        # Start heat_solver and simulation domain
        self.domain = domain_mgr(input_data_dir=input_data_dir, filename=self.geometry_file, toolpathdir=self.toolpath_file, verbose=self.verbose, timestep_override=timestep_override)
        self.heat_solver = heat_solve_mgr(self.domain)
        
        # def_max_itr: time for original simulation to run to completion
        self.def_max_itr = int(self.domain.end_sim_time/self.domain.dt)

        # laserpowerfile: profile of laser power w.r.t time
        self.laserpowerfile = laserpowerfile
        if self.laserpowerfile != None:
            # Read laser power input and timestep-sync file
            inp = pd.read_csv(os.path.join(input_data_dir, "laser_inputs", self.geom_dir, self.laserpowerfile) + ".csv").to_numpy()
            self.laser_power_seq = inp[:, 0]
            self.timesteps = inp[:, 1]
            self.las_max_itr = len(self.timesteps)
        else:
            self.laser_power_seq = None
            self.las_max_itr = np.inf

        # las_max_itr: length of laser input signal
        self.max_itr = min(self.las_max_itr, self.def_max_itr)

        # VTK output steps
        self.VtkOutputStep = VtkOutputStep  # Time step between iterations

        # Zarr output steps
        self.ZarrOutputStep = ZarrOutputStep

        ### Initialization of outputs
        # Start datarecorder object to save pointwise data
        if CalcNodeSurfDist:
            self.zarr_stream = AuxDataRecorder(nnodes=self.domain.nodes.shape[0],
                                                outputFolderPath=(os.path.join("./zarr_output",
                                                                            self.geom_dir,
                                                                            self.laserpowerfile) +"_aux.zarr")
            )
            
        else:
            self.zarr_stream = DataRecorder(nnodes=self.domain.nodes.shape[0],
                                            nele=self.domain.elements.shape[0],
                                            outputFolderPath=(os.path.join("./zarr_output",
                                                                        self.geom_dir,
                                                                        self.laserpowerfile) +".zarr")
            )

            # Record nodes and nodal locations 
            self.zarr_stream.nodelocs = self.domain.nodes
            self.zarr_stream.ele = self.domain.elements

        # VtkFileNum: .vtk output iteration
        # ZarrFileNum
        self.VtkFileNum = 0
        self.ZarrFileNum = 0

        # Save initial state as vtk file
        if self.outputVtkFiles:
            if CalcNodeSurfDist:
                filename = os.path.join('vtk_files', self.geom_dir, self.laserpowerfile+"_aux", 'u{:05d}.vtk'.format(self.VtkFileNum))
                self.save_dist_vtk
            else:
                filename = os.path.join('vtk_files', self.geom_dir, self.laserpowerfile, 'u{:05d}.vtk'.format(self.VtkFileNum))
                self.save_vtk(filename)
            self.VtkFileNum = self.VtkFileNum + 1
    
    
    def run(self):
        ''' Run the simulation. '''

        # Time loop
        self.tic_start = time.perf_counter()
        self.tic_jtr = self.tic_start

        self.active_nodes_previous = self.domain.active_nodes.astype('i1')

        while self.domain.current_sim_time < self.domain.end_sim_time - 1e-8 and self.heat_solver.current_step < self.max_itr :
            self.step()

    
    def step(self, laser_power=None):
        ''' Run a single step of the simulation. '''

        # Load the current step of the laser profile, and multiply by the absortivity
        if laser_power == None:
            if self.laserpowerfile == None:
                raise ValueError("No laser power input provided to the step function, and no laser power file provided to the model constructor.")
            self.heat_solver.q_in = self.laser_power_seq[self.heat_solver.current_step] * self.domain.absortivity
        else:
            self.heat_solver.q_in = laser_power * self.domain.absortivity
        
        # Check that the time steps agree
        if np.abs(self.domain.current_sim_time - self.timesteps[self.heat_solver.current_step]) / self.domain.dt > 0.01:
            # Check if the current domain is correct
            # In the future, probably best to just check this once at the beginning instead of every iteration
            warnings.warn("Warning! Time steps of LP input are not well aligned with simulation steps")

        # Run the solver
        self.heat_solver.time_integration()

        # Save timestamped zarr file at specified rate
        if self.heat_solver.current_step % self.ZarrOutputStep == 0:

            # Free unused memory blocks
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

            # Get active nodes.
            active_nodes = self.domain.active_nodes.astype('i1')

            # Save output file
            self.ZarrFileNum = self.ZarrFileNum + 1
            self.RecordTempsZarr(active_nodes, self.active_nodes_previous)
            self.active_nodes_previous = active_nodes

        # save .vtk file at specified rate
        if self.heat_solver.current_step % self.VtkOutputStep == 0:
            # Print time and completion status to terminal
            self.toc_jtr = time.perf_counter()
            self.elapsed_wall_time = self.toc_jtr - self.tic_start
            self.percent_complete = self.domain.current_sim_time/self.domain.end_sim_time
            self.time_remaining = (self.elapsed_wall_time/self.domain.current_sim_time)*(self.domain.end_sim_time - self.domain.current_sim_time)
            if self.verbose:
                print("Simulation time:  {:0.2} s, Percentage done:  {:0.3}%, Elapsed Time: {:0.3} s".format(
                    self.domain.current_sim_time, 100.*self.domain.current_sim_time/self.domain.end_sim_time, self.elapsed_wall_time))
                self.stats_append = np.expand_dims(np.array([self.elapsed_wall_time, self.domain.current_sim_time, self.percent_complete, self.time_remaining]), axis=0)
                with open('debug.csv', 'a') as exportfile:
                    np.savetxt(exportfile, self.stats_append, delimiter=',')
    
            # vtk file filename and save
            if self.outputVtkFiles:
                filename = os.path.join('vtk_files', self.geom_dir, self.laserpowerfile, 'u{:05d}.vtk'.format(self.VtkFileNum))
                self.save_vtk(filename)
                
            # iterate file number
            self.VtkFileNum = self.VtkFileNum + 1
            self.output_time = self.domain.current_sim_time


    def calc_geom_params(self):
        ''' Calculate surface distances. '''

        # Save the node birth times.
        self.zarr_stream.streamobj["ff_timestamp_node_deposition"].oindex[:] = np.expand_dims(self.domain.node_birth, 1)

        # Time loop
        self.tic_start = time.perf_counter()
        self.tic_jtr = self.tic_start
        while self.domain.current_sim_time < self.domain.end_sim_time - 1e-8 and self.heat_solver.current_step < self.max_itr :

            # Don't run the solver - instead, just move the laser
            self.heat_solver.update_field_no_integration()

            # Determine which files to save.
            saveZarr = self.heat_solver.current_step % self.ZarrOutputStep == 0
            saveVtk = self.heat_solver.current_step % self.VtkOutputStep == 0

            # Calculate distances.
            if saveZarr or saveVtk:
                # Find closest surfaces
                self.nodal_surf_distance = self.heat_solver.find_closest_surf_dist()
                # Free unused memory blocks
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()

                # Find laser distance
                self.nodal_laser_distance = self.heat_solver.find_laser_dist()

            # Save timestamped zarr file
            if saveZarr:
                # Save output file
                self.ZarrFileNum = self.ZarrFileNum + 1
                self.RecordAuxZarr()

            # save .vtk file if the current time is greater than an expected output time
            # offset time by dt/10 due to floating point error
            # honestly this whole thing should really be done with integers
            if saveVtk:
                # Print time and completion status to terminal
                self.toc_jtr = time.perf_counter()
                self.elapsed_wall_time = self.toc_jtr - self.tic_start
                self.percent_complete = self.domain.current_sim_time/self.domain.end_sim_time
                self.time_remaining = (self.elapsed_wall_time/self.domain.current_sim_time)*(self.domain.end_sim_time - self.domain.current_sim_time)
                if self.verbose:
                    print("Simulation time:  {:0.2} s, Percentage done:  {:0.3}%, Elapsed Time: {:0.3} s".format(
                        self.domain.current_sim_time, 100.*self.domain.current_sim_time/self.domain.end_sim_time, self.elapsed_wall_time))
                    self.stats_append = np.expand_dims(np.array([self.elapsed_wall_time, self.domain.current_sim_time, self.percent_complete, self.time_remaining]), axis=0)
                    with open('debug.csv', 'a') as exportfile:
                        np.savetxt(exportfile, self.stats_append, delimiter=',')
                
                # vtk file filename and save
                if self.outputVtkFiles:
                    filename = os.path.join('vtk_files', self.geom_dir, self.laserpowerfile+"_aux", 'u{:05d}.vtk'.format(self.VtkFileNum))
                    self.save_dist_vtk(filename)
                    
                # iterate file number
                self.VtkFileNum = self.VtkFileNum + 1
                self.output_time = self.domain.current_sim_time


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

    def RecordTempsZarr(self, active_nodes, active_nodes_prev, outputmode="structured"):
        '''Records a single data point to a zarr file'''

        timestep = np.expand_dims(np.expand_dims(self.domain.current_sim_time, axis=0), axis=1)
        pos_x = np.expand_dims(np.expand_dims(self.heat_solver.laser_loc[0].get(), axis=0), axis=1)
        pos_y = np.expand_dims(np.expand_dims(self.heat_solver.laser_loc[1].get(), axis=0), axis=1)
        pos_z = np.expand_dims(np.expand_dims(self.heat_solver.laser_loc[2].get(), axis=0), axis=1)
        laser_power = np.expand_dims(np.expand_dims(self.heat_solver.q_in, axis=0), axis=1)
        ff_temperature = np.expand_dims(self.heat_solver.temperature.get(), axis=0)
        active_elements = np.expand_dims(self.domain.active_elements.astype('i1'), axis=0)

        activated_nodes = np.where(active_nodes != active_nodes_prev)[0]

        if outputmode == "structured":
            # For each of the data streams, append the data for the current time step
            # expanding dimensions as needed to match
            self.zarr_stream.streamobj["timestamp"].append(timestep, axis=0)
            self.zarr_stream.streamobj["dt_pos_x"].append(pos_x, axis=0)
            self.zarr_stream.streamobj["dt_pos_y"].append(pos_y, axis=0)
            self.zarr_stream.streamobj["dt_pos_z"].append(pos_z, axis=0)
            self.zarr_stream.streamobj["dt_laser_power"].append(laser_power, axis=0)
            self.zarr_stream.streamobj["ff_dt_active_nodes"].append(np.expand_dims(active_nodes, axis=0), axis=0)
            self.zarr_stream.streamobj["ff_dt_temperature"].append(ff_temperature, axis=0)
            self.zarr_stream.streamobj["ff_dt_active_elements"].append(active_elements, axis=0)
            self.zarr_stream.streamobj["ff_laser_power_birth"].oindex[activated_nodes] = laser_power[0][0]

        elif outputmode == "bulked":
            new_row = np.zeros([1, (5+self.domain.nodes.shape[0])])
            new_row[0, 1] = timestep[0, 0]
            new_row[0, 2] = pos_x[0, 0]
            new_row[0, 3] = pos_y[0, 0]
            new_row[0, 4] = pos_z[0, 0]
            new_row[0, 5] = laser_power[0, 0]
            new_row[0, 6:(6+self.domain.nodes.shape[0])] = laser_power[0]
            #self.zarr_stream.streamobj["all_floats"].append(new_row, axis=0)
            #self.zarr_stream.streamobj["active_nodes"].append(active_nodes, axis=0)

        else:
            raise Exception("Error! Invalid zarr output type!")
    
    def RecordAuxZarr(self):
        '''Records distance information to zarr file'''
        timestep = np.expand_dims(self.domain.current_sim_time, axis=0)
        laser_dist = self.nodal_laser_distance
        surf_dist = self.nodal_surf_distance
        self.zarr_stream.streamobj["timestamp"][self.ZarrFileNum] = timestep
        self.zarr_stream.streamobj["ff_dt_dist_node_laser"][self.ZarrFileNum] = laser_dist
        self.zarr_stream.streamobj["ff_dt_dist_node_boundary"][self.ZarrFileNum] = surf_dist


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
    
    def save_dist_vtk(self, filename):
        active_elements = self.domain.elements[self.domain.active_elements].tolist()
        active_cells = np.array([item for sublist in active_elements for item in [8] + sublist])
        active_cell_type = np.array([vtk.VTK_HEXAHEDRON] * len(active_elements))
        points = self.domain.nodes.get()
        active_grid = pv.UnstructuredGrid(active_cells, active_cell_type, points)
        active_grid.point_data['surf_dist'] = self.nodal_surf_distance
        active_grid.point_data['laser_dist'] = self.nodal_laser_distance
        try:
            os.makedirs(os.path.dirname(filename))
            active_grid.save(filename)
        except:
            active_grid.save(filename)

class AuxDataRecorder():
    def __init__(self,
        nnodes,
        outputFolderPath
    ):
        
        # Location to save file
        self.outputFolderPath = outputFolderPath

        # Types of data being captured
        self.dataStreams = [
            "timestamp",
            "ff_dt_dist_node_laser",
            "ff_dt_dist_node_boundary",
            "ff_timestamp_node_deposition"
        ]

        # Dimension of one time-step of each data stream
        dims = [1, nnodes, nnodes, nnodes]
        # Type of each data stream
        types = ['f8', 'f8', 'f8', 'f8']

        self.dimsdict = {self.dataStreams[itr]:dims[itr] for itr in range(0, len(self.dataStreams))}
        self.typedict = {self.dataStreams[itr]:types[itr] for itr in range(0, len(self.dataStreams))}

        # dict containing the data streams themselves
        self.streamobj = dict.fromkeys(self.dataStreams)
        
        # Create zarr datasets for each data stream with length 1
        self.out_root = z.group(self.outputFolderPath)
        for stream in self.dataStreams:
            if stream == "ff_timestamp_node_deposition":
                self.streamobj[stream] = self.out_root.create_dataset(stream,
                                                                        shape=(self.dimsdict[stream], 1),
                                                                        dtype=self.typedict[stream],
                                                                        compressor=None,
                                                                        overwrite=True)
            else:
                self.streamobj[stream] = self.out_root.create_dataset(stream,
                                                                        shape=(1, self.dimsdict[stream]),
                                                                        chunks=(1, self.dimsdict[stream]),
                                                                        dtype=self.typedict[stream],
                                                                        compressor=None,
                                                                        overwrite=True)

class DataRecorder():
    def __init__(self,
        nnodes,
        nele,
        outputFolderPath,
        outputmode = "structured"
    ):
        
        # Location to save file
        if outputmode == "structured":
            self.outputFolderPath = outputFolderPath
            # Types of data being captured
            self.dataStreams = [
                "timestamp",
                "dt_pos_x",
                "dt_pos_y",
                "dt_pos_z",
                "dt_laser_power",
                "ff_dt_active_nodes",
                "ff_dt_temperature",
                "ff_dt_active_elements",
                "ff_laser_power_birth"
            ]

            # Dimension of one time-step of each data stream
            dims = [1, 1, 1, 1, 1, nnodes, nnodes, nele, nnodes]
            # Type of each data stream
            types = ['f8', 'f8', 'f8', 'f8', 'f8', 'i1', 'f8', 'i1', 'f8']

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
            if stream == "ff_laser_power_birth":
                self.streamobj[stream] = self.out_root.create_dataset(stream,
                                                                        shape=(self.dimsdict[stream], 1),
                                                                        dtype=self.typedict[stream],
                                                                        compressor=None,
                                                                        overwrite=True)
            else:
                self.streamobj[stream] = self.out_root.create_dataset(stream,
                                                                        shape=(1, self.dimsdict[stream]),
                                                                        chunks=(1, self.dimsdict[stream]),
                                                                        dtype=self.typedict[stream],
                                                                        compressor=None,
                                                                        overwrite=True)
        
        # Zarr datasets containing elements, node locations
        self.nodelocs = self.out_root.create_dataset("node_coords", shape=(nnodes, 3), dtype='f8', overwrite=True)
        self.ele = self.out_root.create_dataset("elements", shape=nele, dtype='i8', overwrite=True)

if __name__ == "__main__":
    with cp.cuda.Device(1).use():
        tic = time.perf_counter()
        timestep = 0.02 # seconds
        folder = os.path.dirname(os.path.abspath(__file__))
        model = FeaModel(f'{folder}/../../examples/data','thin_wall', 'LP_1', ZarrOutputStep=timestep, CalcNodeSurfDist=True, outputVtkFiles=True, timestep_override=timestep, VtkOutputStep=1.0)
        toc1 = time.perf_counter()
        model.calc_geom_params()
        toc2 = time.perf_counter()
        print(f"Time to Simulate: {toc2-toc1:0.4f}")
        print(f"Total time to run: {toc2-tic:0.4f}")

        simtime = toc2-toc1
        projectedtime = (model.domain.end_sim_time/12.854) * simtime

        print(f"End Time: {projectedtime:0.4f}")