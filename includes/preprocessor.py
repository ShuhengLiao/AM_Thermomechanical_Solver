from numba import jit,vectorize,guvectorize,cuda
import numpy as np
import pandas as pd
import pyvista as pv
from pyvirtualdisplay import Display
import vtk

def write_keywords(file_name,output_file,height):
    with open(file_name,'r') as input_file:
        lines = input_file.readlines()

        #find the start line of node
        for num in range(0,len(lines)):
            if "*Node" in lines[num]:
                break
        node_num = num + 1

        #find the start line of element
        for num in range(node_num,len(lines)):
            if "*Element" in lines[num]:
                 break
        element_num = num + 1

        #find the end line of element
        for num in range(element_num,len(lines)):
            if "*" in lines[num]:
                 break
        end_num = num

        #collect all the node
        node = [lines[i][:-1].split(',') for i in range(node_num,element_num-1)]
        for i in range(len(node)):
            for j in range(len(node[0])):
                node[i][j] = float(node[i][j])
        node = np.asarray(node)

        #collect all the element
        element = [lines[i][:-1].split(',') for i in range(element_num,end_num)]
        for i in range(len(element)):
            for j in range(len(element[0])):
                element[i][j] = float(element[i][j])
        element = np.asarray(element)

    #part ID, if any of the node is higher than the substrate then define as 1
    part_id = 2*np.ones([len(element),1])
    for e in range(0,len(element)):
        for id in element[e][1:9]:
            if node[int(id-1)][3]>height:
                part_id[e] = 1
    
    #node set 1, all nodes
    node_set1 = node[:,0]

    #node set 2, z>=0
    node_set2 = []
    for i in range(len(node)):
        if node[i,3]>=height:
            node_set2.append(int(node[i,0]))
    node_set2 = np.asarray(node_set2)

    #node set 3, z==-height
    node_set3 = []
    for i in range(len(node)):
        if node[i,3]==0:
            node_set3.append(int(node[i,0]))
    node_set3 = np.asarray(node_set3)

    f = open(output_file,'w')
    #write node information
    f.write('*NODE\n')
    f.write('$#   nid               x               y               z      tc      rc\n')
    for n in node:
        f.write("%8d"%n[0])
        f.write("%16f"%n[1])
        f.write("%16f"%n[2])
        f.write("%16f"%(n[3]-height))
        f.write('       0       0\n')
          
    #write element information
    f.write('*ELEMENT_SOLID\n')
    f.write('$#   eid     pid      n1      n2      n3      n4      n5      n6      n7      n8\n') 
    for e,p in zip(element,part_id):
        f.write("%8d"%e[0])
        f.write("%8d"%p)
        for i in range (1,9):
            f.write("%8d"%e[i])
        f.write('\n')
    
    #write node set1
    f.write('*SET_NODE_LIST\n')
    f.write('$#     sid       da1       da2       da3       da4    solver\n')
    f.write('         1\n')
    f.write('$#    nid1      nid2      nid3      nid4      nid5      nid6      nid7      nid8\n')
    for i in range(0,len(node_set1)):
        f.write("%10d"%node_set1[i])
        if (i+1)%8 == 0:
            f.write('\n')
    f.write('\n')
    
    #write node set2        
    f.write('*SET_NODE_LIST\n')
    f.write('$#     sid       da1       da2       da3       da4    solver\n')
    f.write('         2\n')
    f.write('$#    nid1      nid2      nid3      nid4      nid5      nid6      nid7      nid8\n')
    for i in range(0,len(node_set2)):
        f.write("%10d"%node_set2[i])
        if (i+1)%8 == 0:
            f.write('\n')
    f.write('\n')
    
    #write node set3       
    f.write('*SET_NODE_LIST\n')
    f.write('$#     sid       da1       da2       da3       da4    solver\n')
    f.write('         3\n')
    f.write('$#    nid1      nid2      nid3      nid4      nid5      nid6      nid7      nid8\n')
    for i in range(0,len(node_set3)):
        f.write("%10d"%node_set3[i])
        if (i+1)%8 == 0:
            f.write('\n')
    f.write('\n')
    
    
    #write solid set1
    f.write('*SET_SOLID\n')
    f.write('$#     sid    solver\n')
    f.write('         1MECH\n')
    f.write('$#      k1        k2        k3        k4        k5        k6        k7        k8\n')
    e_num = 0
    for i in range(0,len(part_id)):
        if part_id[i] == 1:
            f.write("%10d"%(i+1))
            e_num = e_num + 1
            if (e_num)%8 == 0:
                f.write('\n')
    f.write('\n')
    
    #write solid set2
    f.write('*SET_SOLID\n')
    f.write('$#     sid    solver\n')
    f.write('         2MECH\n')
    f.write('$#      k1        k2        k3        k4        k5        k6        k7        k8\n')
    e_num = 0
    for i in range(0,len(part_id)):
        if part_id[i] == 2:
            f.write("%10d"%(i+1))
            e_num = e_num + 1
            if (e_num)%8 == 0:
                f.write('\n')
    f.write('\n')
    f.write('*END')
    f.close()
    
def load_mesh_file(filename):
    nodes = []
    node_sets = {}
    elements = []

    with open(filename) as f:
        while True:
            line = next(f)
            if not line.split():
                continue
            if line.split()[0] == '*NODE':
                first = True
                while True:
                    line = next(f)
                    if line[0] == '*':
                        break
                    if line[0] == '$':
                        continue
                    text = line.split()
                    if first:
                        node_base = int(text[0])
                        first = False
                    nodes.append([float(text[1]),float(text[2]),float(text[3])])
            if line.split()[0] == '*END':
                break
    
    
    with open(filename) as f:
        line = next(f)
        while True:
            if not line.split():
                line = next(f)
                continue
            elif line.split()[0] == '*SET_NODE_LIST':
                line = next(f)
                line = next(f)
                key = int(line.split()[0])
                node_list = []
                while True:
                    line = next(f)
                    if line[0] == '*':
                        break
                    if line[0] == '$':
                        continue
                    for text in line.split():
                        node_list.append(int(text)-node_base)
                node_sets[key] = node_list
            elif line.split()[0] == '*END':
                break
            else:
                line = next(f)
                
                
    with open(filename) as f:
        while True:
            line = next(f)
            if not line.split():
                continue
            if line.split()[0] == '*ELEMENT_SOLID':
                first = True
                while True:
                    line = next(f)
                    if line[0] == '*':
                        break
                    if line[0] == '$':
                        continue
                    text = line.split()
                    if first:
                        element_base = int(text[0])
                        first = False
                    elements.append([int(text[2])-node_base, int(text[3])-node_base, int(text[4])-node_base, int(text[5])-node_base,
                                     int(text[6])-node_base, int(text[7])-node_base, int(text[8])-node_base, int(text[9])-node_base])
            if line.split()[0] == '*END':
                break
                
    return np.array(nodes),np.array(elements)

def load_toolpath(filename):
    toolpath_raw=pd.read_table(filename, delimiter=r"\s+",header=None, names=['time','x','y','z','state'])
    return toolpath_raw.to_numpy()


@jit(nopython=True)
def assign_birth_time(ele_nodes,ele_ctrl,ele_topz,toolpath,element_birth,radius,path_resolution,mode):
    for i in range(1,toolpath.shape[0]):
        if toolpath[i,4] == 0:
            continue
        direction = toolpath[i,1:4]-toolpath[i-1,1:4]
        d = np.linalg.norm(direction)
        dir_norm = direction/d
        num = round(d/path_resolution)
        t = np.linspace(toolpath[i-1,0],toolpath[i,0],num+1)
        X = np.interp(t,[toolpath[i-1,0],toolpath[i,0]],[toolpath[i-1,1],toolpath[i,1]])
        Y = np.interp(t,[toolpath[i-1,0],toolpath[i,0]],[toolpath[i-1,2],toolpath[i,2]])
        if mode == 0:
            for j in range (0,num):
                for k in range(0,ele_nodes.shape[0]):
                    if element_birth[k]==-1 and ele_topz[k]<=toolpath[i,3]+1e-5:
                        distance = (ele_ctrl[k,0]-X[j])**2 + (ele_ctrl[k,1]-Y[j])**2
                        if distance < radius**2+1e-5:
                            element_birth[k] = t[j]
        # flat head                    
        if mode == 1:
            for j in range (0,num):
                for k in range(0,ele_nodes.shape[0]):
                    if element_birth[k]==-1 and ele_topz[k]<=toolpath[i,3]+1e-5:
                        distance = (ele_ctrl[k,0]-X[j])**2 + (ele_ctrl[k,1]-Y[j])**2
                        distance1 = abs((ele_ctrl[k,0]-X[j])*dir_norm[0] + (ele_ctrl[k,1]-Y[j])*dir_norm[1])
                        distance2 = distance - distance1
                        if distance1 < radius**2+1e-5 and distance2 < radius**2+1e-5:
                            element_birth[k] = t[j]
        # no birth
        if mode == 2:
            for k in range(0,ele_nodes.shape[0]):
                element_birth[k] = 0

                        
def write_birth(output_file,toolpath_file,path_resolution,radius,
                gif_start=0,gif_end=-1,nFrame=200,mode=0,camera_position=[(0, -100, 180),(0, 0, 0),(0.0, 0.0, 1.0)]):
    nodes,elements = load_mesh_file(output_file)
    ele_nodes = nodes[elements] 
    ele_ctrl = ele_nodes.sum(axis=1)/8
    ele_topz = ele_nodes[:,:,2].max(axis=1)
    toolpath = load_toolpath(toolpath_file)
    element_birth = -1*np.ones(elements.shape[0])
    element_birth[ele_topz<=0] = 0
    assign_birth_time(ele_nodes,ele_ctrl,ele_topz,toolpath,element_birth,radius,path_resolution,mode)
    # save gif
    if gif_end == -1:
        gif_end = toolpath[-1,0]
    time = np.linspace(gif_start,gif_end,nFrame)
    x = np.interp(time,toolpath[:,0],toolpath[:,1])
    y = np.interp(time,toolpath[:,0],toolpath[:,2])
    z = np.interp(time,toolpath[:,0],toolpath[:,3])
    toolpath_interp = np.array([time,x,y,z]).transpose()
    
    display = Display(visible=0)
    _ = display.start()
    p = pv.Plotter(window_size=(1000,800))
    #p.show(auto_close=False)
    # Open a gif
    p.open_gif("birth.gif")
    for step in range(0,nFrame):
        p.clear()
        t = toolpath_interp[step,0]
        laser = toolpath_interp[step,1:4]
        active_elements = [element.tolist() for element, birth_time in zip(elements, element_birth) if (birth_time <= t)]
        cells = np.array([item for sublist in active_elements for item in [8] + sublist])
        cell_type = np.array([vtk.VTK_HEXAHEDRON] * len(active_elements))
        points = np.array(nodes)
        grid = pv.UnstructuredGrid(cells, cell_type, points)
        p.camera_position = camera_position
        p.add_mesh(grid,show_edges=True, color='#A9DFBF',edge_color='#1B2631',lighting=True)
        p.add_points(laser,point_size = 10,color='red')
        p.add_axes()
        p.write_frame()  # this will trigger the render

    p.close()
    
    
    element_birth = np.concatenate((np.arange(0,elements.shape[0])[:,np.newaxis],element_birth[:,np.newaxis]),axis=1)
    element_birth = element_birth[element_birth[:,1].argsort()]
    f = open(output_file,'r+')
    old = f.read()
    f.seek(0)
    #write node information       
    f.write('*DEFINE_CURVE\n')
    f.write('         1                 1.0       1.0\n')
    for e_b in element_birth:
        if e_b[1] > 0:  
            f.write("%20.8f"%e_b[1])
            f.write("%20.8f\n"%(e_b[0]+1))
        if e_b[1] < 0:  
            f.write("%20.8f"%100000)
            f.write("%20.8f\n"%(e_b[0]+1))
    f.write(old)
    
    
def write_parameters(output_file):
    f = open(output_file,'r+')
    old = f.read()
    f.seek(0)
    text = '''*KEYWORD_ID
DED
*PARAMETER
Rboltz    5.6704E-14
*PARAMETER
Rambient       300.0
*PARAMETER
Rabszero         0.0
*TOOL_FILE
file_name
*GAUSS_LASER
laser_power radius effieciency
*SCALAR_OUT
temp
solid_rate
theta_hist
nid_true
*CONTROL_TERMINATION
$$  ENDTIM    ENDCYC     DTMIN    ENDENG    ENDMAS
   time
*CONTROL_TIMESTEP
$$  DTINIT    TSSFAC      ISDO    TSLIMT     DT2MS      LCTM     EROD       E     MSIST
    1.0E-2       1.0
*CONTROL_SOLUTION
$$    SOLN
         0
*DATABASE_NODOUT
$$      DT    BINARY      LCUR      IOPT      DTHF     BINHF
    10.000         0
*MAT_THERMAL_ISOTROPIC
$HMNAME MATS       1MATT1_1
         1   density   solidus   liqudius   latent_heat
    cp    cond
*MAT_THERMAL_ISOTROPIC
$HMNAME MATS       2MATT1_2
         2   density   solidus   liqudius   latent_heat
    cp    cond
*PART
$HWCOLOR COMPS       1       3
Substrate
         1         0         1
$HWCOLOR COMPS       2       4
Build
         2         0         2
*LOAD_NODE_SET
$HMNAME LOADCOLS       1LoadNode_1
         3         1               300.0
$	 moving flux
         2         5                 0.0
$	 Radiation
         2         4                 0.2
$	 convection
         2         3             0.00005
*INITIAL_TEMPERATURE_SET
$HMNAME LOADCOLS       1InitialTemp_1
$HWCOLOR LOADCOLS       1       3
         1     300.0
'''
    f.write(text)
    f.write(old)