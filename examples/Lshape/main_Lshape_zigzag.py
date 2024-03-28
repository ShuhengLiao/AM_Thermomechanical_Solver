import sys
import cupy as cp
import numpy as np
import cupyx.scipy.sparse as cusparse
from gamma.simulator.gamma import domain_mgr, heat_solve_mgr
from gamma.simulator.func import elastic_stiff_matrix,constitutive_problem,transformation,disp_match
cp.cuda.Device(1).use()
import pyvista as pv
import vtk



def save_vtk(filename):
    n_e_save = cp.sum(domain.active_elements)
    n_n_save = cp.sum(domain.active_nodes)
    active_elements = domain.elements[domain.active_elements].tolist()
    active_cells = np.array([item for sublist in active_elements for item in [8] + sublist])
    active_cell_type = np.array([vtk.VTK_HEXAHEDRON] * len(active_elements))
    points = domain.nodes[0:n_n_save].get() + 5*U[0:n_n_save].get()
    Sv =  transformation(cp.sqrt(1/2*((S[0:n_e_save,:,0]-S[0:n_e_save,:,1])**2 + (S[0:n_e_save,:,1]-S[0:n_e_save,:,2])**2 + (S[0:n_e_save,:,2]-S[0:n_e_save,:,0])**2 + 6*(S[0:n_e_save,:,3]**2+S[0:n_e_save,:,4]**2+S[0:n_e_save,:,5]**2))),domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    S11 = transformation(S[0:n_e_save,:,0], domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    S22 = transformation(S[0:n_e_save,:,1], domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    S33 = transformation(S[0:n_e_save,:,2], domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    S12 = transformation(S[0:n_e_save,:,3], domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    S23 = transformation(S[0:n_e_save,:,4], domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    S13 = transformation(S[0:n_e_save,:,5], domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    active_grid = pv.UnstructuredGrid(active_cells, active_cell_type, points)
    active_grid.point_data['temp'] = heat_solver.temperature[0:n_n_save].get()
    active_grid.point_data['S_von'] = Sv.get()
    active_grid.point_data['S11'] = S11.get()
    active_grid.point_data['S22'] = S22.get()
    active_grid.point_data['S33'] = S33.get()
    active_grid.point_data['S12'] = S12.get()
    active_grid.point_data['S23'] = S23.get()
    active_grid.point_data['S13'] = S13.get()
    active_grid.point_data['U1'] = U[0:n_n_save,0].get()
    active_grid.point_data['U2'] = U[0:n_n_save,1].get()
    active_grid.point_data['U3'] = U[0:n_n_save,2].get()
    active_grid.save(filename)
    
    
domain = domain_mgr(filename='Lshape_zigzag.k')
#domain = domain_mgr(filename='Lshape_bound.k')
heat_solver = heat_solve_mgr(domain)
endtime = domain.end_time
n_n = len(domain.nodes)
n_e = len(domain.elements)
n_p = 8
n_q = 8
n_int = n_e * n_q
file_num = 0

# values of elastic material parameters
poisson =  0.3                        # Poisson's ratio
a1 = 10000
young1 = cp.array(np.loadtxt('../0_properties/TI64_Young_Debroy.txt')[:,1]/1e6)
temp_young1 = cp.array(np.loadtxt('../0_properties/TI64_Young_Debroy.txt')[:,0])
Y1 = cp.array(np.loadtxt('../0_properties/TI64_Yield_Debroy.txt')[:,1]/1e6*np.sqrt(2/3))
temp_Y1 = cp.array(np.loadtxt('../0_properties/TI64_Yield_Debroy.txt')[:,0])
scl1 = cp.array(np.loadtxt('../0_properties/TI64_Alpha_Debroy.txt')[:,1])
temp_scl1 = cp.array(np.loadtxt('../0_properties/TI64_Alpha_Debroy.txt')[:,0])
T_Ref = domain.ambient

# Initialization for the whole boundary-value problem
E = cp.zeros((n_e,n_q,6))                        # strain tensors at integration points
S = cp.zeros((n_e,n_q,6))
Ep_prev = cp.zeros((n_e,n_q,6))                  # plastic strain tensors at integration points
Hard_prev = cp.zeros((n_e,n_q,6))
U = cp.zeros((n_n,3))
dU = cp.zeros((n_n,3))
F = cp.zeros((n_n,3))
f = cp.zeros((n_n,3))
alpha_Th = cp.zeros((n_e,n_q,6))
idirich = cp.array(domain.nodes[:, 2] == -15.0 ) 
n_e_old = cp.sum(domain.element_birth < 1e-5)
n_n_old = cp.sum(domain.node_birth < 1e-5)

nodes_pos = domain.nodes[domain.elements]
Jac = cp.matmul(domain.Bip_ele,nodes_pos[:,cp.newaxis,:,:].repeat(8,axis=1)) # J = B*x [B:8(nGP)*3(dim)*8(nN), x:nE*8*8*3]
ele_detJac = cp.linalg.det(Jac)

# Tolerence for Newton stopping criterion
tol = 1.0e-8                           # non-dimensionalized tolerence 
# Maximum Number of N_R Iterations allowed
Maxit = 20

t = 0
last_mech_time = 0
output_timestep = 10
#filename = 'results_bound/bound_{}.vtk'.format(file_num)
filename = 'results_zigzag/zigzag_{}.vtk'.format(file_num)
save_vtk(filename)
file_num = file_num + 1

while domain.current_time<endtime-domain.dt:
    t = t+1
    heat_solver.time_integration()
    if t % 5000 == 0:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        print("Current time:  {}, Percentage done:  {}%".format(domain.current_time,100*domain.current_time/domain.end_time))  
        heat_solver.time_integration()
            
    n_e_active = cp.sum(domain.element_birth < domain.current_time)
    n_n_active = cp.sum(domain.node_birth < domain.current_time) 
    
    if heat_solver.laser_state == 0 and n_e_active == n_e_old:
        implicit_timestep = 0.1
    else:
        implicit_timestep = 0.05
        
    if domain.current_time >= last_mech_time + implicit_timestep:
        

        active_eles = domain.elements[0:n_e_active]
        active_nodes = domain.nodes[0:n_n_active]
        
        if n_n_active>n_n_old:
            if domain.nodes[n_n_old:n_n_active,2].max()>domain.nodes[0:n_n_old,2].max():
                U = disp_match(domain.nodes, U, n_n_old, n_n)
        
        
        temperature_nodes = heat_solver.temperature[domain.elements]
        temperature_ip = (domain.Nip_ele[:,cp.newaxis,:]@temperature_nodes[:,cp.newaxis,:,cp.newaxis].repeat(8,axis=1))[:,:,0,0]
        temperature_ip = cp.clip(temperature_ip,300,2300)
        
        Q = cp.zeros(domain.nodes.shape, dtype=bool)
        Q[0:n_n_active,:] = 1 
        Q[idirich,:] = 0
        
        young = cp.interp(temperature_ip,temp_young1,young1)
        shear = young/(2*(1+poisson))        # shear modulus
        bulk = young/(3*(1-2*poisson))       # bulk modulus
        scl = cp.interp(temperature_ip,temp_scl1,scl1)
        a  = a1*cp.ones_like(young)
        alpha_Th[:,:,0:3] = scl[:,:,cp.newaxis].repeat(3,axis=2)
        Y = cp.interp(temperature_ip,temp_Y1,Y1)
        
        K_elast,B,D_elast,ele_B,ele_D,iD,jD,ele_detJac = elastic_stiff_matrix(active_eles,active_nodes,domain.Bip_ele, shear[0:n_e_active], bulk[0:n_e_active])
    
        for beta in [1.0,0.5,0.3,0.1]:
            U_it = U[0:n_n_active]
            for it in range(0,Maxit):
                E[0:n_e_active] = cp.reshape(B@U_it.flatten(),(-1,8,6))
                E[0:n_e_active] = E[0:n_e_active] - (temperature_ip[0:n_e_active,:,cp.newaxis].repeat(6,axis=2) - T_Ref) *alpha_Th[0:n_e_active]

                S, DS, IND_p,_,_ = constitutive_problem(E[0:n_e_active], Ep_prev[0:n_e_active], Hard_prev[0:n_e_active], shear[0:n_e_active], bulk[0:n_e_active], a[0:n_e_active], Y[0:n_e_active])
                vD = ele_detJac[:,:,cp.newaxis,cp.newaxis].repeat(6,axis=2).repeat(6,axis=3) * DS
                D_p = cusparse.csr_matrix((cp.ndarray.flatten(vD), (cp.ndarray.flatten(iD),cp.ndarray.flatten(jD))), shape = D_elast.shape, dtype = cp.float_)
                K_tangent = K_elast + B.transpose()*(D_p-D_elast)*B
                #K_tangent = B.transpose()*(D_p)*B
                n_plast = len(IND_p[IND_p])
                print(' plastic integration points: ', n_plast, ' of ', IND_p.shape[0]*IND_p.shape[1])
                F = B.transpose() @ ((ele_detJac[:,:,cp.newaxis].repeat(6,axis=2)*S).reshape(-1))
                dU[Q],error = cusparse.linalg.cg(K_tangent[Q[0:n_n_active].flatten()][:,Q[0:n_n_active].flatten()],-F[Q[0:n_n_active].flatten()],tol=tol)
                U_new = U_it + beta*dU[0:n_n_active,:] 
                q1 = beta**2*dU[0:n_n_active].flatten()@K_elast@dU[0:n_n_active].flatten()
                q2 = U_it[0:n_n_active].flatten()@K_elast@U_it[0:n_n_active].flatten()
                q3 = U_new[0:n_n_active].flatten()@K_elast@U_new[0:n_n_active].flatten()
                if q2 == 0 and q3 == 0:
                    criterion = 0
                else:
                    criterion = q1/(q2+q3)
                    print('  stopping criterion=  ', criterion)

                U_it = cp.array(U_new) 
                # test on the stopping criterion
                if  criterion < tol:
                    print('F = ', cp.linalg.norm(F[Q[0:n_n_active].flatten()]))
                    break
            else:
                continue
            break
        else:
            raise Exception('The Newton solver does not converge for the current timestep: {}'.format(t))
      
        U[0:n_n_active] = U_it        
        E[0:n_e_active] = cp.reshape(B@U_it.flatten(),(-1,8,6))
        E[0:n_e_active] = E[0:n_e_active] - (temperature_ip[0:n_e_active,:,cp.newaxis].repeat(6,axis=2)-T_Ref)*alpha_Th[0:n_e_active]
          
        S, DS, IND_p,Ep,Hard = constitutive_problem(E[0:n_e_active], Ep_prev[0:n_e_active], Hard_prev[0:n_e_active], shear[0:n_e_active], bulk[0:n_e_active], a[0:n_e_active], Y[0:n_e_active])
        Ep_prev[0:n_e_active] = Ep
        Hard_prev[0:n_e_active] = Hard
        n_e_old = n_e_active
        n_n_old = n_n_active
        last_mech_time = domain.current_time
        if domain.current_time >= file_num*(output_timestep):
            filename = 'results_zigzag/zigzag_{}.vtk'.format(file_num)
            #filename = 'results_bound/bound_{}.vtk'.format(file_num)
            save_vtk(filename)
            file_num = file_num + 1