import cupy as cp
import cupyx.scipy.sparse as cusparse

def elastic_stiff_matrix(elements, nodes, Bip_ele, shear, bulk):
    n_n = nodes.shape[0]
    n_e = elements.shape[0]
    n_p = elements.shape[1]
    n_q = 8
    n_int = n_e*n_q
    nodes_pos = nodes[elements]
    Jac = cp.matmul(Bip_ele,nodes_pos[:,cp.newaxis,:,:].repeat(8,axis=1)) # J = B*x [B:8(nGP)*3(dim)*8(nN), x:nE*8*8*3]
    ele_detJac = cp.linalg.det(Jac)
    iJac = cp.linalg.inv(Jac) #inv J (nE*nGp*dim*dim)
    ele_gradN = cp.matmul(iJac,Bip_ele) # dN/dx = inv(J)*B

    ele_B = cp.zeros([n_e,n_q,6,n_p*3])
    ele_B[:,:,0,0:24:3] = ele_gradN[:,:,0,:]
    ele_B[:,:,1,1:24:3] = ele_gradN[:,:,1,:]
    ele_B[:,:,2,2:24:3] = ele_gradN[:,:,2,:]
    ele_B[:,:,3,0:24:3] = ele_gradN[:,:,1,:]
    ele_B[:,:,3,1:24:3] = ele_gradN[:,:,0,:]
    ele_B[:,:,4,1:24:3] = ele_gradN[:,:,2,:]
    ele_B[:,:,4,2:24:3] = ele_gradN[:,:,1,:]
    ele_B[:,:,5,2:24:3] = ele_gradN[:,:,0,:]
    ele_B[:,:,5,0:24:3] = ele_gradN[:,:,2,:]

    temp = cp.array([[0,1,2]]).repeat(n_p,axis=0).flatten()
    jB = 3*cp.tile(elements[:,cp.newaxis,cp.newaxis,:],(1,n_q,6,1)).repeat(3,axis=3) + temp
    vB = ele_B.reshape(-1,n_p*3)
    jB = jB.reshape(-1,n_p*3)
    iB = cp.arange(0,jB.shape[0])[:,cp.newaxis].repeat(n_p*3,axis=1)
    B = cusparse.csr_matrix((cp.ndarray.flatten(vB),(cp.ndarray.flatten(iB), cp.ndarray.flatten(jB))), shape = (6*n_int, 3*n_n), dtype = cp.float_)

    IOTA = cp.array([[1],[1],[1],[0],[0],[0]]) 
    VOL = cp.matmul(IOTA,IOTA.transpose()) 
    DEV = cp.diag([1,1,1,1/2,1/2,1/2])-VOL/3

    ELASTC = 2*DEV*shear[:,:,cp.newaxis,cp.newaxis] + VOL*bulk[:,:,cp.newaxis,cp.newaxis]
    ele_D = ele_detJac[:,:,cp.newaxis,cp.newaxis]*ELASTC
    temp = cp.arange(0,n_e*n_q*6).reshape(n_e,n_q,6)
    iD = temp[:,:,cp.newaxis,:].repeat(6,axis = 2)
    jD = temp[:,:,:,cp.newaxis].repeat(6,axis = 3)

    D = cusparse.csr_matrix((cp.ndarray.flatten(ele_D),(cp.ndarray.flatten(iD), cp.ndarray.flatten(jD))), shape = (6*n_int, 6*n_int), dtype = cp.float_)
    ele_K =  ele_B.transpose([0,1,3,2])@ele_D@ele_B
    ele_K = ele_K.sum(axis = 1)

    K = B.transpose()*D*B 
    return K,B,D,ele_B,ele_D,iD,jD,ele_detJac

def constitutive_problem(E, Ep_prev, Hard_prev, shear, bulk, a, Y, T_anneal = None, T = None):
    
    # anneal temperature that sets previously accumulated plastic strain values to zero at any intpt with T > T_anneal
    if T_anneal and (T is not None):
        Ep_prev[T > T_anneal,:] = 0
        Hard_prev[T > T_anneal,:] = 0
        
    IOTA = cp.array([[1],[1],[1],[0],[0],[0]])  
    VOL = cp.matmul(IOTA,IOTA.transpose()) 
    DEV = cp.diag([1,1,1,1/2,1/2,1/2])-VOL/3
    E_tr = E-Ep_prev  
    ELASTC = 2*DEV*shear[:,:,cp.newaxis,cp.newaxis] + VOL*bulk[:,:,cp.newaxis,cp.newaxis]
    S_tr = (ELASTC @ E_tr[:,:,:,cp.newaxis]).squeeze()
    SD_tr = (2*DEV*shear[:,:,cp.newaxis,cp.newaxis]@E_tr[:,:,:,cp.newaxis]).squeeze() - Hard_prev
    norm_SD = cp.sqrt(cp.sum(SD_tr[:,:,0:3]*SD_tr[:,:,0:3], axis=2)+2*cp.sum(SD_tr[:,:,3:6]*SD_tr[:,:,3:6], axis=2))

    CRIT = norm_SD-Y
    IND_p = CRIT>0 

    S = cp.array(S_tr)
    DS = cp.ones((S.shape[0],S.shape[1],6,6))*ELASTC

    if not IND_p[IND_p].shape[0]:
        Ep = cp.array(Ep_prev)
        Hard = cp.array(Hard_prev)
        return S, DS, IND_p, Ep, Hard   

    N_hat = SD_tr[IND_p]/norm_SD[IND_p][:,cp.newaxis].repeat(6,axis=1)  
    denom =  2*shear[IND_p]+ a[IND_p] 
    Lambda = CRIT[IND_p]/denom

    S[IND_p] = S[IND_p] - 2*N_hat*(shear[IND_p]*Lambda)[:,cp.newaxis].repeat(6,axis=1)  
    NN_hat = N_hat[:,:,cp.newaxis]@N_hat[:,cp.newaxis,:]
    const = 4*shear[IND_p]**2/denom

    DS[IND_p] = DS[IND_p] - const[:,cp.newaxis,cp.newaxis]*DEV + (const*Y[IND_p]/norm_SD[IND_p])[:,cp.newaxis,cp.newaxis].repeat(6,axis=1).repeat(6,axis=2)*(DEV-NN_hat)


    Ep = cp.array(Ep_prev)
    Ep[IND_p] = Ep[IND_p]+cp.matmul(cp.array([[1],[1],[1],[2],[2],[2]]),Lambda[cp.newaxis]).transpose()*N_hat

    Hard = cp.array(Hard_prev)
    Hard[IND_p] = Hard[IND_p]+(a[IND_p]*Lambda)[:,cp.newaxis].repeat(6,axis=1)*N_hat
    
    return S, DS, IND_p, Ep, Hard

def transformation(Q_int, active_elements, ele_detJac,n_n_save):
    Q_int = Q_int.reshape(1,-1)
    elem = cp.array(active_elements.transpose())                      # elements.transpose() with shape (n_p=8,n_e)
    weight = ele_detJac.reshape(1,-1)
    #n_n = COORD.shape[1]          # number of nodes including midpoints
    n_e = elem.shape[1]            # number of elements
    n_p = 8                        # number of vertices per element
    n_q = 8                        # number of quadrature points
    n_int = n_e*n_q                # total number of integrations points
    # values at integration points, shape(vF1)=shape(vF2)=(n_p,n_int)   
    vF1 = cp.matmul(cp.ones((n_p,1)), weight*Q_int)    
    vF2 = cp.matmul(cp.ones((n_p,1)),weight)

    # row and column indices, shape(iF)=shape(jF)=(n_p,n_int)   
    iF = cp.zeros((n_p,n_int), dtype=cp.int32)         ######
    jF = cp.kron(elem, cp.ones((1,n_q), dtype=cp.int32))

    # the asssembling by using the sparse command - values v for duplicate
    # doubles i,j are automatically added together
    F1 = cusparse.csr_matrix((cp.ndarray.flatten(vF1.transpose()),(cp.ndarray.flatten(iF.transpose()), cp.ndarray.flatten(jF.transpose()))), dtype = cp.float_) 
    F2 = cusparse.csr_matrix((cp.ndarray.flatten(vF2.transpose()),(cp.ndarray.flatten(iF.transpose()), cp.ndarray.flatten(jF.transpose()))), dtype = cp.float_) 

    #
    # Approximated values of the function Q at nodes of the FE mesh
    #
    Q = cp.array(F1/F2)
    Q_node = cp.ones(Q.shape[1])
    Q_node[0:n_n_save] = Q[0,0:n_n_save]
    return Q_node

def disp_match(nodes, U, n_n_old, n_n):
    idar = cp.arange(nodes.shape[0])
    U1 = U
    zel_prev = nodes[0:n_n_old,2].max()
    for k in range(n_n_old, n_n):
        U1[k,:] = U[int(idar[(nodes[:,0] == nodes[k,0]) * (nodes[:,1] == nodes[k,1]) * (nodes[:,2] == zel_prev)]),:]
    return U1