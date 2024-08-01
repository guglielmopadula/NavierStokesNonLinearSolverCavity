import jax
import numpy as np
import scipy.sparse
from jax.experimental import sparse
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import svds
from skfem import *
from skfem.helpers import ddot, dot, grad
from skfem.models.general import divergence, rot
from skfem.models.poisson import laplace, mass, vector_laplace
from skfem.utils import _flatten_dofs
from skfem.visuals.matplotlib import plot
from sparse import COO
from tqdm import trange

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from jax.experimental.sparse import BCOO

mesh = MeshQuad().refined(4)


element = {'u': ElementVector(ElementQuad2()),
           'p': ElementQuad1()}
basis = {variable: Basis(mesh, e, intorder=3)
         for variable, e in element.items()}



def up(x):
    """return the plane Poiseuille parabolic inlet profile"""
    return np.stack([np.ones_like(x[0]), np.zeros_like(x[0])])

def wall(x):
    """return the plane Poiseuille parabolic inlet profile"""
    return np.stack([np.zeros_like(x[0]), np.zeros_like(x[0])])


up_dofs_u = basis['u'].get_dofs(['top'])
up_dofs_p = basis['p'].get_dofs(['top'])
up_basis = FacetBasis(mesh, element['u'], facets=mesh.boundaries['top'])
up_basis=up_basis.project(up)

wall_dofs_u = basis['u'].get_dofs(['right','left','bottom'])
wall_dofs_p = basis['p'].get_dofs(['right','left','bottom'])
wall_basis = FacetBasis(mesh, element['u'], facets=np.concatenate((mesh.boundaries['bottom'],mesh.boundaries['right'],mesh.boundaries['left']),dtype=np.int32))
wall_basis=wall_basis.project(wall)


@BilinearForm
def mass(u,v,w):
    return dot(u,v)



def matmul(u,v):
    v=np.expand_dims(v,axis=1)    
    return np.sum(u*v,axis=1)

@TrilinearForm
def c(u, v, w, _):
    return dot(matmul(grad(u), v),w)

A = asm(vector_laplace, basis['u'])
B = asm(divergence, basis['u'], basis['p'])
C=asm(c, basis['u'])


B=B

A,b = enforce(A,b=np.zeros(A.shape[0]),x=up_basis,D=up_dofs_u)


C=COO(C.indices,C.data,shape=C.shape)


up_dofs_u=_flatten_dofs(up_dofs_u)

for elem in up_dofs_u:
    l=np.where(C.coords[2] == elem)[0]
    C.data[l]=0
    B[:,elem]=0



A = enforce(A, x=wall_basis,D=wall_dofs_u)[0]

wall_dofs_u=_flatten_dofs(wall_dofs_u)

for elem in wall_dofs_u:
    l=np.where(C.coords[2] == elem)[0]
    C.data[l]=0
    B[:,elem]=0



B=B.T
C=C.transpose((2,1,0))

C_T=C.transpose((0,2,1))




Bt=B.T

N=A.shape[1]+B.shape[1]

arr=np.load("stokes.npy")

mu_vec=np.linspace(100,1000,600)

'''

mu=1

arr=np.load("stokes.npy")
#print(left_term(np.array(arr)).shape)


def condition_number(x,mu):
    u=x[:2178]
    p=x[2178:]
    matrix=scipy.sparse.bmat([[1/mu*(C.dot(u)+C_T.dot(u))+A, B],[Bt,csr_matrix((B.shape[1],B.shape[1]))]])
    matrix=matrix.todense()
    matrix=matrix@matrix.T
    return np.linalg.svd(matrix,compute_uv=False)[-1]

#print(arr.shape)
#print(left_term(arr).shape)
#print(left_term_jax(jnp.array(arr)))






arr=np.load("stokes.npy")

print(condition_number(arr,10000))

mu_vec=np.linspace(100,1000,600)



all_data=np.zeros((600,arr.shape[0]))
all_data=np.load("all_data.npy")


for i in trange(600):
    def compute_loss(x):
        u=x[:2178]
        p=x[2178:]
        return np.linalg.norm((1/mu_vec[i])*C.dot(u).dot(u)+A.dot(u)-B.dot(p)-b)**2+np.linalg.norm(Bt.dot(u))**2

    def jac(x):
        u=x[:2178]
        p=x[2178:]
        l=1/mu_vec[i]*(C.dot(u).dot(u))+A.dot(u)-B.dot(p)-b
        tmp=(C.dot(u)+C_T.dot(u))*(1/mu_vec[i])+A
        bb=np.asarray((B.multiply(B)).sum(axis=1)).reshape(-1)
        grad_u=np.asarray(np.sum(l*tmp,axis=0)).reshape(-1)+2*u*bb
        l=csr_matrix(l).transpose()
        grad_p=np.asarray(np.sum(B.multiply(l),axis=0)).reshape(-1)
        return np.concatenate((grad_u,grad_p))
        

    res=minimize(compute_loss,arr,options={'disp': True, 'maxiter': 100}, jac=jac, method='L-BFGS-B')
    all_data[i]=res.x

np.save("all_data.npy",all_data)

def compute_loss(mu,x):
    u=x[:2178]
    p=x[2178:]
    return np.linalg.norm((1/mu)*C.dot(u).dot(u)+A.dot(u)-B.dot(p)-b)**2+np.linalg.norm(Bt.dot(u))**2






data=np.load("all_data.npy")

loss=0


for i in trange(600):
    loss=loss+1/(600)*compute_loss(mu_vec[i],data[i])
print(loss)


'''




A=A.tocoo()
B=B.tocoo()
A_data=A.data
B_data=B.data
C_data=C.data
A_coords=np.concatenate((A.row.reshape(-1,1),A.col.reshape(-1,1)),axis=1)
B_coords=np.concatenate((B.row.reshape(-1,1),B.col.reshape(-1,1)),axis=1)
C_coords=C.coords
C_coords=C_coords.T
A_jax=sparse.BCOO((jnp.array(A_data),jnp.array(A_coords)),shape=A.shape)
B_jax=sparse.BCOO((jnp.array(B_data),jnp.array(B_coords)),shape=B.shape)
C_jax=sparse.BCOO((jnp.array(C_data),jnp.array(C_coords)),shape=C.shape)
Bt_jax=B_jax.T
b_jax=jnp.array(b)


def init_params(layers):
    params=jnp.ones(0)
    l=[]
    l.append(len(params))
    keys_means = jax.random.split(jax.random.PRNGKey(0),len(layers))
    for keys_mean,shapes in zip(keys_means,layers):
        n_in=shapes[0]
        n_out=shapes[1]
        W_fwd_mean=jax.random.normal(keys_mean,shape=(n_out,n_in))*jnp.sqrt(2/(n_out*n_in))
        B_fwd_mean=jnp.zeros((n_out))
        W_fwd_mean=W_fwd_mean.reshape(n_out*n_in)
        params=np.concatenate((params,W_fwd_mean))
        l.append(len(params))
        B_fwd_mean=B_fwd_mean.reshape(-1)
        params=np.concatenate((params,B_fwd_mean))
        l.append(len(params))
    return params,l
'''
data=np.load("all_data.npy")


layers=((1,500),
        (500,500),
        (500,500),
        (500,2467))

params,l=init_params(layers)
m_par=len(params)
indexes=np.array(l)
indexes=np.concatenate((indexes[:-1].reshape(-1,1),indexes[1:].reshape(-1,1)),axis=1)
indexes=indexes.reshape(-1,2,2)

def model(x,parameter):
    y_mean=x
    for i in range(len(layers)):
        shapes=layers[i]
        n_in=shapes[0]
        n_out=shapes[1]
        W_fwd_mean=parameter[indexes[i,0,0]:indexes[i,0,1]].reshape(n_out,n_in)
        B_fwd_mean=parameter[indexes[i,1,0]:indexes[i,1,1]].reshape(n_out)
        y_mean=W_fwd_mean@y_mean+B_fwd_mean
        if i!=(len(layers)-1):
            y_mean=jax.nn.gelu(y_mean)
    return y_mean.reshape(-1)

optimizer = optax.adam(1e-04)
opt_state = optimizer.init(params)

def loss(x,y,parameter):
    return (jnp.linalg.norm(model(x,parameter).reshape(-1)-y.reshape(-1))**2).reshape(-1)[0]

loss=jax.jit(jax.value_and_grad(loss,argnums=2))

for epochs in trange(2000):
    tot_val=0
    grad_tot=0
    for i in range(1,len(data)):
        value,grad=loss(jnp.array([mu_vec[i]]),data[i],params)
        tot_val=tot_val+(1/len(data)*(1/data.shape[1]))*value
        grad_tot=grad_tot+(1/len(data))*grad

    print(np.sqrt(tot_val/((1/600)*(1/data.shape[1])*np.linalg.norm(data)**2)))

    updates, opt_state = optimizer.update(grad_tot, opt_state)
    params = optax.apply_updates(params, updates)


np.save("params.npy",params)


layers=((1,500),A_jax=sparse.BCOO((jnp.array(A_data),jnp.array(A_coords)),shape=A.shape)
M1_jax=sparse.BCOO((jnp.array(A_data),jnp.array(A_coords)),shape=A.shape)
M2_jax=sparse.BCOO((jnp.array(A_data),jnp.array(A_coords)),shape=A.shape)

        (500,500),
        (500,500),
        (500,2467))
params,l=init_params(layers)
m_par=len(params)
indexes=np.array(l)
indexes=np.concatenate((indexes[:-1].reshape(-1,1),indexes[1:].reshape(-1,1)),axis=1)
indexes=indexes.reshape(-1,2,2)

def model(x,parameter):
    y_mean=x
    for i in range(len(layers)):
        shapes=layers[i]
        n_in=shapes[0]
        n_out=shapes[1]
        W_fwd_mean=parameter[indexes[i,0,0]:indexes[i,0,1]].reshape(n_out,n_in)
        B_fwd_mean=parameter[indexes[i,1,0]:indexes[i,1,1]].reshape(n_out)
        y_mean=W_fwd_mean@y_mean+B_fwd_mean
        if i!=(len(layers)-1):
            y_mean=jax.nn.gelu(y_mean)
    return y_mean.reshape(-1)




def compute_loss(mu,x):
    u=x[:2178]
    p=x[2178:]
    return np.linalg.norm((1/mu)*C.dot(u).dot(u)+A.dot(u)-B.dot(p)-b)**2+np.linalg.norm(Bt.dot(u))**2

loss=0

for i in trange(600):
    loss=loss+1/600*compute_loss(mu_vec[i],data[i])

print(loss)


u=jnp.array(arr[:2178])
p=jnp.array(arr[2178:])

arr=jnp.array(arr)
def compute_loss(mu,params):
    x=model(mu,params).reshape(-1)+arr
    u=x[:2178]
    p=x[2178:]
    return (jnp.linalg.norm((1/mu)*((C_jax@u)@u).reshape(-1)+(A_jax@u).reshape(-1)-(B_jax@p).reshape(-1)-b_jax)**2+(Bt_jax@u)**2).reshape(-1)[0]

compute_loss=jax.jit(jax.value_and_grad(compute_loss,argnums=1))

optimizer = optax.adam(1e-04)
opt_state = optimizer.init(params)

dem=np.sqrt(1/len(data)*(1/data.shape[1])*(np.linalg.norm(data)**2))
for epochs in trange(5):
    tot_val=0
    l2_err=0
    grad_tot=0
    for i in trange(1,len(data)):
        value,grad=compute_loss(jnp.array([mu_vec[i]]),params)
        rec=arr+model(jnp.array([mu_vec[i]]),params)
        tot_val=tot_val+(1/data.shape[1])*(1/len(data))*value
        l2_err=l2_err+(1/data.shape[1])*1/len(data)*np.linalg.norm(rec.reshape(-1)-data[i].reshape(-1))**2
        grad_tot=grad_tot+(1/len(data))*grad

    print("-----------------------------------------------------------------------------------------------------------")
    print("DisPinn-Loss:,", np.sqrt(tot_val)/dem)
    print("L2 error:," ,np.sqrt(l2_err)/dem)

    updates, opt_state = optimizer.update(grad_tot, opt_state)
    params = optax.apply_updates(params, updates)


np.save("params_dispinn.npy",params)
'''

@LinearForm
def start(v,w):
    x,y=w.x
    f= (np.abs(y-1)<0.001)*np.array([1.,0.]) + np.array([0.,0.])
    return dot(f,v)

time_sim=np.zeros((100,arr.shape[0]))
M = asm(mass, basis['u'])

start_vec=np.zeros_like(arr)



for elem in up_dofs_u:
    M[:,elem]=0




'''

for i in trange(1,100):
    def compute_loss(x):
        u=x[:2178]
        p=x[2178:]
        return np.linalg.norm((M.dot(u)-M.dot(time_sim[i-1,:2178]))/(mu_vec[0])+(1/mu_vec[0])*C.dot(u).dot(u)+A.dot(u)-B.dot(p)-b)**2+np.linalg.norm(Bt.dot(u))**2

    def jac(x):
        u=x[:2178]
        p=x[2178:]
        l=(M.dot(u)-M.dot(time_sim[i-1,:2178]))/(mu_vec[0])+1/mu_vec[0]*(C.dot(u).dot(u))+A.dot(u)-B.dot(p)-b
        tmp=(M)/(mu_vec[0])+(C.dot(u)+C_T.dot(u))*(1/mu_vec[0])+A
        bb=np.asarray((B.multiply(B)).sum(axis=1)).reshape(-1)
        grad_u=np.asarray(np.sum(l*tmp,axis=0)).reshape(-1)+2*u*bb
        l=csr_matrix(l).transpose()
        grad_p=np.asarray(np.sum(B.multiply(l),axis=0)).reshape(-1)
        return np.concatenate((grad_u,grad_p))
        

    res=minimize(compute_loss,arr,options={'disp': True, 'maxiter': 1000}, jac=jac, method='L-BFGS-B')
    time_sim[i]=res.x

np.save("time_sim.npy",time_sim)
'''

M=M.tocoo()

M_coords=np.concatenate((M.row.reshape(-1,1),M.col.reshape(-1,1)),axis=1)

M_jax=sparse.BCOO((jnp.array(M.data),jnp.array(M_coords)),shape=M.shape)


def fwd1(mu,u,p,v):
    return 1/mu*(M_jax@(u-v))+(1/mu)*((C_jax@u)@u).reshape(-1)+(A_jax@u).reshape(-1)-(B_jax@p).reshape(-1)-b_jax

def fwd2(u):
    return Bt_jax@u

fwd1=jax.vmap(fwd1,(None,0,0,0))
fwd2=jax.vmap(fwd2)

layers=((1,500),
        (500,500),
        (500,500),
        (500,2467*100))

params,l=init_params(layers)
m_par=len(params)
indexes=np.array(l)
indexes=np.concatenate((indexes[:-1].reshape(-1,1),indexes[1:].reshape(-1,1)),axis=1)
indexes=indexes.reshape(-1,2,2)


def model(x,parameter):
    y_mean=x
    for i in range(len(layers)):
        shapes=layers[i]
        n_in=shapes[0]
        n_out=shapes[1]
        W_fwd_mean=parameter[indexes[i,0,0]:indexes[i,0,1]].reshape(n_out,n_in)
        B_fwd_mean=parameter[indexes[i,1,0]:indexes[i,1,1]].reshape(n_out)
        y_mean=W_fwd_mean@y_mean+B_fwd_mean
        if i!=(len(layers)-1):
            y_mean=jax.nn.gelu(y_mean)
    return y_mean.reshape(-1)

start_vec_jax=jnp.array(start_vec)
u_add=jnp.concatenate((start_vec_jax[:2178].reshape(1,-1),jnp.tile(arr[:2178].reshape(1,-1),(99,1))),axis=0)
p_add=jnp.concatenate((start_vec_jax[2178:].reshape(1,-1),jnp.tile(arr[2178:].reshape(1,-1),(99,1))),axis=0)
start_vec_jax=start_vec_jax[:2178]
def compute_loss(mu,params):
    x=model(mu,params).reshape(100,-1)
    u=x[:,:2178]+u_add
    p=x[:,2178:]+p_add
    loss=(1/100)*(jnp.linalg.norm((fwd1(mu,u[1:],p[1:],u[:-1]))**2+jnp.linalg.norm(fwd2(u[1:]))**2).reshape(-1)[0]+(jnp.linalg.norm(u[0]-start_vec_jax)**2)).reshape(-1)[0]
    return loss

compute_loss=jax.jit(jax.value_and_grad(compute_loss,argnums=1))

optimizer = optax.adam(1e-04)
opt_state = optimizer.init(params)


time_sim=np.load("time_sim.npy").reshape(-1)

tmp=jnp.concatenate((u_add,p_add),axis=1).reshape(-1)
for epochs in trange(3):
    tot_val=0
    l2_err=0
    grad_tot=0
    val=model(jnp.array([mu_vec[0]]),params).reshape(-1)+tmp
    print("L2 loss:", np.linalg.norm(val-time_sim)/np.linalg.norm(time_sim))
    for i in trange(1,len(mu_vec)):
        value,grad=compute_loss(jnp.array([mu_vec[i]]),params)
        tot_val=tot_val+(1/arr.shape[0])*(1/len(mu_vec))*value
        grad_tot=grad_tot+(1/len(mu_vec))*grad
    print("-----------------------------------------------------------------------------------------------------------")
    print("DisPinn-Loss:,", np.sqrt(tot_val)/np.linalg.norm(time_sim))

    updates, opt_state = optimizer.update(grad_tot, opt_state)
    params = optax.apply_updates(params, updates)


np.save("params_dispinn_timedep.npy",params)

#test with mu=100


