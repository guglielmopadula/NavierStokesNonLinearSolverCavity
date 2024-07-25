import jax
import numpy as np
import scipy.sparse
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




A = asm(vector_laplace, basis['u'])
B = asm(divergence, basis['u'], basis['p'])



def matmul(u,v):
    v=np.expand_dims(v,axis=1)    
    return np.sum(u*v,axis=1)

@TrilinearForm
def c(u, v, w, _):
    return dot(matmul(grad(u), v),w)



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

mu=1

def diffusion_term(x):
    u=x[:2178]
    p=x[2178:]
    return A.dot(u)-B.dot(p)-b


def jac_diffusion_term(x):
    u=x[:2178]
    grad_u=A
    grad_p=-B
    h=hstack((grad_u,grad_p))
    return h


def left_term(x):
    u=x[:2178]
    u=np.array(u)
    return (C.dot(u).dot(u))

arr=np.load("stokes.npy")
#print(left_term(np.array(arr)).shape)

def jac_left_term(x):
    u=x[:2178]
    u=np.array(u)
    u=COO(u)
    grad_u=((C.dot(u)+C_T.dot(u))).tocsr()
    grad_p=csr_matrix((B.shape[0],B.shape[1]))
    h=hstack((grad_u,grad_p))
    return h


def left_term_jax(input):
    result_shape=jax.ShapeDtypeStruct((2178,),input.dtype)
    return jax.pure_callback(left_term,result_shape,input)

def csr_to_jax(M):
    M2=M.tocoo()
    row = jnp.array(M2.row.astype(np.int64))
    col = jnp.array(M2.col.astype(np.int64))
    edge_index = jnp.concatenate([row.reshape(-1,1), col.reshape(-1,1)], axis=1)
    val = jnp.array(M2.data.astype(np.float64))
    out = jax.experimental.sparse.BCOO((val, edge_index), shape=M.shape)
    return out

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


left_term_jax=jax.custom_jvp(left_term_jax)


@left_term_jax.defjvp
def _left_term_jax_jvp(primals,tangents):
    v,=primals
    z,=tangents
    tmp=csr_to_jax(jac_left_term(v))
    return left_term_jax(v),tmp@z



def other_term(x):
    u=x[:2178]
    return Bt.dot(u)

def jac_other_term(x):
    grad_p=csr_matrix((Bt.shape[0],Bt.shape[0]))
    return scipy.sparse.bmat([[Bt, grad_p]])

def other_term_jax(input):
    result_shape=jax.ShapeDtypeStruct((289,),input.dtype)
    return jax.pure_callback(other_term,result_shape,input)

other_term_jax=jax.custom_jvp(other_term_jax)


@other_term_jax.defjvp
def _other_term_jax_jvp(primals,tangents):
    v,=primals
    z,=tangents
    tmp=csr_to_jax(jac_other_term(v))
    return other_term_jax(v),tmp@z






def diffusion_term_jax(input):
    result_shape=jax.ShapeDtypeStruct((2178,),input.dtype)
    return jax.pure_callback(diffusion_term,result_shape,input)

diffusion_term_jax=jax.custom_jvp(diffusion_term_jax)


@diffusion_term_jax.defjvp
def _diffusion_term_jax_jvp(primals,tangents):
    v,=primals
    z,=tangents
    tmp=csr_to_jax(jac_diffusion_term(v))
    return diffusion_term_jax(v),tmp@z


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
        l=1/mu*(C.dot(u).dot(u))+A.dot(u)-B.dot(p)-b
        tmp=(C.dot(u)+C_T.dot(u))*(1/mu)+A
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




def compute_loss(mu,x):
    return (jnp.linalg.norm((1/mu)*left_term_jax(x).reshape(-1)+diffusion_term_jax(x).reshape(-1))**2+(jnp.linalg.norm(other_term_jax(x))**2).reshape(-1))[0]

def compute_loss(mu,x):
    u=x[:2178]
    p=x[2178:]
    return np.linalg.norm((1/mu)*C.dot(u).dot(u)+A.dot(u)-B.dot(p)-b)**2+np.linalg.norm(Bt.dot(u))**2

loss=0

for i in trange(600):
    loss=loss+1/600*compute_loss(mu_vec[i],data[i])

print(loss)


def compute_loss(mu,params):
    return (jnp.linalg.norm((1/mu)*left_term_jax(model(mu,params)).reshape(-1)+diffusion_term_jax(model(mu,params)).reshape(-1))**2+(jnp.linalg.norm(other_term_jax(model(mu,params)))**2).reshape(-1))[0]

compute_loss=jax.value_and_grad(compute_loss,argnums=1)

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

