from skfem import *
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence, rot
from skfem.helpers import grad, dot, ddot
from skfem.utils import _flatten_dofs
import numpy as np
from scipy.optimize import minimize
from sparse import COO
from skfem.visuals.matplotlib import plot
import numpy as np
from scipy.sparse import csr_matrix,hstack
import scipy.sparse
#import torch
#from tqdm import trange
#from torch import nn
#torch.set_default_dtype(torch.float64)

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

A=A.tocoo()
B=B.tocoo()
A_data=A.data
B_data=B.data
C_data=C.data
A_coords=np.concatenate((A.row.reshape(1,-1),A.col.reshape(1,-1)),axis=0)
B_coords=np.concatenate((B.row.reshape(1,-1),B.col.reshape(1,-1)),axis=0)
C_coords=C.coords
print(B.shape)

np.savez("all_files.npz",A_data=A_data,B_data=B_data,C_data=C_data,A_coords=A_coords,B_coords=B_coords,C_coords=C_coords,b=b)

assert False

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
    return (1/mu)*(C.dot(u).dot(u))


def jac_left_term(x):
    u=x[:2178]
    u=COO(u)
    grad_u=((C.dot(u)+C_T.dot(u))).tocsr()
    grad_p=csr_matrix((B.shape[0],B.shape[1]))
    h=hstack((grad_u,grad_p))
    return h


def other_term(x):
    u=x[:2178]
    return Bt.dot(u)

def jac_other_term(x):
    grad_p=csr_matrix((Bt.shape[0],Bt.shape[0]))
    return scipy.sparse.bmat([[Bt, grad_p]])


def csr_to_torch(M):
    M2=M.tocoo()
    row = torch.from_numpy(M2.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(M2.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    val = torch.from_numpy(M2.data.astype(np.float64))
    out = torch.sparse.DoubleTensor(edge_index, val, torch.Size(M.shape))
    return out


arr=np.load("stokes.npy")

mu_vec=np.linspace(0.1,10,600)

'''
all_data=np.zeros((600,arr.shape[0]))
for i in trange(1,600):
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

'''

data=np.load("all_data.npy")




start_u=np.zeros(2178)
start_u[up_dofs_u]=1.
start_u=torch.tensor(start_u)


torch.manual_seed(0)
arr=torch.tensor(arr)
data=torch.tensor(data)

arr=torch.tensor(np.load("stokes.npy"))
arr_tmp=arr[:2178]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(1,500),nn.Tanh(),nn.Linear(500,500),nn.Tanh(),nn.Linear(500,500),nn.Tanh(),nn.Linear(500,N))

    def forward(self,t):
        tmp=self.model(t)
        u=tmp[:2178]
        p=tmp[2178:]
        return torch.concatenate((start_u+20*t*(u+arr_tmp-start_u),p))

mu_vec=torch.tensor(mu_vec).reshape(-1,1)
model=Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


'''
for epochs in trange(300):
    tmp=0
    down_loss=0
    for i in range(1,600):
        optimizer.zero_grad()
        actual_loss=torch.linalg.norm(model(mu_vec[i])-data[i])**2
        actual_loss.backward()
        optimizer.step()
        with torch.no_grad():
            down_loss=down_loss+torch.linalg.norm(data[i])**2
            tmp=tmp+actual_loss

    with torch.no_grad():
        print(np.sqrt(tmp/down_loss))


torch.save(model, "model_data.pt")
'''
class LeftTerm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.tensor(left_term(input.numpy()))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        tmp=csr_to_torch(jac_left_term(input.numpy()))
        return tmp.transpose(0,1).mm(grad_output.unsqueeze(1)).squeeze(1)


class DiffusionTerm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.tensor(diffusion_term(input.numpy()))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        tmp=csr_to_torch(jac_diffusion_term(input.numpy()))
        return tmp.transpose(0,1).mm(grad_output.unsqueeze(1)).squeeze(1)



class OtherTerm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.tensor(other_term(input.numpy()))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        tmp=csr_to_torch(jac_other_term(input.numpy()))
        return tmp.transpose(0,1).mm(grad_output.unsqueeze(1)).squeeze(1)








def compute_loss(x):
    u=x[:2178]
    p=x[2178:]
    return np.linalg.norm((1/mu)*(C.dot(u).dot(u))+A.dot(u)-B.dot(p)-b)**2+np.linalg.norm(Bt.dot(u))**2 #pressure is normalized

def jac(x):
    u=x[:2178]
    p=x[2178:]
    l=2*((1/mu)*C.dot(u).dot(u)+A.dot(u)-B.dot(p)-b)
    tmp=C.dot(u)+C_T.dot(u)+mu*A
    bb=np.asarray((B.multiply(B)).sum(axis=1)).reshape(-1)
    grad_u=np.asarray(np.sum(l*tmp,axis=0)).reshape(-1)+2*u*bb
    l=csr_matrix(l).transpose()
    grad_p=np.asarray(np.sum(B.multiply(l),axis=0)).reshape(-1)
    return np.concatenate((grad_u,grad_p))

class Loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.tensor(compute_loss(input.numpy()))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output * torch.tensor(jac(input.numpy()))





model=Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


def loss(model,t):
    tmp=torch.func.jacrev(model)(t).reshape(-1)
    u=tmp[:2178]
    return torch.linalg.norm(u-LeftTerm.apply(model(t)))+torch.linalg.norm(OtherTerm.apply(model(t)))


mu_vec=mu_vec.reshape(-1,1)
for epochs in trange(3):
    actual_loss=model(torch.tensor([0.]))**2
    for i in trange(1,600):
        optimizer.zero_grad()
        t=mu_vec[i]
        actual_loss=torch.exp(-20*t)*loss(model,t)
        actual_loss.backward()
        optimizer.step()
        with torch.no_grad():
            print(actual_loss)


optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001)
for epochs in trange(3):
    for i in trange(1,500):
        def closure():
            optimizer.zero_grad()
            t=mu_vec[i]
            actual_loss=torch.exp(-20*t)*loss(model,t)
            actual_loss.backward()
            return actual_loss
        optimizer.step(closure)
        with torch.no_grad():
            actual_loss=torch.exp(-20*mu_vec[i])*loss(model,mu_vec[i])
            print(actual_loss)


torch.save(model,"model_cavity.pt")

assert False

model=torch.load("model_cavity.pt")
with torch.no_grad():
    gradient=torch.func.jacrev(model)(torch.tensor([0.05])).reshape(-1)
    gradient=gradient[:2178]
    u=model(torch.tensor([0.05]))
    print(torch.linalg.norm(gradient-LeftTerm.apply(u)))
    print(torch.linalg.norm(OtherTerm.apply(u)))

u=u[:2178]


mesh.save('test_velocity_net.vtk',
              {'velocity': u[basis['u'].nodal_dofs].T})

