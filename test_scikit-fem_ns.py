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
from scipy.sparse import csr_matrix
import torch

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




mu=1

Bt=B.T
def compute_loss(x):
    u=x[:2178]
    p=x[2178:]
    return np.linalg.norm(C.dot(u).dot(u)+mu*A.dot(u)-B.dot(p)-b)**2+np.linalg.norm(Bt.dot(u))**2

def jac(x):
    u=x[:2178]
    p=x[2178:]
    l=(C.dot(u).dot(u)+mu*A.dot(u)-B.dot(p)-b)
    tmp=C.dot(u)+C_T.dot(u)+mu*A
    bb=np.asarray((B.multiply(B)).sum(axis=1)).reshape(-1)
    grad_u=np.asarray(np.sum(l*tmp,axis=0)).reshape(-1)+2*u*bb
    l=csr_matrix(l).transpose()
    grad_p=np.asarray(np.sum(B.multiply(l),axis=0)).reshape(-1)
    return np.concatenate((grad_u,grad_p))



tmp=np.load("stokes.npy")
jac(tmp)
res=minimize(compute_loss,tmp,options={'disp': True, 'maxiter': 100}, jac=jac, method='L-BFGS-B')
x=res.x
u=x[:2178]


mesh.save('test_velocity.vtk',
              {'velocity': u[basis['u'].nodal_dofs].T})

