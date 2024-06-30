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

A,b = enforce(A,b=np.zeros(A.shape[0]),x=up_basis,D=up_dofs_u)


up_dofs_u=_flatten_dofs(up_dofs_u)

for elem in up_dofs_u:
    B[:,elem]=0


C = asm(mass, basis['p'])

A = enforce(A, x=wall_basis,D=wall_dofs_u)[0]

wall_dofs_u=_flatten_dofs(wall_dofs_u)

for elem in wall_dofs_u:
    B[:,elem]=0


K = bmat([[A, -B.T],
          [-B, 1e-20 * C]], 'csr')

b=np.concatenate((b,np.zeros(C.shape[1])))

x=solve(K,b)
print(A.shape)
print(B.shape)
u=x[:2178]

np.save("stokes.npy",x)

mesh.save('test_velocity.vtk',
              {'velocity': u[basis['u'].nodal_dofs].T})
