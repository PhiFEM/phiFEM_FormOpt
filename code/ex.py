from mpi4py import MPI
import numpy as np
from dolfinx.mesh import create_unit_square
from dolfinx.fem import functionspace, Function, assemble_scalar
from ufl import dx, TestFunction, dot

# Crear malla
mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

# Espacio escalar CG1
V = functionspace(mesh, ("CG", 1))

# Función de prueba zeta
zeta = TestFunction(V)

# Función h que vamos a interpolar
h = Function(V)
h.interpolate(lambda x: x[0] + x[1])  # escalar: x + y

# Definir la forma escalar
a = dot(zeta, h) * dx

# Para ilustrar que funciona: ensamblar el valor total de la forma
# (escalar, no una matriz ni un vector)
