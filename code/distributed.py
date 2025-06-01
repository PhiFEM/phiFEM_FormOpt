from pathlib import Path

import gmsh
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import gmshio, XDMFFile

import numpy as np
from numpy.linalg import norm as Norm

from dolfinx.mesh import (
    Mesh, locate_entities_boundary,
    exterior_facet_indices, meshtags
)

from dolfinx.fem.petsc import (
    LinearProblem, DirichletBC,
    NonlinearProblem,
    assemble_matrix, assemble_vector,
    create_vector, apply_lifting, set_bc
)

from dolfinx.fem import (
    FunctionSpace,
    Function, Constant, 
    create_interpolation_data,
    locate_dofs_topological,
    locate_dofs_geometrical,
    assemble_scalar, form,
    functionspace, dirichletbc,
    compile_form, create_form
)

from ufl import (
    SpatialCoordinate,
    FacetNormal, Measure,
    TrialFunction, TestFunction,
    system, conditional,
    inner, grad, dot, sqrt, gt
)

from dolfinx.nls.petsc import NewtonSolver

from abc import ABC, abstractmethod

import numpy.typing as npt
from typing import Any, List, Tuple, final
from ufl.core.expr import Expr as ufl_expr

from functools import wraps

from plots import plot_domain

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
comm_self = MPI.COMM_SELF

dom_name = "domain"
dat_name = "data"
res_name = "results"
ini_name = "initial"

class Model(ABC):
    """
    Base class for all user-defined models.
    """
    dim: int
    domain: Mesh
    space: FunctionSpace
    path: Path
    
    def __init__(self):
        pass

    @abstractmethod
    def pde(
            self,
            level_set_func
        ) -> List[Tuple[ufl_expr, DirichletBC]]:
        """
        Returns:
            A list with elements of the form
            (wk, bc) or (wk, bc, jac, unk, ini_func)
        where
        wk : weak formulation of a state equation
        bc : the corresponding list of dirichlet boundary conditions
        jac : jacobian of wk
        unk : unknown in the nonlinear equation
        ini_func : callable function to define the initial guess
        """
        pass
        
    @abstractmethod
    def adjoint(
            self,
            level_set_func,
            states
        ) -> List[Tuple[ufl_expr, DirichletBC]]:
        """
        Returns:
            A list with elements of the form
            (wk, bc) or (wk, bc, jac, unk, ini_func)
        where
        wk : weak formulation of the state equation
        bc : the corresponding list of dirichlet boundary conditions
        jac : jacobian of wk
        unk : unknown in the nonlinear equation
        ini_func : callable function to define the initial guess
        """
        pass

    @abstractmethod
    def cost(
            self,
            level_set_func,
            states
        ) -> ufl_expr:
        """
        Returns the cost functional J
        """
        pass
    
    @abstractmethod
    def constraint(
            self,
            level_set_func,
            states
        ) -> List[ufl_expr]:
        """
        Returns:
            [C0, C1, ...]
        where
        Cj : the form of the j-th equality constraint
        """
        pass
    
    @abstractmethod
    def derivative(
            self,
            level_set_func,
            states,
            adjoints
        ) -> Tuple[Tuple[ufl_expr, List[ufl_expr]], Tuple[ufl_expr, List[ufl_expr]]]:
        """
        Returns:
            (S0_J, [S0_C0, S0_C1, ...]), (S1_J, [S1_C0, S1_C1, ...])
        where
        S0_J, S1_J   : derivative components of J
        S0_Cj, S1_Cj : derivative components of Cj
        """
        pass
    
    @abstractmethod
    def bilinear_form(
            self,
            velocity_func,
            test_func
        ) -> Tuple[ufl_expr, bool]:
        """
        Returns:
            the bilinear form B
        """
        pass
    
    def __verification(self, required_attrs: List[str]):
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise NotImplementedError(
                    f"Models must define the '{attr}' attribute."
                )
    
    @final
    def create_initial_level(
            self,
            centers: npt.NDArray[np.float64],
            radii: npt.NDArray[np.float64],
            factor: float = 1.0,
            ord: int = 2
        ) -> None:
        """
        Instantiates a InitialLevel object.
        """

        self.ini_lvl = InitialLevel(centers, radii, factor, ord)

    @final
    def save_initial_level(self, comm: MPI.Comm) -> None:
        self.__verification(['domain', 'path'])
        self.ini_lvl.save(comm, self.domain, self.path)

    @final
    def get_initial_level(self):
        return self.ini_lvl.func
    
    @final
    def runDP(
            self,
            niter: int = 100,
            reinit_step: int = 4,
            reinit_pars: Tuple[int, float] = (8, 1e-2),
            dfactor: float = 1e-2,
            lv_time: Tuple[float, float] = (1e-3, 1.0),
            lv_iter: Tuple[int, int] = (8, 14),
            smooth: bool = False,
            start_to_check: int = 30,
            ctrn_tol: float = 1e-2,
            lgrn_tol: float = 1e-2,
            cost_tol: float = 1e-2,
            prev: int = 10,
            seed: int = 26    
        ) -> None:
        """
        Data Parallelism
        """
        
        self.__verification(['dim', 'domain', 'space', 'path'])

        runDP(
            self, niter, reinit_step, reinit_pars, dfactor,
            lv_time, lv_iter, smooth, start_to_check,
            ctrn_tol, lgrn_tol, cost_tol, prev, seed
        )
    
    @final
    def runTP(self, ):
        """
        Task Parallelism
        """
        pass
    
    @final
    def runMP(self, ):
        """
        Mix Parallelism
        """
        pass


class Subdomain:
    """
    Creates an ufl conditional from a
    list o inequalities of the form
        "expresion of x > 0",
    which represents a subdomain
    of almost zero velocity.
    """

    def __init__(self, domain, cond_func):
        self.domain = domain
        self.cond_func = cond_func

    def expression(self):
        x = SpatialCoordinate(self.domain)
        conditions = self.cond_func(x)
        chi = 1.0
        for cond in conditions:
            chi *= conditional(gt(cond, 0.0), 1.0, 0.0)
        return chi
    
def region_of(domain):
    def wrapper(func):
        return Subdomain(domain, func)
    return wrapper

class Save:
    """
    Class to save "scalar" numerical
    results: cost values, derivative norms,
    Lagrange multipliers, etc.
    """

    def __init__(self):
        
        self.cost = []
        self.nder = []
        self.ppl_obj = None
            
    def add(self, cost, nder):
        
        self.cost.append(cost)
        self.nder.append(nder)
    
    def add_times(self, assembly, resolution):
        self.times = [assembly, resolution]
    
    def add_ppl(self, ppl_obj):
        self.ppl_obj = ppl_obj

    def save(self, path):
        if self.ppl_obj is None:
            np.savez(
                path / f"{dat_name}.npz",
                cost = np.array(self.cost),
                nder = np.array(self.nder),
                times = self.times
            )
        else:
            np.savez(
                path / f"{dat_name}.npz",
                cost = np.array(self.cost),
                ctrs = np.array(self.ppl_obj.list_ct),
                nder = np.array(self.nder),
                Lg = np.array(self.ppl_obj.list_Lg),
                lm = np.array(self.ppl_obj.list_lm),
                mu = np.array(self.ppl_obj.list_mu),
                z = np.array(self.ppl_obj.list_zs),
                delta = np.array(self.ppl_obj.list_dl),
                alpha = np.array(self.ppl_obj.alpha),
                beta = np.array(self.ppl_obj.beta),
                rho = np.array(self.ppl_obj.rho),
                r = np.array(self.ppl_obj.r),
                times = self.times
            )


class InitialLevel:
    """
    Creates the initial level set function
    with ball shaped holes determined by
    centers and radii.
    
    Attributes
    ----------
    centers : numpy.ndarray
        Array of center coordinates.
    radii : numpy.ndarray
        Array of radii.
    dim : int
        Problem dimension.
    factor : float
        Scaling factor.
    ord : int
        Order of the norm.
    
    Methods
    -------
    func(x) :
        Callable function to be interpolated
        by a dolfinx function.
    save() :
        Interpolates func and saves it into
        a xdmf file.
    """

    def __init__(
            self,
            centers: npt.NDArray[np.float64],
            radii: npt.NDArray[np.float64],
            factor: float = 1.0,
            ord: int = 2
        ):
        """
        Arguments
        ---------
        centers : numpy.ndarray
            Array of center coordinates
            of shape (N, 2) or (N, 3).
        radii : numpy.ndarray 
            Array of radii of shape (N,).
        factor : float
            Scaling factor, positive or negative.
        ord : int
            Order of norm. Positive integer
            grater than 1 or infinity (np.inf).
        """
        self.centers = centers
        self.radii = radii
        self.dim = self.centers.shape[1]
        self.factor = factor
        self.ord = ord
    
    def func(self, x):
        """
        Callable function to be interpolated
        by a dolfinx function.
        """
        
        xT = (x[:self.dim].T)[None, :, :]
        #comps = (self.centers[:, None, :] - xT)**2
        #norms = np.sqrt(np.sum(comps, axis = 2))
        norms = Norm(
            self.centers[:, None, :] - xT,
            ord = self.ord, axis = 2
        )
        distances = self.radii[:, None] - norms
        values = self.factor*np.max(distances, axis = 0)

        return self.factor*values
    
    def save(self, comm, domain, save_path):
        """
        Interpolates func and saves it into
        a xdmf file.
        """

        space = functionspace(domain, ("CG", 1))
        intp_func = interpolate(
            [self.func], space, name = "phi0"
        )
        save_functions(
            comm, domain, intp_func,
            save_path / f"{ini_name}.xdmf"
        )


def get_funcs_from(space, values):
    """
    Returns a list of functions of the given
    values and defined on the given space.
    """
    funcs = [Function(space) for _ in range(len(values))]
    for f, v in zip(funcs, values):
        f.x.array[:] = v
    
    return funcs

def space_interpolation(from_space, funcs, to_space):
    
    to_domain = to_space.mesh
    new_funcs = [Function(to_space) for _ in range(len(funcs))]
    dim = to_domain.topology.dim
    fine_mesh_cell_map = to_domain.topology.index_map(dim)
    num_cells_on_proc = fine_mesh_cell_map.size_local + fine_mesh_cell_map.num_ghosts
    cells = np.arange(num_cells_on_proc, dtype = np.int32)
    interp_data = create_interpolation_data(to_space, from_space, cells, padding=1e-14)

    for nf, f in zip(new_funcs, funcs):
        nf.interpolate_nonmatching(f, cells, interpolation_data = interp_data)
    
    return new_funcs


def all_connectivities(domain):
    """
    Creates all basic connectivities.
    """
    topology = domain.topology
    dim = topology.dim

    pairs = [
        (dim, dim),
        (0, dim), (dim, 0),
        (1, dim), (dim, 1)
    ]

    if dim == 3:
        pairs += [(2, dim), (dim, 2)]

    for d0, d1 in pairs:
        topology.create_connectivity(d0, d1)

def dirichlet_extension_from_bcs(domain, space, list_bcs):
    
    dx = Measure("dx", domain = domain)
    n = len(list_bcs)
    ext = [Function(space) for _ in range(n)]
    for i in range(n): ext[i].name = "g" + str(i)

    h = Function(space)
    h.x.array[:] = 0
    eta = TrialFunction(space)
    zeta = TestFunction(space)
    a = inner(grad(eta), grad(zeta))*dx
    L = h*zeta*dx # Right-hand side is zero
    
    for bcs, g in zip(list_bcs, ext):
        basic_solver(form(a), form(L), bcs, g)

    return ext

def dirichlet_extension(domain, space, funcs):
    """
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        d-dimensional problem domain, for d = 2 or 3.
    space :
        Function space
    dir_funcs :
        List of functions to be used as Dirichlet conditions.
        These functions are defined on the whole domain.
        
    Returns
    -------
    List of extended functions.
    """
    dx = Measure("dx", domain = domain)
    nbr_fcs = len(funcs)
    extensions = [Function(space) for _ in range(nbr_fcs)]
    for i in range(nbr_fcs):
        extensions[i].name = "g" + str(i)
    
    dim = domain.topology.dim
    boundary_dofs = locate_dofs_topological(
        space, dim-1, exterior_facet_indices(domain.topology)
    )

    h = Function(space)
    h.x.array[:] = 0
    eta = TrialFunction(space)
    zeta = TestFunction(space)
    a = inner(grad(eta), grad(zeta))*dx
    L = dot(zeta, h)*dx # Right-hand side is zero
    
    for f, g in zip(funcs, extensions):
        basic_solver(
            form(a), form(L), 
            [dirichletbc(f, boundary_dofs)], g
        )

    return extensions


def build_gmsh_model_2d(
        vertices, boundary_parts, mesh_size,
        holes = None, curve = None,
        filename = "domain.msh", plot = False,
        quad = False
    ):
    """
    Arguments
    ---------
        vertices :
            ...
        boundary_parts :
            ...
        mesh_size :
            ...
        holes :
            ...
        curve :
            ...
        path :
            ...
        plot :
            ...
    
    Returns
    -------
        ...

    """
    rank_to_build = 0
    # Tags:
    # point tags = 1, 2, ..., K
    # line tags = 1, 2, ..., K
    if rank == rank_to_build:
        gmsh.initialize()
        gmsh.model.add("2D_Mesh")
        gmsh.option.setNumber("General.Terminal", 0)
        
        K = len(vertices)
        line_tags = [k for k in range(1, K + 1)]
        curve_tag = 1
        surface_tag = 1

        # boundary lines
        gmsh.model.geo.addPoint(
            x = vertices[0][0],
            y = vertices[0][1],
            z = 0.,
            meshSize = 0.,
            tag = 1
        )
        for k in range(1, K):
            gmsh.model.geo.addPoint(
                x = vertices[k][0],
                y = vertices[k][1],
                z = 0.,
                meshSize = 0.,
                tag = k + 1
            )
            gmsh.model.geo.addLine(
                startTag = k,
                endTag = k + 1,
                tag = k
            )
        gmsh.model.geo.addLine(
            startTag = K,
            endTag = 1,
            tag = K
        )
        gmsh.model.geo.addCurveLoop(curveTags = line_tags, tag = curve_tag)
        
        last_tag = K

        if holes is not None:
            list_holes = []
            curve_hole_tag = 2 # 2, 3, ...
            for hole in holes:
                list_holes.append(curve_hole_tag)
                J = len(hole)
                hole_tags = [last_tag + j for j in range(1, J + 1)]
                gmsh.model.geo.addPoint(
                    x = hole[0][0],
                    y = hole[0][1],
                    z = 0.,
                    meshSize = 0.,
                    tag = last_tag + 1
                )
                for j in range(1, J):
                    gmsh.model.geo.addPoint(
                        x = hole[j][0],
                        y = hole[j][1],
                        z = 0.,
                        meshSize = 0.,
                        tag = last_tag + j + 1
                    )
                    gmsh.model.geo.addLine(
                        startTag = last_tag + j,
                        endTag = last_tag + j + 1,
                        tag = last_tag + j
                    )
                gmsh.model.geo.addLine(
                    startTag = last_tag + J,
                    endTag = last_tag + 1,
                    tag = last_tag + J
                )
                gmsh.model.geo.addCurveLoop(curveTags = hole_tags, tag = curve_hole_tag)
                last_tag = last_tag + J # update last tag
                curve_hole_tag = curve_hole_tag + 1
            
            # create surface
            gmsh.model.geo.addPlaneSurface([curve_tag] + list_holes, tag = surface_tag)	
        else:
            # create surface
            gmsh.model.geo.addPlaneSurface([curve_tag], tag = surface_tag)	

        #--------------------------------------------
        # Code for add a interior curve that define
        # a subdomain. This part is used to generate
        # data for inverse problems.

        if curve is not None:
            L = len(curve)
            gmsh.model.geo.addPoint(
                x = curve[0][0],
                y = curve[0][1], 
                z = 0., 
                meshSize = 0.1,
                tag = last_tag + 1
            )
            for l in range(1, L):
                gmsh.model.geo.addPoint(
                    x = curve[l][0],
                    y = curve[l][1],
                    z = 0.,
                    meshSize = 0.1,
                    tag = last_tag + l + 1
                )
                gmsh.model.geo.addLine(
                    startTag = last_tag + l,
                    endTag = last_tag + l + 1, 
                    tag = last_tag + l
                )
            gmsh.model.geo.addLine(
                startTag = last_tag + L,
                endTag = last_tag + 1,
                tag = last_tag + L
            )

            # We have to synchronize before embedding the lines
            gmsh.model.geo.synchronize()
        
            gmsh.model.mesh.embed(
                dim = 1, 
                tags = [last_tag + l for l in range(1, L + 1)],
                inDim = 2,
                inTag = surface_tag
            )
        
        gmsh.model.geo.synchronize()

        # physical groups
        # domain
        gmsh.model.addPhysicalGroup(
            dim = 2,
            tags = [surface_tag],
            tag = 1,
            name = "domain"
        )
        # boundary
        if boundary_parts:
            # add mark to subsets of the boundary
            for facets, marker, name in boundary_parts:
                gmsh.model.addPhysicalGroup(
                    dim = 1,
                    tags = facets,
                    tag = marker,
                    name = name
                )
        else :
            # add mark to the whole boundary
            gmsh.model.addPhysicalGroup(
                dim = 1,
                tags = line_tags,
                tag = 1,
                name = "boundary"
            )

        # size mesh
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        
        if quad:
            gmsh.model.mesh.setRecombine(2, surface_tag)
        
        gmsh.model.mesh.generate(dim = 2)

        if quad:
            nbr_triangles = len(gmsh.model.mesh.getElementsByType(3)[0])
        else:
            nbr_triangles = len(gmsh.model.mesh.getElementsByType(2)[0])
        
        # Plot the mesh
        if plot :
            gmsh.fltk.run()
        
        # Write the mesh in a *.msh file
        gmsh.write(str(filename))
        gmsh.clear()
        gmsh.finalize()

    else:

        nbr_triangles = None
    
    nbr_triangles = comm.bcast(
        nbr_triangles, root = rank_to_build
    )

    return nbr_triangles

def interpolate(funcs, to_space, name = "f"):
    
    n = len(funcs)
    new_funcs = [Function(to_space) for _ in range(n)]
    
    if n == 1:
        new_funcs[0].name = name
        new_funcs[0].interpolate(funcs[0])
    else:
        for i in range(n):
            new_funcs[i].name = name + str(i)
            new_funcs[i].interpolate(funcs[i])

    return new_funcs

def save_domain(comm, domain, filename, facet_tags = None):
    
    with XDMFFile(comm, filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        if facet_tags:
            xdmf.write_meshtags(facet_tags, domain.geometry)

def save_initial(comm, initial_guess, domain, filename):
    
    initial_level = Function(functionspace(domain, ("CG", 1)))
    initial_level.name = "phi"
    initial_level.interpolate(InitialLevel(*initial_guess).func)
    
    with XDMFFile(comm, filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(initial_level, 0)

def save_functions(comm, domain, funcs, filename):
    with XDMFFile(comm, filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        for f in funcs: xdmf.write_function(f, 0)

def save_initial_level(comm, domain, func, filename):
    
    space = functionspace(domain, ("CG", 1))
    intp_func = interpolate([func], space, name = "phi")
    save_functions(comm, domain, intp_func, filename)

def read_gmsh(filename, comm, dim):
    return gmshio.read_from_msh(filename, comm = comm, gdim = dim)

def create_domain_2d_DP(
        vertices, boundary_parts, mesh_size,
        holes = None, curve = None, path = Path(""),
        plot = False
    ):
    """
    Create a polygonal domain with holes
    and an interior closed curve for
    data parallelism (DP).
    """
    filename = path / f"{dom_name}.msh"
    nbr_triangles = build_gmsh_model_2d(
        vertices, boundary_parts, mesh_size,
        holes, curve, filename, plot
    )
    
    # Read and save (.xmdf format) the mesh ======================
    comm.barrier()
    domain, _, facet_tags = read_gmsh(filename, comm, 2)
    all_connectivities(domain)
    save_domain(comm, domain, path / f"{dom_name}.xdmf", facet_tags)
    # ============================================================
    
    # Plot the distributed mesh
    plot_domain(domain, f"rank = {rank}")

    return domain, nbr_triangles, facet_tags

def create_domain_2d_TP(
        vertices, boundary_parts, mesh_size,
        holes = None, curve = None, path = "",
        plot = False
    ):
    """
    Create a polygonal domain with holes
    and an interior closed curve for
    task parallelism (TP).
    """
    filename = Path(path) / f"{dom_name}.msh"
    nbr_triangles = build_gmsh_model_2d(
        vertices, boundary_parts, mesh_size,
        holes, curve, filename, plot
    )

    # Read and save (.xmdf format) the mesh ===============================
    comm.barrier()
    domain, _, facet_tags = read_gmsh(filename, comm_self, 2)
    all_connectivities(domain)
    if rank == 0:
        save_domain(comm_self, domain, path / f"{dom_name}.xdmf", facet_tags)
    # =====================================================================

    # Plot the identical meshes
    plot_domain(domain, f"rank = {rank}") 
    
    return domain, nbr_triangles, facet_tags

def create_domain_2d_MP(
        sub_comm, color,
        vertices, boundary_parts, mesh_size,
        holes = None, curve = None, path = "",
        plot = False
    ):
    """
    Create a polygonal domain with holes
    and an interior closed curve for
    mix parallelism (MP).
    """

    filename = Path(path) / f"{dom_name}.msh"
    nbr_triangles = build_gmsh_model_2d(
        vertices, boundary_parts, mesh_size,
        holes, curve, filename, plot
    )

    # Read and save (.xmdf format) the mesh ==============================
    comm.barrier()
    domain, _, facet_tags = read_gmsh(filename, sub_comm, 2)
    all_connectivities(domain)
    if color == 0:
        save_domain(sub_comm, domain, path / f"{dom_name}.xdmf", facet_tags)
    # ====================================================================

    # Plot the identically distributed meshes
    plot_domain(domain, f"rank = {rank}")

    return domain, nbr_triangles, facet_tags

def print0(i, cost, const_vals, der_norm, steps, extra = ""):
    print(f"i = {i:3.0f} | "
        f"cost = {cost:.4f} | "
        f"cstr = {', '.join(f'{abs(v):.4f}' for v in const_vals)} | "
        f"nder = {der_norm:.4f} | "
        f"steps = {steps:2.0f} | " + extra)

def print1(i, cost, der_norm, steps, extra = ""):
    print(f"i = {i:3.0f} | "
        f"cost = {cost:.6f} | "
        f"nder = {der_norm:.4f} | "
        f"steps = {steps:2.0f} | " + extra)

def assemble_mtx(ufl_form, bcs = []):
    return assemble_matrix(form(ufl_form), bcs)


def eval(formula):
    return assemble_scalar(form(formula))

def const(domain, value):
    return Constant(domain, PETSc.ScalarType(value))

def volume(domain, comm):
    """
    Return the area (2D) or volume (3D).
    """
    dx = Measure("dx", domain = domain)
    local = eval(const(domain, 1)*dx)
    total = comm.allreduce(local, op = MPI.SUM)
    return total

def nbr_fems(domain, dim, comm):
    """
    Return the number of finite elements.
    """
    local = domain.topology.index_map(dim).size_local
    total = comm.allreduce(local, op = MPI.SUM)
    return total

def get_diam2(dim, vol, nfems):
    if dim == 2:
        return 4.0*vol/nfems/np.sqrt(3.0)
    elif dim == 3:
        return (6.0*np.sqrt(2.0)*vol/nfems)**(2.0/3.0)
    else:
        raise ValueError(f"> Unsupported dimension: {dim}.")
    
def create_space(domain, family, rank, degree = 1):
    """
    Creates a function space.

    Parameters
    ----------
    domain:
        The domain.
    family:
        The finite element type.
    degree:
        The polynomial degree.
    rank: 
        - 0 for scalar functions.
        - An integer `n` for vector-valued functions.
        - A tuple `(m, n, ...)` for tensor-valued functions.

    Returns
    -------
        A function space in the given domain.
    """

    if rank == 1:  # Scalar function space
        element = (family, degree)
    elif isinstance(rank, int):  # Vector-valued function space of dimension `rank`
        element = (family, degree, (rank, ))
    elif isinstance(rank, tuple):  # Tensor-valued function space of arbitrary shape
        element = (family, degree, rank)
    else:
        raise ValueError("The 'rank' argument must be an integer or a tuple.")

    return functionspace(domain, element)


def build_solver(domain, bilinear_form, dirichlet_bcs):
    A = assemble_matrix(bilinear_form, dirichlet_bcs)
    A.assemble()
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES) # or PREONLY
    solver.getPC().setType(PETSc.PC.Type.HYPRE) # or LU
    solver.setTolerances(
        rtol = 1e-6,
        atol = 1e-10,
        max_it = 1000
    )
    
    return solver

def create_solver(a, L, bcs, uh):
    """
    Creates a solver. To re-assemble the system and solve it,
    Run problem.sove()

    Reference:
    https://jsdokken.com/FEniCS-workshop/src/form_compilation.html
    """

    petsc_options = {
        "ksp_type": "gmres",      # Krylov method: GMRES (good for non-symmetric matrices)
        "pc_type": "hypre",       # Uses Hypre multigrid preconditioner (efficient for parallel computing)
        "ksp_rtol": 1e-6,         # Relative tolerance for convergence
        "ksp_atol": 1e-10,        # Absolute tolerance for convergence
        "ksp_max_it": 1000,       # Maximum number of iterations
        #"ksp_monitor": None
    }

    problem = LinearProblem(
        a, L, u = uh, bcs = bcs,
        petsc_options = petsc_options
    )

    return problem

def basic_solver(a, L, bcs, uh):
    """
    This configuration provides a good balance
    between performance and parallel scalability,  
    without requiring an expensive LU factorization.

    See for instance:
    https://jsdokken.com/FEniCS-workshop/src/form_compilation.html
    """
    petsc_options = {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-6,
        "ksp_atol": 1e-10,
        "ksp_max_it": 1000,
        #"ksp_monitor": None
    }
    
    problem = LinearProblem(
        a, L, u = uh, bcs = bcs,
        petsc_options = petsc_options
    )
    problem.solve()


class Velocity:
    """
    Implements the velocity equation.
    """
    
    def __init__(self, dim, domain, space, biform, S0, S1):
        """
        dim :
            domain dimension = velocity rank
        domain :
            2 or 3 dimensional mesh
        space :
            space of vector-valued functions
        """
        
        th = TrialFunction(space)
        xi = TestFunction(space)
        dx = Measure("dx", domain = domain)

        b, dirbc = biform(th, xi)

        self.biform = form(b)
        self.liform = form(
            -(dot(S0, xi) + inner(S1, grad(xi)))*dx
        )
        
        if dirbc == True:
            args = (domain, space, dim, dim)
            self.bc = homogeneus_boundary(*args)
        elif dirbc == False:
            self.bc = []
        else:
            pass
            # Implements a Dirichlet
            # boundary condition in 2D.
            # tags, marks = dirbc
            # self.bc = homogeneous_dirichlet(
            #     domain, space, tags, marks, dim
            # )
        
        self.solver = build_solver(
            domain, self.biform, self.bc
        )
    
    def run(self, theta):
        
        L = assemble_vector(self.liform)
        apply_lifting(L, [self.biform], [self.bc])
        L.ghostUpdate(
            addv = PETSc.InsertMode.ADD_VALUES,
            mode = PETSc.ScatterMode.REVERSE
        )
        set_bc(L, self.bc)
        
        self.solver.solve(L, theta.x.petsc_vec)
        theta.x.scatter_forward()
    

class Level:
    """
    Implements the Petrov Galerkin
    and Crank-Nicolson methods to solve
    the transport equation corresponding
    to the level set function.
    """

    def __init__(self, domain, space, phi, tht, diam2, smooth):

        self.domain = domain
        self.dt = const(domain, 0.0)
        dx = Measure("dx", domain = domain)
                
        u = TrialFunction(space)
        v = TestFunction(space)

        tau = 2.0*sqrt(
            1.0/self.dt**2 +
            dot(tht, tht)/diam2
        )
        new_v = v + dot(grad(v), tht)/tau

        # Weak formulation
        a = (u + (self.dt/2.0)*dot(tht, grad(u)))*new_v*dx
        L = (phi - (self.dt/2.0)*dot(tht, grad(phi)))*new_v*dx

        if smooth:
            a += (self.dt/2.0)*diam2*dot(grad(u), grad(v))*dx
            L -= (self.dt/2.0)*diam2*dot(grad(phi), grad(v))*dx
        
        # Pre-compilation
        self.a = form(a)
        self.L = form(L)

    def run(self, phi, steps, tend):
        
        self.dt.value = tend/steps
        
        solver = build_solver(
            self.domain,
            self.a,
            []
        )
        b = create_vector(self.L)
        
        for _ in range(steps):
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, self.L)
            b.ghostUpdate(
                addv = PETSc.InsertMode.ADD_VALUES,
                mode = PETSc.ScatterMode.REVERSE
            )
            solver.solve(b, phi.x.petsc_vec)
            phi.x.scatter_forward()


class Reinit:
    """
    """
    def __init__(self, domain, space, phi, diam2):
        
        self.domain = domain
        self.dt = const(domain, 0.0)
        self.phi_ini = Function(space)
        self.phi_prev = Function(space)
        self.uh = Function(space)
        self.w = Function(space)
        dx = Measure("dx", domain = domain) 

        sign_phi_ini = self.phi_ini/sqrt(self.phi_ini**2 + dot(grad(self.phi_ini), grad(self.phi_ini))*diam2)
        H = lambda p: sign_phi_ini*sqrt(dot(p, p))
        GradH = lambda p: sign_phi_ini*p/sqrt(dot(p, p))

        u = TrialFunction(space)
        v = TestFunction(space)

        # Explicit Euler method
        tau = 2.0*sqrt(
            1.0/self.dt**2 +
            dot(GradH(grad(phi)), GradH(grad(phi)))/diam2
        )
        new_v = v + dot(grad(v), GradH(grad(phi)))/tau

        a = u*new_v*dx
        L = phi*new_v*dx
        L += self.dt*sign_phi_ini*new_v*dx 
        L += (self.dt/2.0)*(H(grad(self.phi_prev)) - 3.0*H(grad(phi)))*new_v*dx

        self.problem = create_solver(
            form(a), form(L), [], self.uh
        )

        a0 = u*new_v*dx
        L0 = self.dt*sign_phi_ini*new_v*dx 
        L0 += (self.phi_ini - self.dt*H(grad(self.phi_ini)))*new_v*dx

        self.problem0 = create_solver(
            form(a0), form(L0), [], self.uh
        )

    def run(self, phi, steps, tend):
        
        self.dt.value = tend/steps
        self.phi_ini.interpolate(phi)

        self.problem0.solve()
        self.phi_prev.x.array[:] = self.uh.x.array[:]

        for _ in range(steps):
            self.w.x.array[:] = phi.x.array[:]
            self.problem.solve()
            phi.x.array[:] = self.uh.x.array[:]
            self.phi_prev.x.array[:] = self.w.x.array[:]


class PPL:
    """
    Perturbed Proximal Lagrangian Method
    Reference:
    Jong Gwang Kim, A new Lagrangian-based ﬁrst-order
    method for nonconvex constrained optimization, 2023
    """
    def __init__(self, n, ini_cost, ini_ctrs):
        
        self.n = n
        self.ones = np.ones(n)
        self.lm = np.repeat(0.0, n)
        self.mu = np.repeat(0.0, n)
        self.zs = np.repeat(0.0, n)
        self.alpha = 2000.0
        self.beta = 0.5
        self.rho = self.alpha/(1.0 + self.alpha*self.beta)
        self.r = 0.999
        self.dl = 0.5
        self.Lg = ini_cost
        self.ct = ini_ctrs

        self.list_Lg = [self.Lg]
        self.list_lm = [self.lm]
        self.list_mu = [self.mu]
        self.list_zs = [self.zs]
        self.list_dl = [self.dl]
        self.list_ct = [self.ct]

    def run(self, cost, constraints):
        
        self.ct = np.array(constraints)
        self.lmu = self.lm - self.mu
        self.mu = self.mu + self.dl*self.lmu/(np.inner(self.lmu, self.lmu) + 1)
        self.lm = self.mu + self.rho*(self.ct - self.ones)
        self.zs = (self.lm - self.mu)/self.alpha
        self.dl = self.r*self.dl

        self.Lg = self.lagrangian(cost)

        self.list_Lg.append(self.Lg)
        self.list_lm.append(self.lm)
        self.list_mu.append(self.mu)
        self.list_zs.append(self.zs)
        self.list_dl.append(self.dl)
        self.list_ct.append(self.ct)

        return self.lm[:]
    
    def lagrangian(self, cost):
        
        return cost + np.inner(self.lm, self.ct - self.zs) + \
            np.inner(self.mu, self.zs) + \
            self.alpha*np.inner(self.zs, self.zs)/2.0 - \
            self.beta*np.inner(self.lm - self.mu, self.lm - self.mu)/2.0
    
    def see(self):
        
        return (f"lagr = {self.Lg:.4f} | "
            f"lm = {', '.join(f'{val:.4f}' for val in self.lm)} | "
            f"mu = {', '.join(f'{val:.4f}' for val in self.mu)}",)[0]


def homogeneus_boundary(domain, space, dim, rank_dimension):
    """
    Create homgeneous boundary conditions over the whole
    boundary. Used to calculate the velocity field.
    """
    boundary_facets = exterior_facet_indices(domain.topology)
    if rank_dimension == 1:
        u_zero = PETSc.ScalarType(0)
    else: 		
        u_zero = np.zeros(
            rank_dimension,
            dtype = PETSc.ScalarType
        )
    bc = dirichletbc(u_zero, \
            locate_dofs_topological(space, dim-1, boundary_facets), space)
    return [bc]

def homogeneous_dirichlet_point(domain, space, points, rank_dimension, values = None):
    """
    Create homogeneous Dirichlet boundary conditions at specific points 
    for scalar or vector functions in 2D or 3D.
    It works for Lagrange P1!

    Parameters:
    - domain: The computational domain.
    - space: The function space.
    - points: A list of coordinate lists representing boundary points.
    - rank_dimension: The rank of the function (1 for scalar, >1 for vector).

    Returns:
    - A list of Dirichlet boundary conditions.
    """
    dim = domain.geometry.dim

    if values is not None:
        bcs = []
        for p, v in zip(points, values):
            dofs = locate_dofs_geometrical(
                space, 
                lambda x: np.logical_and.reduce([np.isclose(x[i], p[i]) for i in range(dim)])
            )
            bcs.append(dirichletbc(PETSc.ScalarType(v), dofs, space))
        
        return bcs

    if rank_dimension == 1:
        u_zero = PETSc.ScalarType(0)
    else: 		
        u_zero = np.zeros(
            rank_dimension,
            dtype = PETSc.ScalarType
        )
    
    bcs = []
    for p in points:
        dofs = locate_dofs_geometrical(
            space,
            lambda x: np.logical_and.reduce([np.isclose(x[i], p[i]) for i in range(dim)])
        )
        bcs.append(dirichletbc(u_zero, dofs, space))
        
    return bcs

def homogeneous_dirichlet_point_coordinate(domain, space, points, coordinates):
    """
    Create homogeneous Dirichlet boundary conditions at specific points 
    and specific coordinate (either x, y, or z)
    It works for Lagrange P1!

    Parameters:
    - domain: The computational domain.
    - space: The function space.
    - points: A list of coordinate lists representing boundary points.
    - coordinates: A list of the coordinate (0, 1, 2) to apply the condition.

    Returns:
    - A list of Dirichlet boundary conditions.
    """

    dim = domain.geometry.dim
    bcs = []
    for p, c in zip(points, coordinates):
        vertex = locate_entities_boundary(domain, 0, lambda x: np.logical_and.reduce([np.isclose(x[i], p[i]) for i in range(dim)]))
        dofs = locate_dofs_topological(space.sub(c), 0, vertex)
        bcs.append(dirichletbc(PETSc.ScalarType(0.0), dofs, space.sub(c)))
    return bcs

def dirichlet_with_values(domain, space, boundary_tags, mrks_dirichlet, values):
    
    bcs = []
    for mk, v in zip(mrks_dirichlet, values):
        dofs = locate_dofs_topological(
            space,
            domain.geometry.dim - 1,
            boundary_tags.indices[boundary_tags.values == mk]
        )
        bcs.append(dirichletbc(PETSc.ScalarType(v), dofs, space))
    
    return bcs

def homogeneous_dirichlet(domain, space, boundary_tags, mrks_dirichlet, rank_dimension):
    """
    Creates homogeneous Dirichlet boundary conditions 
    over faces of dimension domain.geometry.dim - 1. 
    
    Arguments
        domain:
            The domain or mesh.
        space:
            The function space.
        tags:
            Markers for the facets of the domain.
        mrks_dirichlet (List[int]):
            List of tags corresponding to
            Dirichlet boundary conditions.
        rank_dimension:
            1 scalar, 2 vector 2D, 3 vector 3D,
            (3, 3) rank-2 tensor in 3D, etc

    Returns:
        A list of Dirichlet boundary conditions.
    
    """
    if rank_dimension == 1:
        u_zero = PETSc.ScalarType(0)
    else: 		
        u_zero = np.zeros(
            rank_dimension,
            dtype = PETSc.ScalarType
        )
    
    bcs = []
    for mk in mrks_dirichlet:
        dofs = locate_dofs_topological(
            space,
            domain.geometry.dim - 1,
            boundary_tags.indices[boundary_tags.values == mk]
        )
        bcs.append(dirichletbc(u_zero, dofs, space))
    return bcs

def homogeneous_dirichlet_fun(domain, space, funcs, rank_dimension):
    if rank_dimension == 1:
        u_zero = PETSc.ScalarType(0)
    else: 		
        u_zero = np.zeros(
            rank_dimension,
            dtype = PETSc.ScalarType
        )
    bcs = []
    for f in funcs:
        dofs = locate_dofs_geometrical(space, f)
        bcs.append(dirichletbc(u_zero, dofs, space))
    return bcs

def fun_ds(domain, funcs):
    dim_faces = domain.geometry.dim-1
    facet_indices_list = []
    facet_values_list = []
    for i, f in enumerate(funcs, start=1):
        indices = locate_entities_boundary(
            domain, 
            dim = dim_faces,
            marker = f
        )
        facet_indices_list.append(indices)
        facet_values_list.append(np.full(len(indices), i, dtype = np.int32))
    
    facet_indices = np.concatenate(facet_indices_list)
    facet_values = np.concatenate(facet_values_list)

    facet_tag = meshtags(
        domain,
        dim_faces,
        facet_indices,
        facet_values
    )
    
    indexed_ds = Measure("ds", domain=domain, subdomain_data=facet_tag)
    
    return [indexed_ds(i) for i in range(1, len(funcs) + 1)]

def marked_ds(domain, boundary_tags, marks):
    indexed_ds = Measure('ds', domain = domain, subdomain_data = boundary_tags)
    return [indexed_ds(i) for i in marks]


class LineSearch:
    """
    Line search utility to estimate (with randomness)
    the number of steps and final time 
    based on the norm of the velocity field.
    
    Attributes
    ----------
    t_min: float
        Minimum time.
    t_max: float
        Maximum time.
    s_min: int
        Minimum number of steps.
    s_max: int
        Maximum number of steps.
    factor: float
        Scaling factor used to estimate final time.
    noise: float
        Random variation factor, e.g., 0.05 for 5% variation.
    
    Methods
    -------
    _nbr_steps_func(t: float):
        Power function to estimate the number of steps.
    get(derivative_norm: float):
        Computes the number of steps and final time.
    """

    def __init__(
            self,
            time_bounds: Tuple[float, float],
            step_bounds: Tuple[int, int], 
            factor: float,
            noise: float = 0.05
        ):
        """
        Arguments
        ---------
        time_bounds:  List[float, float]
            Minimum and maximum allowed final times.
        step_bounds: List[int, int]
            Minimum and maximum allowed number of steps.
        factor: float
            Scaling factor used to estimate final time.
        noise: float
            Random variation factor (e.g., 0.05 for 5% variation).
        """
        self.t_min, self.t_max = time_bounds
        self.s_min, self.s_max = step_bounds
        self.factor = factor
        self.noise = noise

    def _nbr_steps_func(self, t: float):
        """
        Power function to estimate the number of steps.

        Arguments
        ---------
        t: float
            Final time.
        
        Returns
        -------
        float: estimated number of steps
        """
        
        scaled_t = (t - self.t_min)/(self.t_max - self.t_min)
        nbr_steps = (self.s_max - self.s_min)*scaled_t**(1.0/6.0) + self.s_min
        
        return nbr_steps

    def get(self, derivative_norm: float):
        """
        Computes the number of steps and final time.
        Reference:
        https://docu.ngsolve.org/latest/i-tutorials/unit-7-optimization/01_Shape_Derivative_Levelset.html

        Arguments
        ---------
        derivative_norm : float
            Norm of the velocity field.

        Returns
        -------
        Tuple[int, float]:
            Number of steps and final time. 
        """

        # To prevent division by zero
        safe_norm = max(derivative_norm, 1e-8) 
        # Final time
        tend = self.factor/safe_norm
        # Randomness
        tend = tend*np.random.uniform(
            1.0 - self.noise, 1.0 + self.noise
        )
        # To guarantee tend in [t_min, t_max] 
        tend = max(self.t_min, min(tend, self.t_max))
        
        steps = self._nbr_steps_func(tend)
        # Randomness
        steps = np.random.normal(steps, self.noise*steps)
        # Conversion to integer
        steps = round(steps)
        # To guarantee steps in [s_min, s_max]
        steps = max(self.s_min, min(steps, self.s_max))

        return steps, tend

def get_rank_dimension(shape):
    if len(shape) == 0:
        return 1
    elif len(shape) == 1:
        return shape[0]
    else:
        print("Tensor rank!")
        return shape
    

def solve_pde(space, pde, phi):

    eqs = pde(phi)
    nbr_eqs = len(eqs)
    solutions = [Function(space) for _ in range(nbr_eqs)]
    for i in range(nbr_eqs):
        solutions[i].name = "u" + str(i)
    
    for (weak_form, bcs), u in zip(eqs, solutions):
        bi, li = system(weak_form)
        basic_solver(form(bi), form(li), bcs, u)
    
    return solutions

def dir_extension_from(comm, domain, space, pde, func, filename):
    """
    Calculate the Dirichlet extension of the solutions of a set
    of partial differential equations.

    Arguments
        comm: Communicator.
        domain: Domain problem.
        space: Space of functions.
        pde: Partial differential equations.
        func: Level set function.
        filename: Name of the xdmf file containing the results.

    Returns
        Dirichlet extension functions.
    """
    
    phi = interpolate(
        funcs = [func],
        to_space = create_space(domain, "CG", 1),
        name = "phi"
    )[0]

    solutions = solve_pde(space, pde, phi)
    extensions = dirichlet_extension(domain, space, solutions)
    
    save_functions(comm, domain,  [phi] + solutions + extensions, filename)
    
    return extensions

def global_scalar(value, comm, postprocess = None):
    
    local_val = assemble_scalar(value)
    global_val = comm.allreduce(local_val, op = MPI.SUM)
    
    return postprocess(global_val) if postprocess else global_val

def global_scalar_list(values, comm):
    
    local_vals = [assemble_scalar(v) for v in values]
    
    return [comm.allreduce(v, op = MPI.SUM) for v in local_vals]


class NonlinearSolverWrapper():
    """
    Wrapper for solving nonlinear problems.
    """
    
    def __init__(self, solver, u, initial):
        """
        Arguments
        ---------
        solver : dolfinx.nls.petsc.NewtonSolver
            Non-linear solver.
        u : dolfinx.fem.Function
            Function to save the solution.
        initial : callable
            Initial guess. A function that can be
            evaluated at mesh points. For instance, 
            a lambda function.
        """
        self.solver = solver
        self.u = u
        self.initial = initial
    
    def solve(self):
        
        self.u.interpolate(self.initial)
        n, converged = self.solver.solve(self.u)
        if not converged:
            print(f"> Newton solver did not converge!")


def create_non_lin_solver(comm, F, bcs, J, u, initial):
    """
    Creates a Newton solver for non-linear problems.
    """
    
    problem = NonlinearProblem(F, u, bcs = bcs, J = J)
    solver = NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True
    solver.maximum_iterations = 10

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    opts[f"{option_prefix}pc_type"] = "hypre"
    opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
    ksp.setFromOptions()
    
    return NonlinearSolverWrapper(solver, u, initial)

def runDP(
        model: Model,
        niter: int,
        reinit_step: int,
        reinit_pars: Tuple[int, float],
        dfactor: float,
        lv_time: Tuple[float, float],
        lv_iter: Tuple[int, int],
        smooth: bool,
        start_to_check: int,
        ctrn_tol: float,
        lgrn_tol: float,
        cost_tol: float,
        prev: int,
        seed: int
    ) -> None:
    
    """
    Implements Data Parallelism.
    """

    start_assembly = MPI.Wtime()

    rein_steps, rein_end = reinit_pars
    min_time, max_time = lv_time
    min_iter, max_iter = lv_iter

    # Constants ===========================
    filename = model.path / f"{res_name}.xdmf"
    stop_flag = False
    
    dim = model.dim
    domain = model.domain
    
    vol = volume(domain, comm)
    nfems = nbr_fems(domain, dim, comm)

    if rank == 0:
        diam2 = get_diam2(dim, vol, nfems)
        np.random.seed(seed)
        lsearch = LineSearch(
            (min_time, max_time),
            (min_iter, max_iter),
            dfactor
        )
        tosave = Save()
    else:
        diam2 = None
    
    diam2 = comm.bcast(diam2, root = 0)
    
    # List of variables
    # -----------------
    # sp_lset : space of level set functions
    # sp_vlty : space of velocity fields
    # phi : level set function
    # tht : velocity field
    # ste_eqs : list of state equations
    # adj_eqs : list of adjoint equations
    # ste_fcs : list of state functions
    # adj_fcs : list of adjoint functions
    # ste_pbs : list of state problems
    # adj_pbs : list of adjoint problems
    # eq_ctrs : list of equality constraints
    # nbr_ste : number of states 
    # nbr_adj : number of adjoints
    # nbr_ctr : number of constraints
    # J : cost functional
    # C : list of constraints
    # L : list of lagrange multipliers
    # S0, S1 : derivative components 

    # Level set function
    sp_lset = create_space(domain, "CG", 1)
    phi = Function(sp_lset)
    phi.name = "phi"
    # Velocity field
    sp_vlty = create_space(domain, "CG", dim)
    tht = Function(sp_vlty)
    tht.name = "tht"

    # State equations/functions
    ste_eqs = model.pde(phi)
    nbr_ste = len(ste_eqs)
    ste_fcs = [Function(model.space) for _ in range(nbr_ste)]
    for i in range(nbr_ste):
        ste_fcs[i].name = "u" + str(i)
    
    # Solvers creation
    ste_pbs = []
    for i in range(nbr_ste):
        ste_problem = ste_eqs[i]
        if len(ste_problem) == 2:
            weak_form, bcs = ste_problem
            bi, li = system(weak_form)
            ste_pbs.append(
                create_solver(form(bi), form(li), bcs, ste_fcs[i])
            )
        else:
            weak_form, bcs, jacobian, seudo_state, ini_func = ste_problem
            compiled_weak_form = compile_form(comm, weak_form)
            compiled_jacobian = compile_form(comm, jacobian)
            form_weak_form = create_form(
                compiled_weak_form,
                [model.space], domain, {},
                {seudo_state: ste_fcs[i], phi: phi}, {}
            )
            form_jacobian = create_form(
                compiled_jacobian,
                [model.space, model.space], domain, {},
                {seudo_state: ste_fcs[i], phi: phi}, {}
            )
            ste_pbs.append(
                create_non_lin_solver(
                    comm, form_weak_form, bcs,
                    form_jacobian, ste_fcs[i], ini_func
                )
            )    

    # Adjoint equations/functions
    adj_eqs = model.adjoint(phi, ste_fcs)
    nbr_adj = len(adj_eqs)
    if nbr_adj > 0:
        adj_fcs = [Function(model.space) for _ in range(nbr_adj)]
        for i in range(nbr_adj):
            adj_fcs[i].name = "p" + str(i)
        
        # Solvers creation
        adj_pbs = []
        for i in range(nbr_adj):
            adj_problem = adj_eqs[i]
            if len(adj_problem) == 2:
                weak_form, bcs = adj_problem
                bi, li = system(weak_form)
                adj_pbs.append(
                    create_solver(form(bi), form(li), bcs, adj_fcs[i])
                )
            else:
                weak_form, bcs, jacobian, seudo_adjoint, ini_func = adj_problem
                compiled_weak_form = compile_form(comm, weak_form)
                compiled_jacobian = compile_form(comm, jacobian)
                form_weak_form = create_form(
                    compiled_weak_form,
                    [model.space], domain, {},
                    {seudo_adjoint: adj_fcs[i], phi: phi}, {}
                )
                form_jacobian = create_form(
                    compiled_jacobian,
                    [model.space, model.space], domain, {},
                    {seudo_adjoint: adj_fcs[i], phi: phi}, {}
                )
                adj_pbs.append(
                    create_non_lin_solver(
                        comm, form_weak_form, bcs,
                        form_jacobian, adj_fcs[i], ini_func
                    )
                )    
    else:
        adj_fcs = []

    # Cost functional
    J = form(model.cost(phi, ste_fcs))
    # Derivative components
    S0_cts, S1_cts = model.derivative(phi, ste_fcs, adj_fcs)
    S0 = S0_cts[0]
    S1 = S1_cts[0]
    # Equality constraints
    eq_ctrs = model.constraint(phi, ste_fcs)
    nbr_ctr = len(eq_ctrs)
    if nbr_ctr > 0:
        # Compilation of the constraints
        C = [form(c) for c in eq_ctrs]
        # Lagrange multipliers
        L = [const(domain, 0) for _ in range(nbr_ctr)]
        # Creation of the derivatives components
        for i in range(nbr_ctr):
            S0 += L[i]*S0_cts[1][i]
            S1 += L[i]*S1_cts[1][i]
    # Derivative norm
    nDJ = form((model.bilinear_form(tht, tht))[0])
    # To calculate the velocity field
    cls_vlty = Velocity(
        dim, domain, sp_vlty,
        model.bilinear_form, S0, S1
    )
    # To calculate the level set function
    cls_lset = Level(
        domain, sp_lset, phi, tht, diam2, smooth
    )
    # Reinicialization
    if reinit_step:
        cls_rein = Reinit(
            domain, sp_lset, phi, diam2
        )
        
    local_assembly = MPI.Wtime() - start_assembly
    max_assembly = comm.allreduce(local_assembly, op = MPI.MAX)

    # Iteration i = 0 =======
    start_solve = MPI.Wtime()

    phi.interpolate(model.get_initial_level())
    
    [p.solve() for p in ste_pbs]
    comm.Barrier()

    cost = global_scalar(J, comm)
    
    if nbr_ctr > 0:
        ctrs = global_scalar_list(C, comm)
        if rank == 0:
            cls_meth = PPL(nbr_ctr, cost, ctrs)
    
    if nbr_adj > 0:
        [p.solve() for p in adj_pbs]
        comm.barrier()

    cls_vlty.run(tht)
    nder = global_scalar(nDJ, comm, np.sqrt)

    if rank == 0:
        lset_steps, lset_end = lsearch.get(nder)
    else:
        lset_steps, lset_end = None, None
    lset_steps = comm.bcast(lset_steps, root = 0)
    lset_end = comm.bcast(lset_end, root = 0)

    # print ==============================================
    if rank == 0:
        print("> Iterations:")
        if nbr_ctr > 0:
            print0(0, cost, ctrs, nder, 0, cls_meth.see())
        else:
            print1(0, cost, nder, 0)
        tosave.add(cost, nder)
    # ====================================================

    with XDMFFile(comm, filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi, 0)
        for f in ste_fcs: xdmf.write_function(f, 0)
        for f in adj_fcs: xdmf.write_function(f, 0)
        xdmf.write_function(tht, 0)

        for iter in range(1, niter + 1):
            
            cls_lset.run(phi, lset_steps, lset_end)
            # Reinitialization
            if reinit_step and iter > start_to_check:
                if iter%reinit_step == 0:
                    cls_rein.run(phi, rein_steps, rein_end)			
            
            [p.solve() for p in ste_pbs]
            comm.barrier()

            cost = global_scalar(J, comm)
            
            if nbr_ctr > 0:
                ctrs = global_scalar_list(C, comm)
                if rank == 0:
                    lm = cls_meth.run(cost, ctrs)
                else:
                    lm = None
                lm = comm.bcast(lm, root = 0)
                for i in range(nbr_ctr):
                    L[i].value = lm[i]
            
            if nbr_adj > 0:
                [p.solve() for p in adj_pbs]
                comm.barrier()
            
            cls_vlty.run(tht)
            nder = global_scalar(nDJ, comm, np.sqrt)
            
            if rank == 0:
                lset_steps, lset_end = lsearch.get(nder)
            else:
                lset_steps, lset_end = None, None
            lset_steps = comm.bcast(lset_steps, root = 0)
            lset_end = comm.bcast(lset_end, root = 0)

            xdmf.write_function(phi, iter)
            for f in ste_fcs: xdmf.write_function(f, iter)
            for f in adj_fcs: xdmf.write_function(f, iter)
            xdmf.write_function(tht, iter)
            
            if rank == 0:
                if nbr_ctr > 0:
                    print0(iter, cost, ctrs, nder, lset_steps, cls_meth.see())
                else:
                    print1(iter, cost, nder, lset_steps)
                tosave.add(cost, nder)

                if iter > start_to_check:
                    if nbr_ctr > 0:
                        ctrn_errs = [c - 1.0 for c in ctrs]
                        lgrn_last = cls_meth.list_Lg[-1]
                        lgrn_errs = [l - lgrn_last for l in cls_meth.list_Lg[-prev:-1]]
                        cond1 = Norm(ctrn_errs, np.inf) < ctrn_tol
                        cond2 = Norm(lgrn_errs, np.inf) < lgrn_tol*abs(lgrn_last)
                        stop_flag = cond1 and cond2
                    else:
                        cost_last = tosave.cost[-1]
                        cost_diff = [j - cost_last for j in tosave.cost[-prev:-1]]
                        stop_flag = Norm(cost_diff, np.inf) < cost_tol*abs(cost_last)

            stop_flag = comm.bcast(stop_flag, root = 0)

            if stop_flag:
                if rank == 0: print("> Stopping condition reached!")
                break

    local_solve = MPI.Wtime() - start_solve
    max_solve = comm.allreduce(local_solve, op = MPI.MAX)

    if rank == 0:
        if nbr_ctr > 0:
            tosave.add_ppl(cls_meth)
        tosave.add_times(max_assembly, max_solve)
        tosave.save(model.path)
        print(f"> Assembly time = {max_assembly} s")
        print(f"> Resolution time = {max_solve} s")


def runTP(
        model: Model,
        initial_guess: Tuple[Any, ...],
        niter: int = 100,
        save_path: Path = Path(""),
        reinit_step: int = 4,
        reinit_pars: Tuple[int, float] = (8, 1e-2),
        dfactor: float = 1e-2,
        lv_time: Tuple[float, float] = (1e-3, 1.0),
        lv_iter: Tuple[int, int] = (8, 14),
        smooth: bool = False,
        start_to_check: int = 30,
        ctrn_tol: float = 1e-2,
        lgrn_tol: float = 1e-2,
        cost_tol: float = 1e-2,
        prev: int = 10,
        seed: int = 26
    ) -> None:
    
    """
    Implements Task Parallelism.
    """

    start_assembly = MPI.Wtime()

    rein_steps, rein_end = reinit_pars
    min_time, max_time = lv_time
    min_iter, max_iter = lv_iter

    # Constants ===========================
    filename = save_path / f"{res_name}.xdmf"
    stop_flag = False
    
    dim = model.dim
    domain = model.domain
    
    if rank == 0:
        vol = volume(domain, MPI.COMM_SELF)
        nfems = nbr_fems(domain, dim, MPI.COMM_SELF)
        diam2 = get_diam2(dim, vol, nfems)
        inilset = InitialLevel(*initial_guess)
        np.random.seed(seed)
        lsearch = LineSearch(
            (min_time, max_time),
            (min_iter, max_iter),
            dfactor
        )
        tosave = Save()
    else:
        vol = None
        nfems = None
        diam2 = None
        inilset = None

    vol = comm.bcast(vol, root = 0)
    nfems = comm.bcast(nfems, root = 0)
    diam2 = comm.bcast(diam2, root = 0)
    inilset = comm.bcast(inilset, root = 0)
    
    # Level set function
    sp_lset = create_space(domain, "CG", 1)
    phi = Function(sp_lset)
    phi.name = "phi"
    # Velocity field
    sp_vlty = create_space(domain, "CG", dim)
    tht = Function(sp_vlty)
    tht.name = "tht"

    # State equations/functions/problems
    ste_eqs = model.pde(phi)
    nbr_ste = len(ste_eqs)
    ste_fcs = [Function(model.space) for _ in range(nbr_ste)]
    for i in range(nbr_ste):
        ste_fcs[i].name = "u" + str(i)

    weak_form, bcs = ste_eqs[rank]
    bi, li = system(weak_form)
    ste_pb = create_solver(
        form(bi), form(li), bcs, ste_fcs[rank]
    )

    # Adjoint equations/functions/problems
    adj_eqs = model.adjoint(phi, ste_fcs)
    nbr_adj = len(adj_eqs)
    if nbr_adj > 0:
        adj_fcs = [Function(model.space) for _ in range(nbr_adj)]
        for i in range(nbr_adj):
            adj_fcs[i].name = "p" + str(i)
        
        weak_form, bcs = adj_eqs[rank]
        bi, li = system(weak_form)
        adj_pb = create_solver(
            form(bi), form(li), bcs, adj_fcs[rank]
        )
    else:
        adj_fcs = []

    if rank == 0:
        # Cost functional
        J = form(model.cost(phi, ste_fcs))
        # Derivative components
        S0_cts, S1_cts = model.derivative(phi, ste_fcs, adj_fcs)
        S0 = S0_cts[0]
        S1 = S1_cts[0]
        # Equality constraints
        eq_ctrs = model.constraint(phi, ste_fcs)
        nbr_ctr = len(eq_ctrs)
        if nbr_ctr > 0:
            # Compilation of the constraints
            C = [form(c) for c in eq_ctrs]
            # Lagrange multipliers
            L = [const(domain, 0) for _ in range(nbr_ctr)]
            # Creation of the derivatives components
            for i in range(nbr_ctr):
                S0 += L[i]*S0_cts[1][i]
                S1 += L[i]*S1_cts[1][i]
        # Derivative norm
        nDJ = form((model.bilinear_form(tht, tht))[0])
        # To calculate the velocity field
        cls_vlty = Velocity(
            dim, domain, sp_vlty,
            model.bilinear_form, S0, S1
        )
        # To calculate the level set function
        cls_lset = Level(
            domain, sp_lset, phi, tht, diam2, smooth
        )
        # Reinitialization
        if reinit_step:
            cls_rein = Reinit(
                domain, sp_lset, phi, diam2
            )
    
    local_assembly = MPI.Wtime() - start_assembly
    max_assembly = comm.allreduce(local_assembly, op = MPI.MAX)

    # Iteration i = 0 =======
    start_solve = MPI.Wtime()

    # --------------------------------------------
    if rank == 0:
        phi.interpolate(inilset.func)
        phi_vls = phi.x.array[:]
    else:
        phi_vls = None
    phi.x.array[:] = comm.bcast(phi_vls, root = 0)
    #---------------------------------------------
    
    ste_pb.solve()
    comm.barrier()

    # ---------------------------------------------------
    # Alternative form, without ste_vls -----------------
    # ---------------------------------------------------
    # for i in range(size):
    # 	if rank == i:
    # 		u_i = ste_fcs[i].x.array[:]
    # 	else:
    # 		u_i = None	
    # 	ste_fcs[i].x.array[:] = comm.bcast(u_i, root = i)
    # ---------------------------------------------------

    # ------------------------------------------------
    ste_vls = comm.allgather(ste_fcs[rank].x.array[:])
    for i in range(size):
        if i == rank: continue
        ste_fcs[i].x.array[:] = ste_vls[i]
    # ------------------------------------------------

    comm.barrier()

    if rank == 0:
        cost = global_scalar(J, MPI.COMM_SELF)
        if nbr_ctr > 0:
            ctrs = global_scalar_list(C, MPI.COMM_SELF)
            cls_meth = PPL(nbr_ctr, cost, ctrs)

    if nbr_adj > 0:
        adj_pb.solve()
        comm.barrier()

        # ---------------------------------------------------
        # Alternative form, without adj_vls -----------------
        # ---------------------------------------------------
        # for i in range(size):
        # 	if rank == i:
        # 		p_i = adj_fcs[i].x.array[:]
        # 	else:
        # 		p_i = None
        # 	adj_fcs[i].x.array[:] = comm.bcast(p_i, root = i)
        # ---------------------------------------------------

        # ------------------------------------------------
        adj_vls = comm.allgather(adj_fcs[rank].x.array[:])
        for i in range(size):
            if i == rank: continue
            adj_fcs[i].x.array[:] = adj_vls[i]
        # ------------------------------------------------
        comm.barrier()

    if rank == 0:
        cls_vlty.run(tht)
        nder = global_scalar(nDJ, MPI.COMM_SELF, np.sqrt)
        lset_steps, lset_end = lsearch.get(nder)
    
    # print ==============================================
    if rank == 0:
        print("> Iterations:")
        if nbr_ctr > 0:
            print0(0, cost, ctrs, nder, 0, cls_meth.see())
        else:
            print1(0, cost, nder, 0)
        tosave.add(cost, nder)
    # ====================================================

    if rank == 0:
        xdmf = XDMFFile(MPI.COMM_SELF, filename, "w")
        xdmf.write_mesh(domain)
        xdmf.write_function(phi, 0)
        for f in ste_fcs: xdmf.write_function(f, 0)
        for f in adj_fcs: xdmf.write_function(f, 0)
        xdmf.write_function(tht, 0)

    for iter in range(1, niter + 1):
        # -------------------------------------------------
        if rank == 0:
            cls_lset.run(phi, lset_steps, lset_end)
            # Reinitialization
            if reinit_step and iter > start_to_check:
                if iter%reinit_step == 0:
                    cls_rein.run(phi, rein_steps, rein_end)
            phi_vls = phi.x.array[:]
        else:
            phi_vls = None
        phi.x.array[:] = comm.bcast(phi_vls, root = 0)
        # -------------------------------------------------
            
        ste_pb.solve()
        comm.barrier()

        # ------------------------------------------------
        ste_vls = comm.allgather(ste_fcs[rank].x.array[:])
        for i in range(size):
            if i == rank: continue
            ste_fcs[i].x.array[:] = ste_vls[i]
        # ------------------------------------------------

        comm.barrier()

        if rank == 0:
            cost = global_scalar(J, MPI.COMM_SELF)
            if nbr_ctr > 0:
                ctrs = global_scalar_list(C, MPI.COMM_SELF)
                lm = cls_meth.run(cost, ctrs)
                for i in range(nbr_ctr):
                    L[i].value = lm[i]

        if nbr_adj > 0:
            adj_pb.solve()
            comm.barrier()

            # ------------------------------------------------
            adj_vls = comm.allgather(adj_fcs[rank].x.array[:])
            for i in range(size):
                if i == rank: continue
                adj_fcs[i].x.array[:] = adj_vls[i]
            # ------------------------------------------------
            comm.barrier()
        
        if rank == 0:
            cls_vlty.run(tht)
            nder = global_scalar(nDJ, MPI.COMM_SELF, np.sqrt)
            lset_steps, lset_end = lsearch.get(nder)

        if rank == 0:
            xdmf.write_function(phi, iter)
            for f in ste_fcs: xdmf.write_function(f, iter)
            for f in adj_fcs: xdmf.write_function(f, iter)
            xdmf.write_function(tht, iter)

        if rank == 0:
            if nbr_ctr > 0:
                print0(iter, cost, ctrs, nder, lset_steps, cls_meth.see())
            else:
                print1(iter, cost, nder, lset_steps)
            tosave.add(cost, nder)

            if iter > start_to_check:
                if nbr_ctr > 0:
                    ctrn_errs = [c - 1.0 for c in ctrs]
                    lgrn_last = cls_meth.list_Lg[-1]
                    lgrn_errs = [l - lgrn_last for l in cls_meth.list_Lg[-prev:-1]]
                    cond1 = Norm(ctrn_errs, np.inf) < ctrn_tol
                    cond2 = Norm(lgrn_errs, np.inf) < lgrn_tol*abs(lgrn_last)
                    stop_flag = cond1 and cond2
                else:
                    cost_last = tosave.cost[-1]
                    cost_diff = [j - cost_last for j in tosave.cost[-prev:-1]]
                    stop_flag = Norm(cost_diff, np.inf) < cost_tol*abs(cost_last)

        stop_flag = comm.bcast(stop_flag, root = 0)

        if stop_flag:
            if rank == 0: print("> Stopping condition reached!")
            break
    
    if rank == 0:
        xdmf.close()

    local_solve = MPI.Wtime() - start_solve
    max_solve = comm.allreduce(local_solve, op = MPI.MAX)

    if rank == 0:
        if nbr_ctr > 0:
            tosave.add_ppl(cls_meth)
        tosave.add_times(max_assembly, max_solve)
        tosave.save(save_path)
        print(f"> Assembly time = {max_assembly} s")
        print(f"> Resolution time = {max_solve} s")


def runMP(
        sub_comm: MPI.Comm,
        model: Model,
        initial_guess: Tuple[Any, ...],
        niter: int = 100,
        save_path: Path = Path(""),
        reinit_step: int = 4,
        reinit_pars: Tuple[int, float] = (8, 1e-2),
        dfactor: float = 1e-2,
        lv_time: Tuple[float, float] = (1e-3, 1.0),
        lv_iter: Tuple[int, int] = (8, 14),
        smooth: bool = False,
        start_to_check: int = 30,
        ctrn_tol: float = 1e-2,
        lgrn_tol: float = 1e-2,
        cost_tol: float = 1e-2,
        prev: int = 10,
        seed: int = 26
    ) -> None:
    
    """
    Implements Mix Parallelism.
    """
    
    start_assembly = MPI.Wtime()
    
    rein_steps, rein_end = reinit_pars
    min_time, max_time = lv_time
    min_iter, max_iter = lv_iter

    group_size = sub_comm.size
    nbr_groups = size // group_size
    color = rank * nbr_groups // size
    sub_rank = sub_comm.rank
    
    # Constants ===========================
    filename = save_path / f"{res_name}.xdmf"
    stop_flag = False
    
    dim = model.dim
    domain = model.domain
    
    if color == 0:
        vol = volume(domain, sub_comm)
        nfems = nbr_fems(domain, dim, sub_comm)
    
    if rank == 0:
        diam2 = get_diam2(dim, vol, nfems)
        inilset = InitialLevel(*initial_guess)
        np.random.seed(seed)
        lsearch = LineSearch(
            (min_time, max_time),
            (min_iter, max_iter),
            dfactor
        )
        tosave = Save()
    else:
        diam2 = None
        inilset = None

    diam2 = comm.bcast(diam2, root = 0)
    inilset = comm.bcast(inilset, root = 0)
    
    # Level set function
    sp_lset = create_space(domain, "CG", 1)
    phi = Function(sp_lset)
    phi.name = "phi"
    # Velocity field
    sp_vlty = create_space(domain, "CG", dim)
    tht = Function(sp_vlty)
    tht.name = "tht"

    # State equations/functions/problems
    ste_eqs = model.pde(phi)
    nbr_ste = len(ste_eqs)
    ste_fcs = [Function(model.space) for _ in range(nbr_ste)]
    for i in range(nbr_ste):
        ste_fcs[i].name = "u" + str(i)
    weak_form, bcs = ste_eqs[color]
    bi, li = system(weak_form)
    ste_pb = create_solver(
        form(bi), form(li), bcs, ste_fcs[color]
    )

    # Adjoint equations/functions/problems
    adj_eqs = model.adjoint(phi, ste_fcs)
    nbr_adj = len(adj_eqs)
    if nbr_adj > 0:
        adj_fcs = [Function(model.space) for _ in range(nbr_adj)]
        for i in range(nbr_adj):
            adj_fcs[i].name = "p" + str(i)
        weak_form, bcs = adj_eqs[color]
        bi, li = system(weak_form)
        adj_pb = create_solver(
            form(bi), form(li), bcs, adj_fcs[color]
        )
    else:
        adj_fcs = []

    if color == 0:
        # Cost functional
        J = form(model.cost(phi, ste_fcs))
        # Derivative components
        S0_cts, S1_cts = model.derivative(phi, ste_fcs, adj_fcs)
        S0 = S0_cts[0]
        S1 = S1_cts[0]
        # Equality constraints
        eq_ctrs = model.constraint(phi, ste_fcs)
        nbr_ctr = len(eq_ctrs)
        if nbr_ctr > 0:
            # Compilation of the constraints
            C = [form(c) for c in eq_ctrs]
            # Lagrange multipliers
            L = [const(domain, 0) for _ in range(nbr_ctr)]
            # Creation of the derivatives components
            for i in range(nbr_ctr):
                S0 += L[i]*S0_cts[1][i]
                S1 += L[i]*S1_cts[1][i]
        # Derivative norm
        nDJ = form((model.bilinear_form(tht, tht))[0])
        # To calculate the velocity field
        cls_vlty = Velocity(
            dim, domain, sp_vlty,
            model.bilinear_form, S0, S1
        )
        # To calculate the level set function
        cls_lset = Level(
            domain, sp_lset, phi, tht, diam2, smooth
        )
        # Reinicialization
        if reinit_step:
            cls_rein = Reinit(
                domain, sp_lset, phi, diam2
            )
    
    local_assembly = MPI.Wtime() - start_assembly
    max_assembly = comm.allreduce(local_assembly, op = MPI.MAX)

    # Iteration i = 0 =======
    start_solve = MPI.Wtime()

    # ------------------------------------
    if color == 0:
        phi.interpolate(inilset.func)
        phi_vls_loc = phi.x.array[:]
    else:
        # Colect None if color is not 0
        phi_vls_loc = None
    phi_vls = comm.allgather(phi_vls_loc)
    if color > 0:
        phi.x.array[:] = phi_vls[sub_rank]
    # ------------------------------------

    comm.barrier()
    ste_pb.solve()
    comm.barrier()
    
    # ----------------------------------------------------------
    ste_vls = comm.allgather(ste_fcs[color].x.array[:])
    for i in range(nbr_groups):
        if i == color: continue
        ste_fcs[i].x.array[:] = ste_vls[i*group_size + sub_rank]
    # ----------------------------------------------------------

    comm.barrier()

    if color == 0:
        cost = global_scalar(J, sub_comm)
        if nbr_ctr > 0:
            ctrs = global_scalar_list(C, sub_comm)
            if rank == 0:
                cls_meth = PPL(nbr_ctr, cost, ctrs)

    if nbr_adj > 0:
        adj_pb.solve()
        comm.barrier()

        # ----------------------------------------------------------
        adj_vls = comm.allgather(adj_fcs[color].x.array[:])
        for i in range(nbr_groups):
            if i == color: continue
            adj_fcs[i].x.array[:] = adj_vls[i*group_size + sub_rank]
        # ----------------------------------------------------------
        comm.barrier()
    
    if color == 0:
        cls_vlty.run(tht)
        nder = global_scalar(nDJ, sub_comm, np.sqrt)
    
        if rank == 0:
            lset_steps, lset_end = lsearch.get(nder)
        else:
            lset_steps, lset_end = None, None
        lset_steps = sub_comm.bcast(lset_steps, root = 0)
        lset_end = sub_comm.bcast(lset_end, root = 0)

    # print ==============================================
    if rank == 0:
        print("> Iterations:")
        if nbr_ctr > 0:
            print0(0, cost, ctrs, nder, 0, cls_meth.see())
        else:
            print1(0, cost, nder, 0)
        tosave.add(cost, nder)
    # ====================================================
    
    if color == 0:
        xdmf = XDMFFile(sub_comm, filename, "w")
        xdmf.write_mesh(domain)
        xdmf.write_function(phi, 0)
        for f in ste_fcs: xdmf.write_function(f, 0)
        for f in adj_fcs: xdmf.write_function(f, 0)
        xdmf.write_function(tht, 0)
    
    for iter in range(1, niter + 1):
        # -------------------------------------------------
        if color == 0:
            cls_lset.run(phi, lset_steps, lset_end)
            # Reinitialization
            if reinit_step and iter > start_to_check:
                if iter%reinit_step == 0:
                    cls_rein.run(phi, rein_steps, rein_end)
            phi_vls_loc = phi.x.array[:]
        else:
            phi_vls_loc = None
        phi_vls = comm.allgather(phi_vls_loc)
        if color > 0:
            phi.x.array[:] = phi_vls[sub_rank]
        # -------------------------------------------------
        
        comm.barrier()
        ste_pb.solve()
        comm.barrier()

        # ----------------------------------------------------------
        ste_vls = comm.allgather(ste_fcs[color].x.array[:])
        for i in range(nbr_groups):
            if i == color: continue
            ste_fcs[i].x.array[:] = ste_vls[i*group_size + sub_rank]
        # ----------------------------------------------------------

        comm.barrier()

        if color == 0:
            cost = global_scalar(J, sub_comm)
            if nbr_ctr > 0:
                ctrs = global_scalar_list(C, sub_comm)
                if rank == 0:
                    lm = cls_meth.run(cost, ctrs)
                else:
                    lm = None
                lm = sub_comm.bcast(lm, root = 0)
                for i in range(nbr_ctr):
                    L[i].value = lm[i]

        if nbr_adj > 0:
            adj_pb.solve()
            comm.barrier()

            # ----------------------------------------------------------
            adj_vls = comm.allgather(adj_fcs[color].x.array[:])
            for i in range(nbr_groups):
                if i == color: continue
                adj_fcs[i].x.array[:] = adj_vls[i*group_size + sub_rank]
            # ----------------------------------------------------------
            comm.barrier()

        if color == 0:
            cls_vlty.run(tht)
            nder = global_scalar(nDJ, sub_comm, np.sqrt)
        
            if rank == 0:
                lset_steps, lset_end = lsearch.get(nder)
            else:
                lset_steps, lset_end = None, None
            lset_steps = sub_comm.bcast(lset_steps, root = 0)
            lset_end = sub_comm.bcast(lset_end, root = 0)

        if color == 0:
            xdmf.write_function(phi, iter)
            for f in ste_fcs: xdmf.write_function(f, iter)
            for f in adj_fcs: xdmf.write_function(f, iter)
            xdmf.write_function(tht, iter)

        if rank == 0:
            if nbr_ctr > 0:
                print0(iter, cost, ctrs, nder, lset_steps, cls_meth.see())
            else:
                print1(iter, cost, nder, lset_steps)
            tosave.add(cost, nder)

            if iter > start_to_check:
                if nbr_ctr > 0:
                    ctrn_errs = [c - 1.0 for c in ctrs]
                    lgrn_last = cls_meth.list_Lg[-1]
                    lgrn_errs = [l - lgrn_last for l in cls_meth.list_Lg[-prev:-1]]
                    cond1 = Norm(ctrn_errs, np.inf) < ctrn_tol
                    cond2 = Norm(lgrn_errs, np.inf) < lgrn_tol*abs(lgrn_last)
                    stop_flag = cond1 and cond2
                else:
                    cost_last = tosave.cost[-1]
                    cost_diff = [j - cost_last for j in tosave.cost[-prev:-1]]
                    stop_flag = Norm(cost_diff, np.inf) < cost_tol*abs(cost_last)
        
        stop_flag = comm.bcast(stop_flag, root = 0)
    
        if stop_flag:
            if rank == 0: print("> Stopping condition reached!")
            break

    if color == 0:
        xdmf.close()
    
    local_solve = MPI.Wtime() - start_solve
    max_solve = comm.allreduce(local_solve, op = MPI.MAX)

    if rank == 0:
        if nbr_ctr > 0:
            tosave.add_ppl(cls_meth)
        tosave.add_times(max_assembly, max_solve)
        tosave.save(save_path)
        print(f"> Assembly time = {max_assembly} s")
        print(f"> Resolution time = {max_solve} s")
