import distributed as dib

from models import (
    Logistic,
    Compliance, 
    CompliancePlus,
    InverseElasticity,
    Heat,
    HeatPlus,
    HeatMultiple
)

import numpy as np
from mpi4py import MPI
from pathlib import Path
from dolfinx.fem import functionspace

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
comm_self = MPI.COMM_SELF

"""
test_01 : Symmetric cantilever 2D (Data Parallelism)
test_02 : Symmetric cantilever 3D (Data Parallelism)
test_03 : Cantilever with two loads (Data Parallelism)
test_04 : Cantilever with two loads (Task Parallelism)
test_05 : Cantilever with two loads (Mix Parallelism)
test_06 : Elasticity inverse problem (Data Parallelism)
test_07 : Elasticity inverse problem (Task Parallelism)
test_08 : Elasticity inverse problem (Mix Parallelism)
test_09 : Heat conduction problem 1 (Data Parallelism)
test_10 : Heat conduction problem 2 (Data Parallelism)
test_11 : Heat conduction problem 3 (Data Parallelism)
test_12 : Heat conduction with two sinks (Data Parallelism)
test_13 : Heat conduction with two sinks (Task Parallelism)
test_14 : Heat conduction with two sinks (Mix Parallelism)
"""

def test_00():
    test_path = Path("../results/t00/")
    dim = 2
    rank_dim = 1
    mesh_size = 0.012
    
    vertices = np.array([
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0)
    ])
    
    output = dib.create_domain_2d_DP(
        vertices, [], mesh_size,
        path = test_path,
        plot = True
    )
    
    domain, nbr_tri, boundary_tags = output

    # Space for the PDE solution
    space = dib.create_space(domain, "CG", rank_dim)
    
    # Create the model
    md = Logistic(
        dim, domain, space
    )
    md.ini_func = lambda x: 1 + 0.2 * np.sin(6 * np.pi * x[0]) * np.sin(6 * np.pi * x[1])
    
    # Initial guess: centers and radii
    centers = []
    
    centers += [(0, 0.25*i) for i in range(5)]
    centers += [(0.25, 0.25*i) for i in range(5)]
    centers += [(0.5, 0.25*i) for i in range(5)]
    centers += [(0.75, 0.25*i) for i in range(5)]
    centers += [(1, 0.25*i) for i in range(5)]
    
    centers = np.array(centers)
    radii = np.repeat(0.1, centers.shape[0])
    
    dib.save_initial(
        comm, (centers, radii, -1),
        domain, test_path / "initial.xdmf"
    )

    dib.run(
        model = md,
        initial_guess = (centers, radii, -1),
        niter = 300,
        save_path = test_path,
        reinit_step = 6,
        ctrn_tol = 1e-2,
        lv_iter = (10, 16),
        dfactor = 1e-1,
        smooth = True
    )

def test_01():
    """
    Run: mpirun -np <nbr of processes> python test.py 01
    For instance: mpirun -np 2 python test.py 01
    """

    test_name = "Symmetric cantilever 2D (Data Parallelism)"
    test_path = Path("../results/t01/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.015
    
    vertices = np.array([
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 0.45),
        (2.0, 0.55),
        (2.0, 1.0),
        (0.0, 1.0)
    ])
    
    dir_idx, dir_mkr = [6], 1
    neu_idx, neu_mkr = [3], 2
    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx, neu_mkr, "neu")
    ]

    # Create gmsh domain for Data Parallelism
    output = dib.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )
    
    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
        
    # Space for the PDE solution
    space = dib.create_space(domain, "CG", rank_dim)
    
    # Dirichlet condition
    dirichlet_bcs = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir_mkr], rank_dim
    )

    # Boundary to force application 
    ds_g = dib.marked_ds(
        domain,
        boundary_tags,
        [neu_mkr]
    )
    
    area = 1.0
    g = (0.0, -2.0)
    
    # Create the model
    md = Compliance(
        dim, domain, space,
        g, ds_g[0],
        dirichlet_bcs, area
    )

    @dib.region_of(domain)
    def sub_domain(x):
        # 0.42 < x[1] < 0.58
        # 1.95 < x[0]
        ineqs = [
            x[1] - 0.42,
            0.58 - x[1],
            x[0] - 1.95
		]
        return ineqs
    
    md.sub = [sub_domain.expression()]

    # Initial guess: centers and radii
    centers = np.array([
        (0.3, 0.0), (1.0, 0.0),
        (1.7, 0.0), (2.0, 0.0),
        (0.3, 1.0), (1.0, 1.0),
        (1.7, 1.0), (2.0, 1.0),
        (0.3, 0.5), (1.0, 0.5),
        (1.7, 0.5), (0.0, 0.25),
        (0.65, 0.25), (1.35, 0.25),
        (2.0, 0.35), (0.0, 0.75),
        (0.65, 0.75), (1.35, 0.75),
        (2.0, 0.65), (0.0, 0.5)
    ])
    
    centers = np.array(centers)
    radii = np.repeat(0.1, centers.shape[0])
    
    dib.save_initial(
        comm, (centers, radii),
        domain, test_path / "initial.xdmf"
    )
    
    dib.runDP(
        model = md,
        initial_guess = (centers, radii),
        niter = 100,
        save_path = test_path,
        reinit_step = 4,
        ctrn_tol = 1e-3,
        dfactor = 1e-1,
        smooth = True
    )


def test_02():
    """
    Run: mpirun -np <nbr of processes> python test.py 02
    """

    test_name = "Symmetric cantilever 3D (Data Parallelism)"
    test_path = Path("../results/t02/")
    dim = 3
    rank_dim = 3
    mesh_size = 60
    
    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
    
    # Create a dolfinx domain
    from dolfinx.mesh import create_box
    domain = create_box(
        comm,
        [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]],
        [2*mesh_size, mesh_size, mesh_size]
    )
    
    dib.all_connectivities(domain)
    dib.save_domain(
        comm, domain, test_path / "domain.xdmf"
    )

    # Space
    space = dib.create_space(domain, "CG", rank_dim)

    # marker functions
    def boundary_dirichlet(x):
        return np.isclose(x[0], 0.0)
    
    def boundary_neumann(x):
        in_plane = np.isclose(x[0], 2.0)
        in_square = np.maximum(np.abs(x[1] - 0.5), np.abs(x[2] - 0.5)) <= 0.1 + 1e-6
        return in_plane & in_square
    
    # Dirichlet conditions
    dirichlet_bcs = dib.homogeneous_dirichlet_fun(
        domain, space, [boundary_dirichlet], rank_dim
    )
    # Boundary to force application 
    ds_g = dib.fun_ds(domain, [boundary_neumann])
    volume, g = 0.8, [0.0, 0.0, -4.0]
    # Create the model
    md = Compliance(
        dim, domain, space,
        g, ds_g[0], dirichlet_bcs, volume
    )

    @dib.region_of(domain)
    def sub_domain(x):
        ineqs = [
            x[1] - 0.375,
            0.625 - x[1],
            x[2] - 0.375,
            0.625 - x[2],
            x[0] - 1.95
		]
        return ineqs

    md.sub = [sub_domain.expression()]

    centers = np.array([
        (2., .25, .25), (2., .75, .25), (2., .25, .75), (2., .75, .75),
        (2., 0., 0.), (2., 0., 1.), (2., 1., 0.), (2., 1., 1.),
        (2., .5, 0.), (2., .5, 1.), (2., 0., .5,), (2., 1., .5),
        (1.7, 0., 0.), (1.7, 0., 1.), (1.7, 1., 0.), (1.7, 1., 1.),
        (1.7, .5, .5), (1.7, 0, 0.5), (1.7, 1., 0.5), (1.7, 0.5, 0.), (1.7, 0.5, 1.),
        (1.35, 0.25, 0.25), (1.35, 0.75, 0.25), (1.35, 0.25, 0.75), (1.35, 0.75, 0.75), 
        (1., 0., 0.), (1., 0., 1.), (1., 1., 0.), (1., 1., 1.),
        (1., .5, .5), (1.0, 0, 0.5), (1.0, 1., 0.5), (1.0, 0.5, 0.), (1.0, 0.5, 1.),
        (0.65, 0.25, 0.25), (0.65, 0.75, 0.25), (0.65, 0.25, 0.75), (0.65, 0.75, 0.75),
        (0.3, 0., 0.), (0.3, 0., 1.), (0.3, 1., 0.), (0.3, 1., 1.),
        (0.3, .5, .5), (0.3, 0, 0.5), (0.3, 1., 0.5), (0.3, 0.5, 0.), (0.3, 0.5, 1.),
        (0., .5, 0.), (0., .5, 1.), (0., 0., .5,), (0., 1., .5), (0., .5, .5),
        (0.0, 0.25, 0.25), (0.0, 0.75, 0.25), (0.0, 0.25, 0.75), (0.0, 0.75, 0.75)
    ])
    radii = np.repeat(0.1, centers.shape[0])
    dib.save_initial(
        comm, (centers, radii),
        domain, test_path / "initial.xdmf")

    #Run data parallelism
    dib.runDP(
        model = md,
        initial_guess = (centers, radii),
        niter = 200,
        save_path = test_path,
        reinit_step = 4,
        ctrn_tol = 1e-3,
        dfactor = 1e-1
    )


def test_03():
    """
    Run: mpirun -np <nbr of processes> python test.py 03
    For instance: mpirun -np 2 python test.py 03
    """

    test_name = "Multiple load cases (Data Parallelism)"
    test_path = Path("../results/t03/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.012
    
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.1],
        [1.0, 0.9],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    
    dir_idx, dir_mkr = [6], 1
    neu_idx_bot, neu_mkr_bot = [2], 2
    neu_idx_top, neu_mkr_top = [4], 3

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx_bot, neu_mkr_bot, "neu_bot"),
        (neu_idx_top, neu_mkr_top, "neu_top")
    ]

    # Create gmsh domain for Data Parallelism
    output = dib.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )
    
    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = dib.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir_mkr], rank_dim
    )
    # Boundary to force application 
    ds_g = dib.marked_ds(
        domain, boundary_tags,
        [neu_mkr_bot, neu_mkr_top]
    )
    area = 0.5
    g = [(0.0, -2.0), (0.0, -2.0)]
    # Create the model
    md = CompliancePlus(
        dim, domain, space,
        g, ds_g, dirichlet_bcs, area
    )

    # Initial guess: centers and radii
    centers = []
    centers += [(i*0.2, 0.25) for i in range(5)]
    centers += [(0.1 + i*0.2, 0.5) for i in range(5)]
    centers += [(i*0.2, 0.75) for i in range(5)]
    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])
    dib.save_initial(
        comm, (centers, radii),
        domain, test_path / "initial.xdmf"
    )

    # Run Data Parallelism
    dib.runDP(
        model = md,
        initial_guess = (centers, radii),
        save_path = test_path,
        niter = 100,
        reinit_step = 4,
        ctrn_tol = 1e-3,
        dfactor = 1e-1,
        smooth = True
    )


def test_04():
    """
    Run: mpirun -np 2 python test.py 04
    """
    
    # Verification
    task_nbr = 2
    if size != task_nbr:
        print(f"Nbr of processes must be = {task_nbr}")
        return

    test_name = "Multiple load cases (Task Parallelism)"
    test_path = Path("../results/t04/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.012
    
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.1],
        [1.0, 0.9],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    
    dir_idx, dir_mkr = [6], 1
    neu_idx_bot, neu_mkr_bot = [2], 2
    neu_idx_top, neu_mkr_top = [4], 3

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx_bot, neu_mkr_bot, "neu_bot"),
        (neu_idx_top, neu_mkr_top, "neu_top")
    ]

    # Create gmsh domain for Task Parallelism
    output = dib.create_domain_2d_TP(
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )
    
    domain, nbr_tri, boundary_tags = output
    
    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
    
    # Space
    space = dib.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir_mkr], rank_dim
    )
    # Boundary to force application
    ds_g = dib.marked_ds(
        domain,
        boundary_tags,
        [neu_mkr_bot, neu_mkr_top]
    )
    area = 0.5
    g = [(0.0, -2.0), (0.0, -2.0)]
    # Create the model
    md = CompliancePlus(
        dim, domain, space,
        g, ds_g, dirichlet_bcs, area
    )

    # Initial guess: centers and radii
    centers = []
    centers += [(i*0.2, 0.25) for i in range(5)]
    centers += [(0.1 + i*0.2, 0.5) for i in range(5)]
    centers += [(i*0.2, 0.75) for i in range(5)]
    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])
    if rank == 0:
        dib.save_initial(
            comm_self, (centers, radii),
            domain, test_path/"initial.xdmf"
        )
    
    # Run Task Parallelism
    dib.runTP(
        model = md,
        initial_guess = (centers, radii),
        save_path = test_path,
        niter = 100,
        reinit_step = 4,
        ctrn_tol = 1e-3,
        dfactor = 1e-1,
        smooth = True
    )


def test_05():
    """
    Run: mpirun -np <2n> python test.py 05
    For instance: mpirun -np 4 python test.py 05
    """
    
    # Verification
    nbr_groups = 2
    if size%nbr_groups != 0:
        print(f"Nbr of processes must be divisible by {nbr_groups}")
        return

    # Subcommunicators
    color = rank * nbr_groups // size
    sub_comm = comm.Split(color, rank)

    test_name = "Multiple load cases (Mix Parallelism)"
    test_path = Path("../results/t05/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.012
    
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.1],
        [1.0, 0.9],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    
    dir_idx, dir_mkr = [6], 1
    neu_idx_bot, neu_mkr_bot = [2], 2
    neu_idx_top, neu_mkr_top = [4], 3

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx_bot, neu_mkr_bot, "neu_bot"),
        (neu_idx_top, neu_mkr_top, "neu_top")
    ]

    # Create gmsh domain for Mix Parallelism
    output = dib.create_domain_2d_MP(
        sub_comm, color,
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )
    
    domain, nbr_tri, boundary_tags = output
        
    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
    
    # Space
    space = dib.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir_mkr], rank_dim
    )
    # Boundary to force application
    ds_g = dib.marked_ds(
        domain,
        boundary_tags,
        [neu_mkr_bot, neu_mkr_top]
    )
    area = 0.5
    g = [(0.0, -2.0), (0.0, -2.0)]
    # Create the model
    md = CompliancePlus(
        dim, domain, space,
        g, ds_g, dirichlet_bcs, area
    )

    # Initial guess: centers and radii
    centers = []
    centers += [(i*0.2, 0.25) for i in range(5)]
    centers += [(0.1+i*0.2, 0.5) for i in range(5)]
    centers += [(i*0.2, 0.75) for i in range(5)]
    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])
    if color == 0:
        dib.save_initial(
            sub_comm, (centers, radii),
            domain, test_path / "initial.xdmf"
        )
    
    # Run Mix Parallelism
    dib.runMP(
        sub_comm = sub_comm,
        model = md,
        initial_guess = (centers, radii),
        save_path = test_path,
        niter = 100,
        reinit_step = 4,
        ctrn_tol = 1e-3,
        dfactor = 1e-1,
        smooth = True
    )


def test_06():
    """
    Run: mpirun -np <nbr of processes> python test.py 06
    For instance: mpirun -np 3 python test.py 06
    """

    test_name = "Elasticity inverse problem (Data Parallelism)"
    test_path = Path("../results/t06/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.015
    
    # Data generation

    def semi_ellipse(a, b, eps, npts):
        """
        Coordinates of a ellipse
        crossing the x-axis
        """
        t_ = np.arcsin((b - eps)/b)
        t = np.linspace(-t_, np.pi + t_, npts)
        x = a*np.cos(t)
        y = b*np.sin(t) + (b - eps)
        return x, y
    
    npts = 80 # npts % 4 = 0
    part = npts//4
    
    vertices = np.column_stack(
        semi_ellipse(0.75, 0.5, 0.2, npts)
    )
    
    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part//2 + 1), 2
    neu_idxA, neu_mkrA = part//2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part//2 + np.arange(part + 1, 2*part + 1), 4
    neu_idxC, neu_mkrC = part//2 + np.arange(2*part + 1, 3*part + 1), 5
    bL_idx, bL_mkr = np.arange(part//2 + 3*part + 1, npts), 6
    
    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (bR_idx, bR_mkr, "bRight"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (bL_idx, bL_mkr, "bLeft"),
    ]
    
    def bean_curve(n):
        """
        Coordinates of a bean to be used
        as a subdomain.
        """
        ampl = 0.5
        x0 = 0.0
        y0 = 0.2
        angle = np.pi/6.0
        t = np.linspace(0, np.pi, n, endpoint = False)
        r = ampl*(np.cos(t)**3 + np.sin(t)**3)
        x = r*np.cos(t) + x0
        y = r*np.sin(t) + y0
        cos_ = np.cos(angle)
        sin_ = np.sin(angle)
        x_rot = cos_*x - sin_*y
        y_rot = sin_*x + cos_*y
        return x_rot, y_rot
    
    subdomain = np.column_stack(bean_curve(60))

    if rank == 0:
        np.save(
            test_path / "subdomain.npy",
            subdomain
        )
    
    filename = test_path / "domain0.msh"

    # Create the gmsh domain0.msh
    nbr_tri0 = dib.build_gmsh_model_2d(
        vertices, boundary_parts, 0.6*mesh_size,
        curve = subdomain,
        filename = filename,
        plot = True
    )

    # Read the domain0
    output = dib.read_gmsh(
        filename, comm, dim = 2
    )
    
    domain0, _, boundary_tags = output

    # Set all connectivities on domain0
    dib.all_connectivities(domain0)

    # Space for data generation
    space0 = dib.create_space(domain0, "CG", rank_dim)
    # Forces
    forces = [
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0)
    ]

    # Dirichlet boundary conditions
    dirbc_partial = dib.homogeneous_dirichlet(
        domain0,
        space0,
        boundary_tags,
        [dir_mkr],
        rank_dim
    )
    
    dirbc_total = dib.homogeneus_boundary(
        domain0,
        space0,
        dim,
        rank_dim
    )

    # Create measures to apply Neumman condition
    ds_parts = dib.marked_ds(
        domain0,
        boundary_tags,
        [bR_mkr, neu_mkrA, neu_mkrB, neu_mkrC, bL_mkr]
    )

    # Measures for force application
    ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]] 
    
    # Measure for adjoint problem
    ds1 = sum(ds_parts[1:], start = ds_parts[0])

    # Instance for data generation
    # We need the method pde0
    md0 = InverseElasticity(
        dim, domain0, space0,
        forces, ds_forces, ds1,
        dirbc_partial, dirbc_total
    )

    # Function that defines
    # the level set function
    # to generate the data
    def beam_equation(x):
        """
        Implicit equation :
        ((x-x0)^2 + (y-y0)^2)^2 = a((x-x0)^3 + (y-y0)^3) 
        """
        ampl = 0.5
        x0 = 0.0
        y0 = 0.2
        angle = np.pi/6.0
        cos_ = np.cos(angle)
        sin_ = np.sin(angle)
        # inverse rotation
        x_irot = cos_*x[0] + sin_*x[1] - x0
        y_irot = -sin_*x[0] + cos_*x[1] - y0
        left_part = (x_irot**2 + y_irot**2)**2
        right_part = ampl*(x_irot**3 + y_irot**3)
        values = left_part - right_part
        return np.log(25.0*values + 1.0)
    
    # Dirichlet extensions
    extensions = dib.dir_extension_from(
        comm, domain0, space0,
        md0.pde0, beam_equation,
        test_path / "extensions.xdmf"
    )

    npts = 80 # npts % 4 = 0
    part = npts//4
    
    vertices = np.column_stack(
        semi_ellipse(0.75, 0.5, 0.2, npts)
    )
    
    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part//2 + 1), 2
    neu_idxA, neu_mkrA = part//2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part//2 + np.arange(part + 1, 2*part + 1), 4
    neu_idxC, neu_mkrC = part//2 + np.arange(2*part + 1, 3*part + 1), 5
    bL_idx, bL_mkr = np.arange(part//2 + 3*part + 1, npts), 6
    
    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (bR_idx, bR_mkr, "bRight"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (bL_idx, bL_mkr, "bLeft"),
    ]

    # Create gmsh domain for Data Parallelism
    output = dib.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )
    
    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
        print(f"> Nbr of triangles for data generation = {nbr_tri0}")

    # Space
    space = dib.create_space(domain, "CG", rank_dim)
    # Dirichlet boundary conditions
    dirbc_partial = dib.homogeneous_dirichlet(
        domain,
        space,
        boundary_tags,
        [dir_mkr],
        rank_dim
    )
    dirbc_total = dib.homogeneus_boundary(
        domain,
        space,
        dim,
        rank_dim
    )
    # Boundary to apply Neumann conditions
    ds_parts = dib.marked_ds(
        domain,
        boundary_tags,
        [bR_mkr, neu_mkrA, neu_mkrB, neu_mkrC, bL_mkr]
    )

    ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]] 
    ds1 = sum(ds_parts[1:], start = ds_parts[0])
    forces = [
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0)
    ]
    # Create the model
    md = InverseElasticity(
        dim, domain, space,
        forces, ds_forces, ds1,
        dirbc_partial, dirbc_total
    )

    # Space for interpolation (degree = 2)
    g_space = dib.create_space(
        domain, "CG", rank_dim, degree = 2
    )
    # Interpolation between different spaces
    # from different domains
    g_funcs = dib.space_interpolation(
        from_space = space0,
        funcs = extensions,
        to_space = g_space
    )

    # To save as P1 functions
    g_space_1 = dib.create_space(domain, "CG", rank_dim)
    g_funcs_1 = dib.interpolate(
        funcs = g_funcs,
        to_space = g_space_1,
        name = "g"
    )
    dib.save_functions(
        comm, domain,
        g_funcs_1, test_path / "g1.xdmf"
    )

    md.gs = g_funcs

    # Initial guess: centers and radii
    centers = np.array([(0.0, 0.4)])
    radii = np.array([0.15])
    dib.save_initial(
        comm, (centers, radii, -1),
        domain, test_path / "initial.xdmf"
    )
    
    # Run Data Parallelism
    dib.runDP(
        model = md,
        initial_guess = (centers, radii, -1),
        save_path = test_path,
        niter = 200,
        dfactor = 1e-1,
        cost_tol = 1e-1
    )


def test_07():
    """
    Run: mpirun -np 6 python test.py 07
    """

    # Verification
    task_nbr = 6
    if size != task_nbr:
        print(f"Nbr of processes must be = {task_nbr}")
        return

    test_name = "Elasticity inverse problem (Task Parallelism)"
    test_path = Path("../results/t07/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.015

    # Data generation

    def semi_ellipse(a, b, eps, npts):
        """
        Coordinates of a ellipse
        crossing the x-axis
        """
        t_ = np.arcsin((b - eps)/b)
        t = np.linspace(-t_, np.pi + t_, npts)
        x = a*np.cos(t)
        y = b*np.sin(t) + (b - eps)
        return x, y
    
    npts = 80 # npts % 4 = 0
    part = npts//4
    
    vertices = np.column_stack(
        semi_ellipse(0.75, 0.5, 0.2, npts)
    )
    
    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part//2 + 1), 2
    neu_idxA, neu_mkrA = part//2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part//2 + np.arange(part + 1, 2*part + 1), 4
    neu_idxC, neu_mkrC = part//2 + np.arange(2*part + 1, 3*part + 1), 5
    bL_idx, bL_mkr = np.arange(part//2 + 3*part + 1, npts), 6
    
    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (bR_idx, bR_mkr, "bRight"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (bL_idx, bL_mkr, "bLeft"),
    ]
    
    def bean_curve(n):
        """
        Coordinates of a bean to be used
        as a subdomain.
        """
        ampl = 0.5
        x0 = 0.0
        y0 = 0.2
        angle = np.pi/6.0
        t = np.linspace(0, np.pi, n, endpoint = False)
        r = ampl*(np.cos(t)**3 + np.sin(t)**3)
        x = r*np.cos(t) + x0
        y = r*np.sin(t) + y0
        cos_ = np.cos(angle)
        sin_ = np.sin(angle)
        x_rot = cos_*x - sin_*y
        y_rot = sin_*x + cos_*y
        return x_rot, y_rot
    
    subdomain = np.column_stack(bean_curve(60))

    if rank == 0:
        np.save(
            test_path / "subdomain.npy",
            subdomain
        )

    filename = test_path / "domain0.msh"

    nbr_tri0 = dib.build_gmsh_model_2d(
        vertices, boundary_parts, 0.6*mesh_size,
        curve = subdomain,
        filename = filename,
        plot = True
    )

    if rank == 0:
        # Read the domain0 in rank = 0
        output = dib.read_gmsh(
            filename, comm_self, 2
        )

        domain0, _, boundary_tags = output
        dib.all_connectivities(domain0)

        # Space defined in rank = 0
        space0 = dib.create_space(domain0, "CG", rank_dim)
        # three forces
        forces = [
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0)
        ]

        # Dirichlet boundary conditions
        dirbc_partial = dib.homogeneous_dirichlet(
            domain0,
            space0,
            boundary_tags,
            [dir_mkr],
            rank_dim
        )
        
        dirbc_total = dib.homogeneus_boundary(
            domain0,
            space0,
            dim,
            rank_dim
        )

        # Measures
        ds_parts = dib.marked_ds(
            domain0,
            boundary_tags,
            [bR_mkr, neu_mkrA, neu_mkrB, neu_mkrC, bL_mkr]
        )

        ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]] 
        ds1 = sum(ds_parts[1:], start = ds_parts[0])

        md0 = InverseElasticity(
            dim, domain0, space0,
            forces, ds_forces, ds1,
            dirbc_partial, dirbc_total
        )

        def beam_equation(x):
            """
            Implicit equation :
            ((x-x0)^2 + (y-y0)^2)^2 = a((x-x0)^3 + (y-y0)^3) 
            """
            ampl = 0.5
            x0 = 0.0
            y0 = 0.2
            angle = np.pi/6.0
            cos_ = np.cos(angle)
            sin_ = np.sin(angle)
            # inverse rotation
            x_irot = cos_*x[0] + sin_*x[1] - x0
            y_irot = -sin_*x[0] + cos_*x[1] - y0
            vals = (x_irot**2 + y_irot**2)**2 - ampl*(x_irot**3 + y_irot**3)
            return np.log(25*vals + 1.0)
        
        extensions = dib.dir_extension_from(
            comm_self, domain0, space0, md0.pde0, beam_equation,
            test_path / "extensions.xdmf"
        )

    npts = 80 # npts % 4 = 0
    part = npts//4
    
    vertices = np.column_stack(
        semi_ellipse(0.75, 0.5, 0.2, npts)
    )
    
    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part//2 + 1), 2
    neu_idxA, neu_mkrA = part//2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part//2 + np.arange(part + 1, 2*part + 1), 4
    neu_idxC, neu_mkrC = part//2 + np.arange(2*part + 1, 3*part + 1), 5
    bL_idx, bL_mkr = np.arange(part//2 + 3*part + 1, npts), 6
    
    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (bR_idx, bR_mkr, "bRight"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (bL_idx, bL_mkr, "bLeft"),
    ]

    # Create gmsh domain for Task Parallelism
    output = dib.create_domain_2d_TP(
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )
    
    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
        print(f"> Nbr of triangles for data generation = {nbr_tri0}")

    # Space
    space = dib.create_space(domain, "CG", rank_dim)
    # Dirichlet boundary conditions
    dirbc_partial = dib.homogeneous_dirichlet(
        domain,
        space,
        boundary_tags,
        [dir_mkr],
        rank_dim
    )
    dirbc_total = dib.homogeneus_boundary(
        domain,
        space,
        dim,
        rank_dim
    )
    # Boundary to force application
    ds_parts = dib.marked_ds(
        domain,
        boundary_tags,
        [bR_mkr, neu_mkrA, neu_mkrB, neu_mkrC, bL_mkr]
    )

    ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]] 
    ds1 = sum(ds_parts[1:], start = ds_parts[0])
    forces = [
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0)
    ]

    # Create the model
    md = InverseElasticity(
        dim, domain, space,
        forces, ds_forces, ds1,
        dirbc_partial, dirbc_total
    )

    g_space = dib.create_space(
        domain, "CG", rank_dim, degree = 2
    )
    if rank == 0:
        # Interpolation
        g_funcs = dib.space_interpolation(
            from_space = space0,
            funcs = extensions,
            to_space = g_space
        )

        # Save
        g_space_1 = dib.create_space(domain, "CG", rank_dim)
        g_funcs_1 = dib.interpolate(
            funcs = g_funcs,
            to_space = g_space_1,
            name = "g"
        )
        dib.save_functions(
            comm_self, domain,
            g_funcs_1, test_path / "g1.xdmf"
        )

        g_values = np.vstack([g.x.array[:] for g in g_funcs])
    
    else:
        g_values = None

    g_values = comm.bcast(g_values, root = 0)

    md.gs = dib.get_funcs_from(g_space, g_values)

    # Initial guess: centers and radii
    centers = np.array([(0.0, 0.4)])
    radii = np.array([0.15])
    if rank == 0:
        dib.save_initial(
            comm_self, (centers, radii, -1),
            domain, test_path / "initial.xdmf"
        )
    
    # Run Task Parallelism
    dib.runTP(
        model = md,
        initial_guess = (centers, radii, -1),
        save_path = test_path,
        niter = 200,
        dfactor = 1e-1,
        cost_tol = 1e-1
    )


def test_08():
    """
    Run: mpirun -np <6n> python test.py 08
    For instance: mpirun -np 12 python test.py 08
    """

    # Verification
    nbr_groups = 6
    if size%nbr_groups != 0:
        print(f"Nbr of processes must be divisible by {nbr_groups}")
        return

    # Subcommunicators
    color = rank * nbr_groups // size
    sub_comm = comm.Split(color, rank)

    test_name = "Elasticity inverse problem (Mix Parallelism)"
    test_path = Path("../results/t08/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.015

    # Data generation

    def semi_ellipse(a, b, eps, npts):
        """
        Coordinates of a ellipse
        crossing the x-axis
        """
        t_ = np.arcsin((b - eps)/b)
        t = np.linspace(-t_, np.pi + t_, npts)
        x = a*np.cos(t)
        y = b*np.sin(t) + (b - eps)
        return x, y
    
    npts = 80 # npts % 4 = 0
    part = npts//4
    
    vertices = np.column_stack(
        semi_ellipse(0.75, 0.5, 0.2, npts)
    )
    
    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part//2 + 1), 2
    neu_idxA, neu_mkrA = part//2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part//2 + np.arange(part + 1, 2*part + 1), 4
    neu_idxC, neu_mkrC = part//2 + np.arange(2*part + 1, 3*part + 1), 5
    bL_idx, bL_mkr = np.arange(part//2 + 3*part + 1, npts), 6
    
    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (bR_idx, bR_mkr, "bRight"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (bL_idx, bL_mkr, "bLeft"),
    ]
    
    def bean_curve(n):
        """
        Coordinates of a bean to be used
        as a subdomain.
        """
        ampl = 0.5
        x0 = 0.0
        y0 = 0.2
        angle = np.pi/6.0
        t = np.linspace(0, np.pi, n, endpoint = False)
        r = ampl*(np.cos(t)**3 + np.sin(t)**3)
        x = r*np.cos(t) + x0
        y = r*np.sin(t) + y0
        cos_ = np.cos(angle)
        sin_ = np.sin(angle)
        x_rot = cos_*x - sin_*y
        y_rot = sin_*x + cos_*y
        return x_rot, y_rot
    
    subdomain = np.column_stack(bean_curve(60))

    if rank == 0:
        np.save(
            test_path / "subdomain.npy",
            subdomain
        )

    filename = test_path / "domain0.msh"

    nbr_tri0 = dib.build_gmsh_model_2d(
        vertices, boundary_parts, 0.6*mesh_size,
        curve = subdomain,
        filename = filename,
        plot = True
    )

    if color == 0:
        # Read the domain0 in rank = 0
        output = dib.read_gmsh(
            filename, sub_comm, 2
        )

        domain0, _, boundary_tags = output
        dib.all_connectivities(domain0)

        # Space defined in rank = 0
        space0 = dib.create_space(domain0, "CG", rank_dim)
        # three forces
        forces = [
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0)
        ]

        # Dirichlet boundary conditions
        dirbc_partial = dib.homogeneous_dirichlet(
            domain0,
            space0,
            boundary_tags,
            [dir_mkr],
            rank_dim
        )
        
        dirbc_total = dib.homogeneus_boundary(
            domain0,
            space0,
            dim,
            rank_dim
        )

        # Measures
        ds_parts = dib.marked_ds(
            domain0,
            boundary_tags,
            [bR_mkr, neu_mkrA, neu_mkrB, neu_mkrC, bL_mkr]
        )

        ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]] 
        ds1 = sum(ds_parts[1:], start = ds_parts[0])

        md0 = InverseElasticity(
            dim, domain0, space0,
            forces, ds_forces, ds1,
            dirbc_partial, dirbc_total
        )

        def beam_equation(x):
            """
            Implicit equation :
            ((x-x0)^2 + (y-y0)^2)^2 = a((x-x0)^3 + (y-y0)^3) 
            """
            ampl = 0.5
            x0 = 0.0
            y0 = 0.2
            angle = np.pi/6.0
            cos_ = np.cos(angle)
            sin_ = np.sin(angle)
            # inverse rotation
            x_irot = cos_*x[0] + sin_*x[1] - x0
            y_irot = -sin_*x[0] + cos_*x[1] - y0
            left_part = (x_irot**2 + y_irot**2)**2
            right_part = ampl*(x_irot**3 + y_irot**3)
            values = left_part - right_part
            return np.log(25.0*values + 1.0)
            
        extensions = dib.dir_extension_from(
            sub_comm, domain0, space0, md0.pde0, beam_equation,
            test_path / "extensions.xdmf"
        )

    npts = 80 # npts % 4 = 0
    part = npts//4
    
    vertices = np.column_stack(
        semi_ellipse(0.75, 0.5, 0.2, npts)
    )
    
    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part//2 + 1), 2
    neu_idxA, neu_mkrA = part//2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part//2 + np.arange(part + 1, 2*part + 1), 4
    neu_idxC, neu_mkrC = part//2 + np.arange(2*part + 1, 3*part + 1), 5
    bL_idx, bL_mkr = np.arange(part//2 + 3*part + 1, npts), 6
    
    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (bR_idx, bR_mkr, "bRight"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (bL_idx, bL_mkr, "bLeft"),
    ]

    # Create gmsh domain for Mix Parallelism
    output = dib.create_domain_2d_MP(
        sub_comm, color,
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )
    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
        print(f"> Nbr of triangles for data generation = {nbr_tri0}")

    # Space
    space = dib.create_space(domain, "CG", rank_dim)
    # Dirichlet boundary conditions
    dirbc_partial = dib.homogeneous_dirichlet(
        domain,
        space,
        boundary_tags,
        [dir_mkr],
        rank_dim
    )
    dirbc_total = dib.homogeneus_boundary(
        domain,
        space,
        dim,
        rank_dim
    )
    # Boundary to force application
    ds_parts = dib.marked_ds(
        domain,
        boundary_tags,
        [bR_mkr, neu_mkrA, neu_mkrB, neu_mkrC, bL_mkr]
    )

    ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]] 
    ds1 = sum(ds_parts[1:], start = ds_parts[0])
    forces = [
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0)
    ]
    # Create the model
    md = InverseElasticity(
        dim, domain, space,
        forces, ds_forces, ds1,
        dirbc_partial, dirbc_total
    )

    g_space = dib.create_space(
        domain, "CG", rank_dim, degree = 2
    )
    
    if color == 0:
        # Interpolation
        g_funcs = dib.space_interpolation(
            from_space = space0,
            funcs = extensions,
            to_space = g_space
        )

        # Save
        g_space_1 = dib.create_space(domain, "CG", rank_dim)
        g_funcs_1 = dib.interpolate(
            funcs = g_funcs,
            to_space = g_space_1,
            name = "g"
        )
        dib.save_functions(
            sub_comm, domain,
            g_funcs_1, test_path/"g1.xdmf"
        )

        g_values_loc = np.vstack([g.x.array[:] for g in g_funcs])
    
    else:
        g_values_loc = None

    g_values = comm.allgather(g_values_loc)
    
    md.gs = dib.get_funcs_from(g_space, g_values[sub_comm.rank])

    # Initial guess: centers and radii
    centers = np.array([(0.0, 0.4)])
    radii = np.array([0.15])
    if color == 0:
        dib.save_initial(
            sub_comm, (centers, radii, -1),
            domain, test_path / "initial.xdmf"
        )
    
    # Run Mix Parallelism
    dib.runMP(
        sub_comm,
        model = md,
        initial_guess = (centers, radii, -1),
        save_path = test_path,
        niter = 200,
        dfactor = 1e-1,
        cost_tol = 1e-1
    )


def test_09():
    """
    Run: mpirun -np <nbr of processes> python test.py 09
    """

    test_name = "Heat conduction (Data Parallelism)"
    test_path = Path("../results/t09/")
    dim = 2
    rank_dim = 1
    mesh_size = 5e-3

    vertices = np.array([
        [0.0, 0.0],
        [0.4, 0.0],
        [0.6, 0.0],
        [1.0, 0.0],
        [1., 1.0],
        [0.0, 1.0]
    ])

    dir_idx, dir_mkr = [2], 1
    
    boundary_parts = [
        (dir_idx, dir_mkr, "dir")
    ]

    # Create gmsh domain for Data Parallelism
    output = dib.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = dib.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir_mkr], rank_dim
    )
    area = 0.25
    # Create the model
    md = Heat(
        dim, domain, space, dirichlet_bcs, area
    )

    @dib.region_of(domain)
    def sub_domain(x):
        ineqs = (
            x[0] - 0.3,
            0.7 - x[0],
            0.05 - x[1]
        )
        return ineqs
    
    md.sub = [sub_domain.expression()]

    centers = [
        (0.1, 0.0),
        (0.2, 0.0),
        (0.3, 0.0),
        (0.7, 0.0),
        (0.8, 0.0),
        (0.9, 0.0)
    ]
    
    centers += [(0.0, i*0.1) for i in range(10)]
    centers += [(1.0, i*0.1) for i in range(10)]
    centers += [(i*0.1, 1.0) for i in range(11)]

    centers += [(0.25, i*0.2) for i in range(1, 5)]
    centers += [(0.5, i*0.2 + 0.1) for i in range(1, 5)]
    centers += [(0.75, i*0.2) for i in range(1, 5)]
    
    centers = np.array(centers)
    radii = np.repeat(0.08, 49)

    dib.save_initial(
        comm, (centers, radii),
        domain, test_path / "initial.xdmf"
    )

    #Run Data Parallelism
    dib.runDP(
        model = md,
        initial_guess = (centers, radii),
        save_path = test_path,
        niter = 200,
        dfactor = 1e-2,
        ctrn_tol = 1e-3,
        lgrn_tol = 1e-2
    )


def test_10():
    """
    Run: mpirun -np <nbr of processes> python test.py 10
    """

    test_name = "Heat conduction (Data Parallelism)"
    test_path = Path("../results/t10/")
    dim = 2
    rank_dim = 1
    mesh_size = 5e-3

    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    dir1_idx, dir1_mkr = [1], 1
    dir2_idx, dir2_mkr = [2], 2
    dir3_idx, dir3_mkr = [3], 3
    dir4_idx, dir4_mkr = [4], 4
    
    boundary_parts = [
        (dir1_idx, dir1_mkr, "dir1"),
        (dir2_idx, dir2_mkr, "dir2"),
        (dir3_idx, dir3_mkr, "dir3"),
        (dir4_idx, dir4_mkr, "dir4")
    ]

    # Create gmsh domain for Data Parallelism
    output = dib.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = dib.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = dib.homogeneus_boundary(
        domain, space, dim, rank_dim
    )
    area = 0.5
    # Create the model
    md = Heat(
        dim, domain, space, dirichlet_bcs, area
    )

    @dib.region_of(domain)
    def sub_domain1(x):
        return (0.05 - x[1], )
    
    @dib.region_of(domain)
    def sub_domain2(x):
        return (x[0] - 0.95, )
    
    @dib.region_of(domain)
    def sub_domain3(x):
        return (x[1] - 0.95, )

    @dib.region_of(domain)
    def sub_domain4(x):
        return (0.05 - x[0], )
    md.sub = [
        sub_domain1.expression(),
        sub_domain2.expression(),
        sub_domain3.expression(),
        sub_domain4.expression()
    ]
    centers = []
    centers += [(0.3, i*0.2 + 0.1) for i in range(1, 4)]
    centers += [(0.5, i*0.2 + 0.1) for i in range(1, 4)]
    centers += [(0.7, i*0.2 + 0.1) for i in range(1, 4)]
    centers += [(0.2, i*0.2 + 0.2) for i in range(4)]
    centers += [(0.4, i*0.2 + 0.2) for i in range(4)]
    centers += [(0.6, i*0.2 + 0.2) for i in range(4)]
    centers += [(0.8, i*0.2 + 0.2) for i in range(4)]
    
    centers = np.array(centers)
    radii = np.repeat(0.05, centers.shape[0])

    dib.save_initial(
        comm, (centers, radii),
        domain, test_path / "initial.xdmf"
    )
    
    dib.runDP(
        model = md,
        initial_guess = (centers, radii),
        save_path = test_path,
        niter = 200,
        dfactor = 1e-1,
        ctrn_tol = 1e-3,
        lgrn_tol = 5e-2,
        lv_iter = (10, 18)
    )


def test_11():
    """
    Run: mpirun -np <nbr of processes> python test.py 11
    """

    test_name = "Heat conduction (Data Parallelism)"
    test_path = Path("../results/t11/")
    dim = 2
    rank_dim = 1
    mesh_size = 5e-3

    vertices = np.array([
        [0.0, 0.0],
        [0.4, 0.0],
        [0.6, 0.0],
        [1.0, 0.0],
        [1.0, 0.4],
        [1.0, 0.6],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [5], 2
    
    boundary_parts = [
        (dir1_idx, dir1_mkr, "dir1"),
        (dir2_idx, dir2_mkr, "dir2")
    ]

    # Create gmsh domain for Data Parallelism
    output = dib.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = dib.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir1_mkr, dir2_mkr], rank_dim
    )

    area = 0.4
    # Create the model
    md = Heat(
        dim, domain, space, dirichlet_bcs, area
    )

    @dib.region_of(domain)
    def sub_domain1(x):
        # 0.3 < x < 0.7
        # y < 0.05
        ineqs = (
            x[0] - 0.3,
            0.7 - x[0],
            0.05 - x[1]
        )
        return ineqs
    
    @dib.region_of(domain)
    def sub_domain2(x):
        # 0.95 < x
        # 0.3 < y < 0.7
        ineqs = (
            x[1] - 0.3,
            0.7 - x[1],
            x[0] - 0.95
        )
        return ineqs
    
    md.sub = [
        sub_domain1.expression(),
        sub_domain2.expression()
    ]

    centers_b = [
        (0.1, 0.0),
        (0.2, 0.0),
        (0.3, 0.0),
        (0.7, 0.0),
        (0.8, 0.0),
        (0.9, 0.0),
        (1.0, 0.0),
        (1.0, 0.1),
        (1.0, 0.2),
        (1.0, 0.3),
        (1.0, 0.7),
        (1.0, 0.8),
        (1.0, 0.9),
    ]

    centers_b += [(0.0, i*0.1) for i in range(11)]
    centers_b += [(i*0.1, 1.0) for i in range(1, 11)]
    centers_b += [(i*0.1, 1.0) for i in range(1, 11)]
    
    centers_i = [(0.1, i*0.2 + 0.3) for i in range(4)]
    centers_i += [(0.2, i*0.2 + 0.2) for i in range(4)]
    centers_i += [(0.3, i*0.2 + 0.3) for i in range(4)]
    centers_i += [(0.4, i*0.2 + 0.2) for i in range(4)]
    centers_i += [(0.5, i*0.2 + 0.3) for i in range(4)]
    centers_i += [(0.6, i*0.2 + 0.2) for i in range(4)]
    centers_i += [(0.7, i*0.2 + 0.3) for i in range(4)]
    centers_i += [(0.8, i*0.2 + 0.2) for i in range(4)]
    
    centers = np.array(centers_b + centers_i)
    radii_b = np.repeat(0.08, len(centers_b))
    radii_i = np.repeat(0.05, len(centers_i))
    radii = np.concatenate((radii_b, radii_i))
    dib.save_initial(
        comm, (centers, radii),
        domain, test_path / "initial.xdmf"
    )

    dib.runDP(
        model = md,
        initial_guess = (centers, radii),
        save_path = test_path,
        niter = 250,
        dfactor = 1e-2,
        ctrn_tol = 1e-3,
        lgrn_tol = 1e-2
    )


def test_12():
    """
    Run: mpirun -np <nbr of processes> python test.py 11
    """

    test_name = "Heat conduction (Data Parallelism)"
    test_path = Path("../results/t12/")
    dim = 2
    rank_dim = 1
    mesh_size = 5e-3

    vertices = np.array([
        [0.0, 0.0],
        [0.4, 0.0],
        [0.6, 0.0],
        [1.0, 0.0],
        [1.0, 0.4],
        [1.0, 0.6],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [5], 2
    
    boundary_parts = [
        (dir1_idx, dir1_mkr, "dir1"),
        (dir2_idx, dir2_mkr, "dir2")
    ]

    # Create gmsh domain for Data Parallelism
    output = dib.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = dib.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir1_mkr, dir2_mkr], rank_dim
    )

    area = 0.4
    # Create the model
    md = HeatPlus(
        dim, domain, space, dirichlet_bcs, area
    )

    @dib.region_of(domain)
    def sub_domain1(x):
        # 0.3 < x < 0.7
        # y < 0.05
        ineqs = (
            x[0] - 0.3,
            0.7 - x[0],
            0.05 - x[1]
        )
        return ineqs
    
    @dib.region_of(domain)
    def sub_domain2(x):
        # 0.95 < x
        # 0.3 < y < 0.7
        ineqs = (
            x[1] - 0.3,
            0.7 - x[1],
            x[0] - 0.95
        )
        return ineqs
    
    md.sub = [
        sub_domain1.expression(),
        sub_domain2.expression()
    ]

    md.wts = [0.5, 0.5]

    centers_b = [
        (0.1, 0.0),
        (0.2, 0.0),
        (0.3, 0.0),
        (0.7, 0.0),
        (0.8, 0.0),
        (0.9, 0.0),
        (1.0, 0.0),
        (1.0, 0.1),
        (1.0, 0.2),
        (1.0, 0.3),
        (1.0, 0.7),
        (1.0, 0.8),
        (1.0, 0.9),
    ]

    centers_b += [(0.0, i*0.1) for i in range(11)]
    centers_b += [(i*0.1, 1.0) for i in range(1, 11)]
    centers_b += [(i*0.1, 1.0) for i in range(1, 11)]
    
    centers_i = [(0.1, i*0.2 + 0.3) for i in range(4)]
    centers_i += [(0.2, i*0.2 + 0.2) for i in range(4)]
    centers_i += [(0.3, i*0.2 + 0.3) for i in range(4)]
    centers_i += [(0.4, i*0.2 + 0.2) for i in range(4)]
    centers_i += [(0.5, i*0.2 + 0.3) for i in range(4)]
    centers_i += [(0.6, i*0.2 + 0.2) for i in range(4)]
    centers_i += [(0.7, i*0.2 + 0.3) for i in range(4)]
    centers_i += [(0.8, i*0.2 + 0.2) for i in range(4)]
    
    centers = np.array(centers_b + centers_i)
    radii_b = np.repeat(0.08, len(centers_b))
    radii_i = np.repeat(0.05, len(centers_i))
    radii = np.concatenate((radii_b, radii_i))
    dib.save_initial(
        comm, (centers, radii),
        domain, test_path / "initial.xdmf"
    )

    dib.runDP(
        model = md,
        initial_guess = (centers, radii),
        save_path = test_path,
        niter = 250,
        dfactor = 1e-2,
        ctrn_tol = 1e-3,
        lgrn_tol = 1e-2
    )


def test_13():
    """
    Run: mpirun -np <nbr of processes> python test.py 13
    """

    test_name = "Heat conduction (Data Parallelism)"
    test_path = Path("../results/t13/")
    dim = 2
    rank_dim = 1
    mesh_size = 5e-3

    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    dir1_idx, dir1_mkr = [1], 1
    dir2_idx, dir2_mkr = [2], 2
    dir3_idx, dir3_mkr = [3], 3
    dir4_idx, dir4_mkr = [4], 4
    
    boundary_parts = [
        (dir1_idx, dir1_mkr, "dir1"),
        (dir2_idx, dir2_mkr, "dir2"),
        (dir3_idx, dir3_mkr, "dir3"),
        (dir4_idx, dir4_mkr, "dir4")
    ]

    # Create gmsh domain for Data Parallelism
    output = dib.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = dib.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = dib.homogeneus_boundary(
        domain, space, dim, rank_dim
    )
    area = 0.5
    # Create the model
    md = Heat(
        dim, domain, space, dirichlet_bcs, area, "1Load"
    )

    # @dib.region_of(domain)
    # def sub_domain(x):
    #     return (0.1**2 - (0.5 - x[0])**2 - (0.5 - x[1])**2, )
    
    # md.sub = [sub_domain.expression()]
    centers = []
    centers += [(0.1, i*0.2 + 0.1) for i in range(5)]
    centers += [(0.3, i*0.2 + 0.1) for i in range(5)]
    centers += [(0.5, i*0.2 + 0.1) for i in range(5) if i != 2]
    centers += [(0.7, i*0.2 + 0.1) for i in range(5)]
    centers += [(0.9, i*0.2 + 0.1) for i in range(5)]
    centers += [(0.0, i*0.2) for i in range(6)]
    centers += [(0.2, i*0.2) for i in range(6)]
    centers += [(0.4, i*0.2) for i in range(6) if i not in (2, 3)]
    centers += [(0.6, i*0.2) for i in range(6) if i not in (2, 3)]
    centers += [(0.8, i*0.2) for i in range(6)]
    centers += [(1.0, i*0.2) for i in range(6)]
    
    
    centers = np.array(centers)
    radii = np.repeat(0.05, centers.shape[0])

    dib.save_initial(
        comm, (centers, radii),
        domain, test_path / "initial.xdmf"
    )
    
    dib.runDP(
        model = md,
        initial_guess = (centers, radii),
        save_path = test_path,
        niter = 200,
        dfactor = 1.0,
        ctrn_tol = 1e-3,
        lgrn_tol = 5e-2,
        lv_iter = (10, 18),
        smooth = True
    )

#========================
def test_13425772():
    """
    Run: mpirun -np <nbr of processes> python test.py 12
    """

    test_name = "Heat conduction under three load cases (Data Parallelism)"
    test_path = Path("../results/t12/")
    dim = 2
    rank_dim = 1
    mesh_size = 1e-2
    
    vertices = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.4],
        [2.0, 0.6],
        [2.0, 1.0],
        [1.1, 1.0],
        [0.9, 1.0],
        [0.0, 1.0],
        [0.0, 0.6],
        [0.0, 0.4],
    ])
    
    dir1_idx, dir1_mkr = [1], 1
    dir2_idx, dir2_mkr = [3], 2
    dir3_idx, dir3_mkr = [6], 3
    dir4_idx, dir4_mkr = [9], 4
    
    boundary_parts = [
        (dir1_idx, dir1_mkr, "dir1"),
        (dir2_idx, dir2_mkr, "dir2"),
        (dir3_idx, dir3_mkr, "dir3"),
        (dir4_idx, dir4_mkr, "dir4")
    ]

    # Create gmsh domain for Data Parallelism
    output = dib.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    g_space = dib.create_space(domain, "CG", 2, rank_dim)
    T = 2.0
    
    g_bcs_1 = dib.dirichlet_with_values(
        domain, g_space,
        boundary_tags,
        [dir1_mkr, dir2_mkr],
        [0.0, T]
    )

    g_bcs_2 = dib.dirichlet_with_values(
        domain, g_space,
        boundary_tags,
        [dir1_mkr, dir3_mkr],
        [0.0, T]
    )

    g_bcs_3 = dib.dirichlet_with_values(
        domain, g_space,
        boundary_tags,
        [dir1_mkr, dir4_mkr],
        [0.0, T]
    )

    g_funcs = dib.dirichlet_extension_from_bcs(
        domain,
        g_space,
        [g_bcs_1, g_bcs_2, g_bcs_3]
    )

    # Save
    g_space_1 = dib.create_space(domain, "CG", 1, rank_dim)
    g_funcs_1 = dib.interpolate(
        funcs = g_funcs,
        to_space = g_space_1,
        name = "g"
    )

    dib.save_functions(
        comm, domain,
        g_funcs_1, test_path / "g1.xdmf"
    )

    space = dib.create_space(domain, "CG", 1, rank_dim)

    bcs_1 = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir1_mkr, dir2_mkr],
        rank_dim
    )

    bcs_2 = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir1_mkr, dir3_mkr],
        rank_dim
    )

    bcs_3 = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir1_mkr, dir4_mkr],
        rank_dim
    )


    dirichlet_bcs = [bcs_1, bcs_2, bcs_3]
    area = 0.75
    # Create the model
    md = HeatMultiple(
        dim, domain, space, dirichlet_bcs, area
    )

    md.wts = [0.4, 0.2, 0.4]
    md.gs = g_funcs	
    @dib.region_of(domain)
    def sub_domain1(x):
        return ((x[0]-1.0)**2 + (x[1]-1.0)**2 < 0.12**2,)
    sub1 = sub_domain1.expression()
    
    @dib.region_of(domain)
    def sub_domain2(x):
        return ((x[0]-0.0)**2 + (x[1]-0.5)**2 < 0.12**2,)
    sub2 = sub_domain2.expression()
    
    @dib.region_of(domain)
    def sub_domain3(x):
        return ((x[0]-2.0)**2 + (x[1]-0.5)**2 < 0.12**2,)
    sub3 = sub_domain3.expression()
    
    md.subs = [sub1, sub2, sub3]
    
    centers = []
    centers += [(i*0.2, 0.0) for i in range(11)]
    centers += [(i*0.2, 0.25) for i in range(11)]

    centers += [(i*0.2, 0.75) for i in range(4)]
    centers += [(i*0.2, 0.75) for i in range(7, 11)]
    centers += [(i*0.2, 1.0) for i in range(4)]
    centers += [(i*0.2, 1.0) for i in range(7, 11)]
    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])

    dib.save_initial(
        comm, (centers, radii),
        domain, test_path / "initial.xdmf"
    )

    # Run Data Parallelism
    dib.runDP(
        model = md,
        initial_guess = (centers, radii),
        niter = 300,
        save_path = test_path,
        reinit_step = 10,
        reinit_pars = (16, 1e-2),
        dfactor = 0.5,
        lv_time = (1e-3, 1e-2),
        lv_iter = (10, 16),
        ctrn_tol = 1e-2,
        lgrn_tol = 1e-3
    )

def test_13Cantiliever():
        
    test_name = "Cantilever"
    test_path = "../results/t02/"
    dim = 2
    rank_dim = 2
    mesh_size = 0.025
    
    # ========== Domain ==========
    vertices = [[0., -1.],
                 [1., -.5],
                [1., -.35], 
                [1., .5],
                [0., 0.],
                [-1., 2.],
                [-2., 1.]]
    dir_index, dir_marker = [6], 1
    neu_index, neu_marker = [2], 2

    boundary_parts = [
        (dir_index, dir_marker, "dirichlet"),
        (neu_index, neu_marker, "neumann")
    ]

    output = dib.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size,
        path = test_path,
        plot = True
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    vertices = vertices + [vertices[0]]
    dir = np.array(
        vertices[(dir_index[0]-1):(dir_index[-1]+1)]
    )
    neu = np.array(
        vertices[(neu_index[0]-1):(neu_index[-1]+1)]
    )
    boundaries = [
        (dir, "red"),
        (neu, "blue")
    ]
    
    dib.see_domain(
        dim,
        test_path + "domain.xdmf",
        boundaries,
        save_path = test_path + "domain.png"
    )

    # ========== Model ==========
    """
    Variable "space" containts vector valued functions
    (u(x) is a 2D vector), then "rank_dimension" is 2.
    """
    space = dib.create_space(
        domain, "CG", 1, rank_dim
    )

    dirichlet_bcs = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir_marker], rank_dim
    )

    ds_g = dib.marked_ds(
        domain,
        boundary_tags,
        [neu_marker]
    )
    
    area = 1.5
    g = [0.0, -2.0]

    md = Compliance(
        dim, domain, space,
        g, ds_g[0], dirichlet_bcs, area
    )

    # ========== Initial guess ==========
    centers = np.array([
        (-1.5, 1.5),
        (-1.2, 1.1),
        (-1.0, 0.8),
        (-0.702, 0.582),
        (-0.6, 0.2),
        (-0.34, -0.143),
        (-0.07, -0.45),
        (0.337, -0.555),
        (0.197, -0.898),
        (-1.181, 1.797),
        (-1.801, 1.186),
        (-0.831, 1.117),
        (-1.362, 0.711),
        (-0.786, 1.596),
        (-1.805, 0.799),
        (-1.092, 1.447),
        (-1.547, 1.0),
        (-0.480, 0.964),
        (-1.322, 0.320),
        (-0.247, 0.493),
        (-0.814, -0.187),
        (-0.497, -0.501),
        (-0.195, -0.807),
        (-0.042, 0.099),
        (0.393, 0.191),
        (0.723, -0.642),
        (0.220, -0.223),
        (0.675, -0.263),
        (0.956, 0.412),
        (0.743, 0.132),
        (-0.963, 0.212)
    ])

    radii = np.repeat(0.12, centers.shape[0])
    
    dib.see_initial_guess(
        dim,
        test_path + "domain.xdmf",
        (centers, radii),
        boundaries = boundaries,
        save_path = test_path + "initial.png"
    )

    # ========== Go ==========

    dib.runDP(
        model = md,
        initial_guess = (centers, radii),
        save_path = test_path,
        boundaries = boundaries,
        niter = 200,
        reinit_step = 10,
        ctr_tol = 1e-3,
        dfactor = 1e-1
    )

def test_1222():
    name_test = "../results/t06/"
    
    # ========== Domain ==========
    size_mesh = 0.015
    vertices = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.1],
        [2.0, 1.0],
        [0.0, 1.0]
    ]
    dir_index, dir_marker = [5], 1
    neu_index, neu_marker = [2], 2
    boundary_parts = [
        (dir_index, dir_marker, "dirichlet"),
        (neu_index, neu_marker, "neumann")
    ]
    hole = [[0.25*np.cos(i) + 0.5, 0.25*np.sin(i) + 0.5] for i in np.linspace(np.pi/2., 3.*np.pi/2., 40)]
    domain, num_tri, boundary_tags = dib.create_domain_2d(
        vertices, [hole], boundary_parts, size_mesh, True
    )
    
    # plot domain
    vertices = vertices + [vertices[0]]
    dir = np.array(vertices[(dir_index[0]-1):(dir_index[-1]+1)]).T
    neu = np.array(vertices[(neu_index[0]-1):(neu_index[-1]+1)]).T
    boundaries = [(dir[0], dir[1], "red"), (neu[0], neu[1], "blue")]
    
    dib.see_domain(domain, boundaries)
    
    # ========== Model ==========
    """
    Variable "space" containts vector valued functions
    (u(x) is a 2D vector), then "rank_dimension" is 2.
    """
    space = functionspace(domain, ("CG", 1, (2, )))
    area = 1.
    g = [0., -4.]
    rank_dimension = 2 
    dirichlet_bcs = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir_marker], rank_dimension
    )
    ds_g = dib.marked_ds(
        domain,
        boundary_tags,
        [neu_marker]
    )[0] 
    SE = Compliance(
        domain, space, g, ds_g, dirichlet_bcs, area
    )

    # ========== Initial guess ==========
    centers = [(.3, 0.), (1., 0.), (1.7, 0.),
            (.3, 1.), (1., 1.), (1.7, 1.), (2., 1.),
            (.3, .5), (1., .5), (1.7, .5),
            (0., .25), (.65, .25), (1.35, .25), (2., .25),
            (0., .75), (.65, .75), (1.35, .75), (2., .75),
            (0, .5)]
    
    radii = len(centers)*[0.1]
    dib.see_initial_guess(
        domain,
        (centers, radii),
        boundaries = boundaries,
        save_path = name_test + "initial.png"
    )
    
    # ========== Go ==========
    
    dib.run(
        SE, domain, space, (centers, radii),
        niter = 1000, reinit_step = 50,
        boundaries = boundaries,
        save_path = name_test
    )
        
def test_120():

    print("\n\tElasticity inverse problem: 2D case")

    name_test = "../results/t12/"
    
    # ========== Domain ==========
    mesh_size = 0.020

    def semi_ellipse(a, b, eps, npts):
        t_ = np.arcsin((b - eps)/b)
        t = np.linspace(-t_, np.pi + t_, npts)
        x = a*np.cos(t)
        y = b*np.sin(t) + (b - eps)
        return x, y
    
    npts = 80 # consider npts % 4 = 0
    part = npts/4
    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.2, npts))
    
    dir_index, dir_marker = [npts], 1
    bR_index, bR_marker = np.arange(1, part/2 + 1, dtype = int), 2
    neu_indexA, neu_markerA = int(part/2) + np.arange(1, part + 1, dtype = int), 3
    neu_indexB, neu_markerB = int(part/2) + np.arange(part + 1, 2*part + 1, dtype = int), 4
    neu_indexC, neu_markerC = int(part/2) + np.arange(2*part + 1, 3*part + 1, dtype = int), 5
    bL_index, bL_marker = np.arange(part/2 + 3*part + 1, npts, dtype = int), 6
    
    boundary_parts = [
        (dir_index, dir_marker, "dirichlet"),
        (bR_index, bR_marker, "bRight"),
        (neu_indexA, neu_markerA, "neumannA"),
        (neu_indexB, neu_markerB, "neumannB"),
        (neu_indexC, neu_markerC, "neumannC"),
        (bL_index, bL_marker, "bLeft"),
    ]

    domain, num_tri, boundary_tags = dib.create_domain_2d(
        vertices, boundary_parts, mesh_size, plot = True
    )

    dim = 2
    rank_dimension = 2
    space = functionspace(domain, ("CG", 1, (rank_dimension, )))
    space_for_g = functionspace(domain, ("CG", 2, (rank_dimension, )))

    forces = [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]]
    
    dirbc_partial = dib.homogeneous_dirichlet(
        domain,
        space,
        boundary_tags,
        [dir_marker],
        rank_dimension
    )
    
    dirbc_total = dib.homogeneus_boundary(
        domain,
        space,
        dim,
        rank_dimension
    )

    ds_parts = dib.marked_ds(
        domain,
        boundary_tags,
        [bR_marker, neu_markerA, neu_markerB, neu_markerC, bL_marker]
    )
    
    ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]] 
    ds1 = ds_parts[0] + ds_parts[1] + ds_parts[2] + ds_parts[3] + ds_parts[4] 

    IE = InverseElasticity(
        dim, domain, space, forces, ds_forces, ds1, dirbc_partial, dirbc_total
    )

    # loaded_data = np.load("extensions.npy")
    # extensions = [Function(space_for_g) for i in range(3)]
    # for i in range(3):
    # 	extensions[i].x.array[:] = loaded_data[i]
    # #IE.gs = extensions

    def beam_equation(x):
        """
        Implicit equation :
        ((x-x0)^2 + (y-y0)^2)^2 = a((x-x0)^3 + (y-y0)^3) 
        """
        # inverse rotation
        angle = np.pi/6.0
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        x0 = cos_theta * x[0] + sin_theta * x[1]
        x1 = -sin_theta * x[0] + cos_theta * x[1]
        return (x0**2 + (x1-0.2)**2)**2 - 0.5*(x0**3 + (x1-0.2)**3)


    states = dib.solve_pde(domain, space, IE.pde, beam_equation)
    print("R:",states[0].x.array)

def test_14():
    print("\n\tSymmetric cantilever")
    print("")
    name_test = "../results/t14/"
    lag_multiplier = 0.3
    # ========== Domain ==========
    size_mesh = 0.015
    vertices = [
        [0.0, 0.0],
        [1.2, 0.0],
        [2.0, 0.0],
        [2.0, 0.4],
        [2.0, 0.6],
        [2.0, 1.0],
        [1.2, 1.0],
        [0.0, 1.0],
        [0.0, 0.85],
        [0.0, 0.15]
    ]
    d_idx_bot, d_mkr_bot = [1], 1
    neu_idx, neu_mkr = [4], 2
    d_idx_top, d_mkr_top = [7], 3
    dir_idx_1, dir_mkr_1 = [8], 4
    dir_idx_2, dir_mkr_2 = [10], 5
    
    boundary_parts = [
        (dir_idx_1, dir_mkr_1, "dir1"),
        (dir_idx_2, dir_mkr_2, "dir2"),
        (neu_idx, neu_mkr, "neu"),
        (d_idx_bot, d_mkr_bot, "dbot"),
        (d_idx_top, d_mkr_top, "dtop")
    ]
    domain, num_tri, boundary_tags = dib.create_domain_2d(
        vertices, boundary_parts, size_mesh, plot = True
    )
    
    # plot domain
    vertices = vertices + [vertices[0]]
    dir1 = np.array(vertices[(dir_idx_1[0]-1):(dir_idx_1[-1]+1)]).T
    dir2 = np.array(vertices[(dir_idx_2[0]-1):(dir_idx_2[-1]+1)]).T
    neu = np.array(vertices[(neu_idx[0]-1):(neu_idx[-1]+1)]).T
    dbot = np.array(vertices[(d_idx_bot[0]-1):(d_idx_bot[-1]+1)]).T
    dtop = np.array(vertices[(d_idx_top[0]-1):(d_idx_top[-1]+1)]).T
    boundaries = [
        (dir1[0], dir1[1], "red"), 
        (dir2[0], dir2[1], "red"), 
        (neu[0], neu[1], "blue"),
        (dbot[0], dbot[1], "green"),
        (dtop[0], dtop[1], "darkgreen"),
    ]
    
    dib.see_domain(domain, boundaries)
    
    # ========== Model ==========
    
    """
    Variable "space" containts vector valued functions
    (u(x) is a 2D vector), then "rank_dimension" is 2.
    """
    space = functionspace(domain, ("CG", 1, (2, )))
    
    area = 1.0
    g = [0.0, -1.0]
    rank_dimension = 2 
    
    dirichlet_bcs = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir_mkr_1, dir_mkr_2], rank_dimension
    )

    ds_g = dib.marked_ds(
        domain,
        boundary_tags,
        [neu_mkr]
    )[0] 

    SE = Compliance(
        domain, space, g, ds_g, dirichlet_bcs, area
    )
    
    SE.bc_th = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [d_mkr_bot, d_mkr_top], rank_dimension
    )
    #SE = ComplianceLg(domain, space, g, ds_g, dirichlet_bcs, lag_multiplier)

    # ========== Initial guess ==========
    centers = [
        (0.0, 0.5), (0.5, 0.5), (1.0, 0.5), (1.5, 0.5),
        (0.25, 0.25), (0.75, 0.25), (1.25, 0.25),
        (0.25, 0.75), (0.75, 0.75), (1.25, 0.75),
        (2.0, 0.0), (2.0, 1.0)
    ]

    radii = 10*[0.1] + 2*[0.25]
    
    dib.see_initial_guess(
        domain,
        (centers, radii),
        boundaries = boundaries,
        save_path = name_test + "initial.png"
    )

    # ========== Go ==========

    dib.run(
        SE, domain, space, (centers, radii),
        boundaries = boundaries,
        save_path = name_test,
        niter = 200,
        reinit_step = 10,
        plot_step = 20,
        dfactor = 1e-1,
        ctrs_tol = 1e-2
    )

def test_15():
    
    print("\n\tSloping roof")
    name_test = "../results/t15/"
    
    # ========== Domain ==========
    
    size_mesh = 0.02
    
    vertices = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 1.0]
    ]
    
    dir_idx, dir_mkr = [1], 1
    neu_idx, neu_mkr = [3], 2
    
    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx, neu_mkr, "neu")
    ]

    out = dib.create_domain_2d(
        vertices,
        boundary_parts,
        size_mesh
    )

    domain, num_tri, boundary_tags = out
    print(f"\n> number of triangles = {num_tri}")
    
    # ========== Plot ==========

    vertices = vertices + [vertices[0]]
    
    dir = np.array(
        vertices[(dir_idx[0]-1):(dir_idx[-1]+1)]
    ).T
    neu = np.array(
        vertices[(neu_idx[0]-1):(neu_idx[-1]+1)]
    ).T

    boundaries = [
        (dir[0], dir[1], "red"),
        (neu[0], neu[1], "blue")
    ]
    
    dib.see_domain(
        domain,
        boundaries = boundaries,
        save_path = name_test + "domain.png"
    )
    
    # ========== Model ==========
    
    dim = 2
    area = 1.5
    g = [0.0, -0.05]
    rank_dimension = 2

    space = dib.create_space(
        domain, "CG", 1, rank_dimension
    )

    dir_bc = dib.homogeneous_dirichlet(
        domain,
        space,
        boundary_tags,
        [dir_mkr],
        rank_dimension
    )

    ds_g = dib.marked_ds(
        domain,
        boundary_tags,
        [neu_mkr]
    )
    
    md = Compliance(
        dim, domain, space,
        g, ds_g[0], dir_bc, area
    )

    # ========== Initial guess ==========
    centers = [(i*0.5, 0.0) for i in range(5)]
    centers += [(0.25 + i*0.5 , 0.25) for i in range(4)]
    centers += [(i*0.5, 0.5) for i in range(5)]
    centers += [(0.25 + i*0.5 , 0.75) for i in range(4)]
    centers += [(1.0 + i*0.5 , 1.0) for i in range(3)]
    centers += [
        (2.0, 1.5),
        (0.0, 0.25),
        (2.0, 0.25),
        (2.0, 0.75),
        (2.0, 1.25) 
    ]
    
    radii = len(centers)*[0.1]
    
    dib.see_initial_guess(
        domain,
        (centers, radii),
        boundaries = boundaries,
        save_path = name_test + "initial.png"
    )

    # ========== Go ==========

    dib.run(
        md, domain, space, (centers, radii),
        boundaries = boundaries,
        save_path = name_test,
        niter = 200,
        reinit_step = 8,
        plot_step = 20,
        dfactor = 1e-1,
        ctrs_tol = 1e-3
    )

def test_16():
    
    print("\n\tLarge sloping roof")
    name_test = "../results/t16/"
    
    # ========== Domain ==========
    
    size_mesh = 0.02
    
    base = 3.2
    domain_area = 3.0
    heigth = domain_area*4.0/3.0/base

    vertices = [
        [0.0, 0.0],
        [base, 0.0],
        [base, heigth],
        [0.0, heigth/2]
    ]
    
    dir_idx, dir_mkr = [1], 1
    neu_idx, neu_mkr = [3], 2
    
    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx, neu_mkr, "neu")
    ]

    out = dib.create_domain_2d(
        vertices,
        boundary_parts,
        size_mesh
    )

    domain, num_tri, boundary_tags = out
    print(f"\n> number of triangles = {num_tri}")
    
    # ========== Plot ==========

    vertices = vertices + [vertices[0]]
    
    dir = np.array(
        vertices[(dir_idx[0]-1):(dir_idx[-1]+1)]
    ).T
    neu = np.array(
        vertices[(neu_idx[0]-1):(neu_idx[-1]+1)]
    ).T

    boundaries = [
        (dir[0], dir[1], "red"),
        (neu[0], neu[1], "blue")
    ]
    
    dib.see_domain(
        domain,
        boundaries = boundaries,
        save_path = name_test + "domain.png"
    )
    
    # ========== Model ==========
    
    dim = 2
    area = domain_area/2.0
    g = [0.0, -0.05]
    rank_dimension = 2

    space = dib.create_space(
        domain, "CG", 1, rank_dimension
    )

    dir_bc = dib.homogeneous_dirichlet(
        domain,
        space,
        boundary_tags,
        [dir_mkr],
        rank_dimension
    )

    ds_g = dib.marked_ds(
        domain,
        boundary_tags,
        [neu_mkr]
    )
    
    md = Compliance(
        dim, domain, space,
        g, ds_g[0], dir_bc, area
    )

    # ========== Initial guess ==========
    centers = [(i*0.4, 0.0) for i in range(9)]
    centers += [(0.2 + i*0.4, 0.25) for i in range(8)]
    centers += [(1.2 + i*0.4, 0.5) for i in range(6)]
    centers += [
        (0.0, 0.2), (2.8, 0.75),
        (base, 0.25), (base, 0.5), (base, 0.75)
    ]
    
    radii = len(centers)*[0.08]
    
    dib.see_initial_guess(
        domain,
        (centers, radii),
        boundaries = boundaries,
        save_path = name_test + "initial.png"
    )

    # ========== Go ==========

    dib.run(
        md, domain, space, (centers, radii),
        boundaries = boundaries,
        save_path = name_test,
        niter = 200,
        reinit_step = 8,
        plot_step = 20,
        dfactor = 1e-1,
        ctrs_tol = 1e-3
    )

def test_17():
    name_test = "../results/t04/"
    
    # ========== Domain ==========
    size_mesh = 0.015

    x = np.linspace(0., 2., 20)
    y = x*(2.-x)/5.
    vertices = [(xi, yi) for xi, yi in zip(x, y)]
    vertices = vertices + [[2., .45], [2., .55]]
    vertices = vertices + [(2.-xi, 1.-yi) for xi, yi in zip(x, y)]
    dir_index, dir_marker = [42], 1
    neu_index, neu_marker = [21], 2
    boundary_parts = [
        (dir_index, dir_marker, "dirichlet"),
        (neu_index, neu_marker, "neumann")
    ]
    domain, num_tri, boundary_tags = dib.create_domain_2d(
        vertices, [], boundary_parts, size_mesh, True
    )
    
    # plot domain
    vertices = vertices + [vertices[0]]
    dir = np.array(vertices[(dir_index[0]-1):(dir_index[-1]+1)]).T
    neu = np.array(vertices[(neu_index[0]-1):(neu_index[-1]+1)]).T
    boundaries = [(dir[0], dir[1], "red"), (neu[0], neu[1], "blue")]
    
    dib.see_domain(domain, boundaries)
    
    # ========== Model ==========
    """
    Variable "space" containts vector valued functions
    (u(x) is a 2D vector), then "rank_dimension" is 2.
    """
    space = functionspace(domain, ("CG", 1, (2, )))
    area = 0.7
    g = [0., -4.]
    rank_dimension = 2 
    dirichlet_bcs = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir_marker], rank_dimension
    )
    ds_g = dib.marked_ds(
        domain,
        boundary_tags,
        [neu_marker]
    )[0] 
    SE = Compliance(
        domain, space, g, ds_g, dirichlet_bcs, area
    )

    # ========== Initial guess ==========
    centers = [(2., 0.),
            (2., 1.),
            (.3, .5), (1., .5), (1.7, .5),
            (0., .25), (.65, .25), (1.35, .3), (2., .25),
            (0., .75), (.65, .75), (1.35, .7), (2., .75),
            (0, .5)]
    
    radii = len(centers)*[0.1]
    dib.see_initial_guess(
        domain,
        (centers, radii),
        boundaries = boundaries,
        save_path = name_test + "initial.png"
    )
    
    # ========== Go ==========
    
    dib.run(
        SE, domain, space, (centers, radii),
        niter = 1000, reinit_step = 20,
        boundaries = boundaries,
        save_path = name_test
    )

def test_18():
    name_test = "../results/t18/"
    
    # ========== Domain ==========
    size_mesh = 0.015
    vertices = [
        [0., 0.],
        [0.95, 0.],
        [1.05, 0.],
        [2., 0.],
        [2., 1.],
        [0., 1.]
    ]
    neu_index, neu_marker = [2], 1
    boundary_parts = [
        (neu_index, neu_marker, "neumann")
    ]
    domain, num_tri, boundary_tags = dib.create_domain_2d(
        vertices, [], boundary_parts, size_mesh, True
    )
    
    # plot domain
    neu = np.array(vertices[(neu_index[0]-1):(neu_index[-1]+1)]).T
    boundaries = [(neu[0], neu[1], "blue")]
    
    dib.see_domain(domain, boundaries)

    # ========== Model ==========
    """
    Variable "space" containts vector valued functions
    (u(x) is a 2D vector), then "rank_dimension" is 2.
    """
    space = functionspace(domain, ("CG", 1, (2, )))
    area = 0.5
    g = [0., -4.]
    rank_dimension = 2 
    
    bc1 = dib.homogeneous_dirichlet_point(
        domain,
        space,
        [[0.0, 0.0]], # boundary point
        rank_dimension
    )

    bc2 = dib.homogeneous_dirichlet_point_coordinate(
        domain,
        space,
        [[2.0, 0.0]], # boundary point
        [1] # y-component
    )

    ds_g = dib.marked_ds(
        domain,
        boundary_tags,
        [neu_marker]
    )[0] 
    
    SE = Compliance(
        domain, space, g, ds_g, [bc1[0], bc2[0]], area
    )

    # ========== Initial guess ==========
    centers = [(0., 1.), (2., 1.),
            (1./3., 1.0), (1., 1.), (5./3., 1.0),
            (1./3., 0.75), (1., 0.75), (5./3., 0.75),
            (1./3., 0.5), (1., 0.5), (5./3., 0.5),
            (1./3., 0.25), (1., 0.25), (5./3., 0.25),
            (0., .375), (2./3., .375), (4./3., .375), (2., .375),
            (0., .625), (2./3., .625), (4./3., .625), (2., .625),
            (0., .875), (2./3., .875), (4./3., .875), (2., .875)]
    
    radii = len(centers)*[0.06]
    dib.see_initial_guess(
        domain,
        (centers, radii),
        boundaries = boundaries,
        save_path = name_test + "initial.png"
    )
    
    # ========== Go ==========
    
    dib.run(
        SE, domain, space, (centers, radii),
        niter = 1000, reinit_step = 50,
        boundaries = boundaries,
        save_path = name_test
    )

def test_18():
    print("Half-wheel")
    name_test = "../results/t05/"
    
    # ========== Domain ==========
    size_mesh = 0.015
    vertices = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.1],
        [2.0, 1.0],
        [0.0, 1.0]
    ]
    dir_index, dir_marker = [5], 1
    neu_index, neu_marker = [2], 2
    boundary_parts = [
        (dir_index, dir_marker, "dirichlet"),
        (neu_index, neu_marker, "neumann")
    ]
    domain, num_tri, boundary_tags = dib.create_domain_2d(
        vertices, [], boundary_parts, size_mesh, True
    )
    
    # plot domain
    vertices = vertices + [vertices[0]]
    dir = np.array(vertices[(dir_index[0]-1):(dir_index[-1]+1)]).T
    neu = np.array(vertices[(neu_index[0]-1):(neu_index[-1]+1)]).T
    boundaries = [(dir[0], dir[1], "red"), (neu[0], neu[1], "blue")]
    
    dib.see_domain(domain, boundaries)
    
    # ========== Model ==========
    """
    Variable "space" containts vector valued functions
    (u(x) is a 2D vector), then "rank_dimension" is 2.
    """
    space = functionspace(domain, ("CG", 1, (2, )))
    area = 1.
    g = [0., -4.]
    rank_dimension = 2 
    dirichlet_bcs = dib.homogeneous_dirichlet(
        domain, space, boundary_tags,
        [dir_marker], rank_dimension
    )
    ds_g = dib.marked_ds(
        domain,
        boundary_tags,
        [neu_marker]
    )[0] 
    SE = Compliance(
        domain, space, g, ds_g, dirichlet_bcs, area
    )

    # ========== Initial guess ==========
    centers = [(.3, 0.), (1., 0.), (1.7, 0.),
            (.3, 1.), (1., 1.), (1.7, 1.), (2., 1.),
            (.3, .5), (1., .5), (1.7, .5),
            (0., .25), (.65, .25), (1.35, .25), (2., .25),
            (0., .75), (.65, .75), (1.35, .75), (2., .75),
            (0, .5)]
    
    radii = len(centers)*[0.1]
    dib.see_initial_guess(
        domain,
        (centers, radii),
        boundaries = boundaries,
        save_path = name_test + "initial.png"
    )
    
    # ========== Go ==========
    
    dib.run(
        SE, domain, space, (centers, radii),
        niter = 1000, reinit_step = 20,
        boundaries = boundaries,
        save_path = name_test
    )

def test_20():
    name_test = "../results/t08/"
    
    # ========== Domain ==========
    size_mesh = 0.015
    vertices = [
        [0., 0.],
        [0.95, 0.],
        [1.05, 0.],
        [2., 0.],
        [2., 1.],
        [0., 1.]
    ]
    neu_index, neu_marker = [2], 1
    boundary_parts = [
        (neu_index, neu_marker, "neumann")
    ]
    domain, num_tri, boundary_tags = dib.create_domain_2d(
        vertices, [], boundary_parts, size_mesh, True
    )
    
    # plot domain
    neu = np.array(vertices[(neu_index[0]-1):(neu_index[-1]+1)]).T
    boundaries = [(neu[0], neu[1], "blue")]
    
    dib.see_domain(domain, boundaries)

    # ========== Model ==========
    """
    Variable "space" containts vector valued functions
    (u(x) is a 2D vector), then "rank_dimension" is 2.
    """
    space = functionspace(domain, ("CG", 1, (2, )))
    area = 0.5
    g = [0., -6.]
    rank_dimension = 2 
    
    bc1 = dib.homogeneous_dirichlet_point(
        domain,
        space,
        [[0.0, 0.0]], # boundary point
        rank_dimension
    )

    bc2 = dib.homogeneous_dirichlet_point_coordinate(
        domain,
        space,
        [[2.0, 0.0]], # boundary point
        [1] # y-component
    )

    ds_g = dib.marked_ds(
        domain,
        boundary_tags,
        [neu_marker]
    )[0] 
    
    SE = Compliance(
        domain, space, g, ds_g, [bc1[0], bc2[0]], area
    )

    # ========== Initial guess ==========
    
    centers = [
            (0., 1.), (2./5., 1.), (4./5., 1.), (6./5., 1.), (8./5., 1.), (2., 1.),
            (0., .75), (2./5., .75), (4./5., .75), (6./5., .75), (8./5., .75), (2., .75),
            (0., .5), (2./5., .5), (4./5., .5), (6./5., .5), (8./5., .5), (2., .5),
            (0., .25), (2./5., .25), (4./5., .25), (6./5., .25), (8./5., .25), (2., .25)]
    
    radii = len(centers)*[0.06]
    dib.see_initial_guess(
        domain,
        (centers, radii),
        boundaries = boundaries,
        save_path = name_test + "initial.png"
    )
    
    # ========== Go ==========
    
    dib.run(
        SE, domain, space, (centers, radii),
        niter = 650, reinit_step = 50,
        boundaries = boundaries,
        save_path = name_test
    )


test_functions = {
    "00": test_00,
    "01": test_01,
    "02": test_02,
    "03": test_03,
    "04": test_04,
    "05": test_05,
    "06": test_06,
    "07": test_07,
    "08": test_08,
    "09": test_09,
    "10": test_10,
    "11": test_11,
    "12": test_12,
    "13": test_13,
}

def main():
    
    import sys

    if len(sys.argv) != 2:
        print("Usage: python tests.py <test_id>")
        print("Example: python test.py 01")
        return

    test_id = sys.argv[1]
    func = test_functions.get(test_id)

    if func:
        func()
    else:
        print(f"Test '{test_id}' not recognized.")

if __name__ == "__main__":
    main()