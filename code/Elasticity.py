import formopt as fop
from Elasticity_models import (
    Hookecomponents,
    SVKcomponents,
    MRcomponents,
    Compliance,
    Mechanism,
    KohnVogelius,
)

import numpy as np
from pathlib import Path
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


def cantilever(mod, test_path, load, alpha, nM=0):
    """
    Cantilever with one load.

    Numerical tests:

    Hooke model
    `cantilever("Hooke", Path("../results/Elasticity/t01/"), load=-10.0, alpha=0.25)`
    Saint-Venant-Kirchhoff model
    `cantilever("SVK", Path("../results/Elasticity/t02/"), load=-10.0, alpha=0.25, nM=8)`
    Mooney-Rivlin model
    `cantilever("MR", Path("../results/Elasticity/t03/"), load=-10.0, alpha=0.25, nM=16)`

    """
    dim = 2
    rank_dim = 2
    mesh_size = 0.01
    vertices = np.array(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 0.46), (2.0, 0.54), (2.0, 1.0), (0.0, 1.0)]
    )
    dir_idx, dir_mkr = [6], 1
    neu_idx, neu_mkr = [3], 2
    boundary_parts = [(dir_idx, dir_mkr, "dir"), (neu_idx, neu_mkr, "neu")]
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )
    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print(f"> Nbr of triangles = {nbr_tri}")

    space = fop.create_space(domain, "CG", rank_dim)
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr])
    alpha = alpha
    g = [(0.0, load)]

    if mod == "Hooke":
        elast_model = Hookecomponents()
    elif mod == "SVK":
        elast_model = SVKcomponents()
    elif mod == "MR":
        elast_model = MRcomponents()

    md = Compliance(
        dim,
        domain,
        space,
        test_path,
        elast_model,
        [g],
        [ds_g],
        dirichlet_bcs,
        alpha,
    )

    md.nN = nM

    @fop.region_of(domain)
    def sub_domain(x):
        # 0.44 < x[1] < 0.56, 1.9 < x[0]
        return [x[1] - 0.44, 0.56 - x[1], x[0] - 1.9]

    md.sub = [sub_domain.expression()]

    dd = 0  # 1
    h = 1.4 / (2.0 + dd)

    centers = [(2.0, 0.35), (2.0, 0.65), (2.0, 0.0), (2.0, 1.0)]
    centers += [(0.0, 0.25), (0.0, 0.5), (0.0, 0.75)]
    # centers += [(0.3 + h / 2.0 + h, 0.0), (0.3 + h / 2.0 + h, 1.0)]
    centers += [(0.3 + i * h, 0.0) for i in range(3 + dd)]
    centers += [(0.3 + h / 2.0 + i * h, 0.25) for i in range(2 + dd)]
    centers += [(0.3 + i * h, 0.5) for i in range(3 + dd)]
    centers += [(0.3 + h / 2.0 + i * h, 0.75) for i in range(2 + dd)]
    centers += [(0.3 + i * h, 1.0) for i in range(3 + dd)]

    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])
    md.create_initial_level(centers, radii)

    md.runDP(
        niter=300,
        dfactor=1e-2,
        lv_iter=(12, 24),
        lv_time=(1e-3, 1.0),
        cost_tol=1e-3,
        smooth=True,
        reinit_step=4,
        reinit_pars=(6, 0.01),
        random_pars=(26, 0.075),
    )


def deformation_cantilever(mod, test_path, last_iter):

    dim = 2
    rank_dim = 2
    dir_idx, dir_mkr = [6], 1
    neu_idx, neu_mkr = [3], 2
    alpha = 0.25
    load = -25.0  # maximum load

    # read the domain
    filename = test_path / "domain.msh"
    domain, _, boundary_tags = fop.read_gmsh(filename, comm, 2)
    fop.all_connectivities(domain)

    # read the level set function
    phi = fop.read_level_set_function(test_path, domain, last_iter)

    space = fop.create_space(domain, "CG", rank_dim)
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )

    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr])
    g = [(0.0, load)]

    if mod == "Hooke":
        elast_model = Hookecomponents()
    elif mod == "SVK":
        elast_model = SVKcomponents()
    elif mod == "MR":
        elast_model = MRcomponents()

    md = Compliance(
        dim,
        domain,
        space,
        test_path,
        elast_model,
        [g],
        [ds_g],
        dirichlet_bcs,
        alpha,
    )

    md.nN = 200

    # We consider fewer iterations due to
    # the lack of convergence of the Moneey-Rivlin model
    # at the last iterations.
    k = 7
    md.nN = md.nN - k
    names = [f"u{i:03}" for i in range(1, md.nN)]
    factor = np.linspace(0.0, 1.0, md.nN)[1:]
    load = load * (md.nN - 1) / (md.nN + k - 1)

    if mod == "Hooke":
        uhs = []
        for fc, nm in zip(factor, names):
            load_i = fc * load
            print(f"> Factor = {fc:6.4f}, Load = {load_i:7.4f}, Function = {nm}")
            md.update_gs([[(0.0, load_i)]])
            uhs.append(fop.SolveLinearProblem(space, md.pde(phi)[0], nm))

    if mod == "SVK":
        md.update_gs([[(0.0, load)]])
        uhs = fop.SolveNonlinearOnce(domain, space, md.pde(phi)[0], names)

    if mod == "MR":
        md.update_gs([[(0.0, load)]])
        uhs = fop.SolveNonlinearOnce(domain, space, md.pde(phi)[0], names)

    fop.save_functions(comm, domain, [phi] + uhs, test_path / "deformations.xdmf")


def inverter_nonlinear():

    path_origin = Path("../results/Elasticity/t12/")
    last_iter = 180

    test_path = Path("../results/Elasticity/t13/")
    kappa = [16.0]
    alpha = 0.36
    eps = 1e-2
    niter = 300
    forces = (8.0, 14.0)
    nN = 10

    a, b = 0.05, 0.05
    dim = 2
    rank_dim = 2
    dirB_idx, dirB_mkr = [10], 1
    dirT_idx, dirT_mkr = [6], 2
    neuL_idx, neuL_mkr = [8], 3
    neuR_idx, neuR_mkr = [3], 4

    # read the domain
    filename = path_origin / "domain.msh"
    domain, _, boundary_tags = fop.read_gmsh(filename, comm, 2)
    fop.all_connectivities(domain)

    # read the level set function
    phi = fop.read_level_set_function(path_origin, domain, last_iter)

    space = fop.create_space(domain, "CG", rank_dim)
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dirB_mkr, dirT_mkr], rank_dim
    )
    ds_g = fop.marked_ds(domain, boundary_tags, [neuL_mkr, neuR_mkr])

    force_in, force_out = forces
    g_in, g_out = [(force_in, 0.0)], [(force_out, 0.0)]

    elast_model = MRcomponents()

    md = Mechanism(
        dim,
        domain,
        space,
        test_path,
        elast_model,
        g_in,
        [ds_g[0]],
        g_out,
        [ds_g[1]],
        dirichlet_bcs,
        kappa,
        alpha,
        eps,
    )

    md.nN = nN

    @fop.region_of(domain)
    def sub_dom_L(x):
        # x < 2*eps and 0.5 - a - eps < y < 0.5 + a + eps
        eps2 = a / 2.0
        return [2.0 * eps2 - x[0], x[1] - 0.5 + a + eps2, 0.5 + a + eps2 - x[1]]

    @fop.region_of(domain)
    def sub_dom_R(x):
        # x > 1.0 - 2*eps and 0.5 - a - eps < y < 0.5 + a + eps
        eps2 = a / 2.0
        return [x[0] - 1.0 + 2.0 * eps2, x[1] - 0.5 + a + eps2, 0.5 + a + eps2 - x[1]]

    md.sub = [sub_dom_L.expression(), sub_dom_R.expression()]
    md.set_initial_level(phi)
    md.runDP(
        niter=niter,
        dfactor=1e-2,
        lv_iter=(10, 24),
        lv_time=(1e-3, 0.1),
        cost_tol=1e-3,
        smooth=True,
        reinit_step=4,
        reinit_pars=(10, 5e-3),
        random_pars=(123, 0.05),
    )


def inverter(mod, test_path, kappa, alpha, eps, niter, elastic_pars, forces, nM=0):
    """
    mod="Hooke",
    test_path=Path("../results/t205/"),
    kappa=[0.01],
    alpha=0.01,
    eps=0.01,
    niter=300,
    elastic_pars=(20.0, 0.3),
    forces=(2.0, 1.0),
    """

    dim, rank_dim, mesh_size = 2, 2, 0.009
    a, b = 0.05, 0.05
    vertices = np.array(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.5 - a),
            (1.0, 0.5 + a),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0 - b),
            (0.0, 0.5 + a),
            (0.0, 0.5 - a),
            (0.0, b),
        ]
    )
    dirB_idx, dirB_mkr = [10], 1
    dirT_idx, dirT_mkr = [6], 2
    neuL_idx, neuL_mkr = [8], 3
    neuR_idx, neuR_mkr = [3], 4
    boundary_parts = [
        (dirB_idx, dirB_mkr, "dir_bottom"),
        (dirT_idx, dirT_mkr, "dir_top"),
        (neuL_idx, neuL_mkr, "neu_left"),
        (neuR_idx, neuR_mkr, "neu_right"),
    ]
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )
    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("Inverter mechanism")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    space = fop.create_space(domain, "CG", rank_dim)
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dirB_mkr, dirT_mkr], rank_dim
    )
    ds_g = fop.marked_ds(domain, boundary_tags, [neuL_mkr, neuR_mkr])

    Ym, Pr = elastic_pars
    force_in, force_out = forces
    g_in, g_out = [(force_in, 0.0)], [(force_out, 0.0)]

    if mod == "Hooke":
        print("> Hooke model")
        elast_model = Hookecomponents(Ym, Pr)
    elif mod == "SVK":
        print("> Saint-Venant-Kirchhoff model")
        elast_model = SVKcomponents(Ym, Pr)
    elif mod == "MR":
        print("> Moony-Rivlin Hooke model")
        elast_model = MRcomponents()

    md = Mechanism(
        dim,
        domain,
        space,
        test_path,
        elast_model,
        g_in,
        [ds_g[0]],
        g_out,
        [ds_g[1]],
        dirichlet_bcs,
        kappa,
        alpha,
        eps,
    )

    md.nN = nM

    @fop.region_of(domain)
    def sub_dom_L(x):
        # x < 2*eps and 0.5 - a - eps < y < 0.5 + a + eps
        eps2 = a / 2.0
        return [2.0 * eps2 - x[0], x[1] - 0.5 + a + eps2, 0.5 + a + eps2 - x[1]]

    @fop.region_of(domain)
    def sub_dom_R(x):
        # x > 1.0 - 2*eps and 0.5 - a - eps < y < 0.5 + a + eps
        eps2 = a / 2.0
        return [x[0] - 1.0 + 2.0 * eps2, x[1] - 0.5 + a + eps2, 0.5 + a + eps2 - x[1]]

    md.sub = [sub_dom_L.expression(), sub_dom_R.expression()]

    centers = [(0.0, 0.25), (0.0, 0.75)]
    centers += [(i * 1.0 / 5.0, 1.0) for i in range(1, 6)]
    centers += [(0.1 + i * 0.2, 5.0 / 6.0) for i in range(5)]
    centers += [(i * 1.0 / 5.0, 2.0 / 3.0) for i in range(6)]
    centers += [(0.1 + i * 0.2, 0.5) for i in range(1, 4)]
    centers += [(i * 1.0 / 5.0, 1.0 / 3.0) for i in range(6)]
    centers += [(0.1 + i * 0.2, 1.0 / 6.0) for i in range(5)]
    centers += [(i * 1.0 / 5.0, 0.0) for i in range(1, 6)]
    centers = np.array(centers)
    radii = np.repeat(0.05, centers.shape[0])
    md.create_initial_level(centers, radii)
    md.save_initial_level(comm)

    md.runDP(
        niter=niter,
        dfactor=1e-2,
        lv_iter=(10, 24),
        lv_time=(1e-3, 0.1),
        cost_tol=1e-3,
        smooth=True,
        reinit_step=4,
        reinit_pars=(10, 5e-3),
        random_pars=(125, 0.1),
    )


def KVtest(mod, test_path, kappa, alpha, eps, niter, elastic_pars, forces, nM=0):

    dim, rank_dim, mesh_size = 2, 2, 0.008
    a, b = 0.05, 0.05
    vertices = np.array(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.5 - a),
            (1.0, 0.5 + a),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0 - b),
            (0.0, 0.5 + a),
            (0.0, 0.5 - a),
            (0.0, b),
        ]
    )
    dirB_idx, dirB_mkr = [10], 1
    dirT_idx, dirT_mkr = [6], 2
    neuL_idx, neuL_mkr = [8], 3
    neuR_idx, neuR_mkr = [3], 4
    boundary_parts = [
        (dirB_idx, dirB_mkr, "dir_bottom"),
        (dirT_idx, dirT_mkr, "dir_top"),
        (neuL_idx, neuL_mkr, "neu_left"),
        (neuR_idx, neuR_mkr, "neu_right"),
    ]
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )
    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("Kohn-Vogelius")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    space = fop.create_space(domain, "CG", rank_dim)
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dirB_mkr, dirT_mkr], rank_dim
    )
    ds_g = fop.marked_ds(domain, boundary_tags, [neuL_mkr, neuR_mkr])

    Ym, Pr = elastic_pars
    force_in, force_out = forces
    g_in, g_out = [force_in], [force_out]

    if mod == "Hooke":
        elast_model = Hookecomponents(Ym, Pr)
    elif mod == "SVK":
        elast_model = SVKcomponents(Ym, Pr)
    elif mod == "MR":
        elast_model = MRcomponents()

    md = KohnVogelius(
        dim,
        domain,
        space,
        test_path,
        elast_model,
        g_in,
        [ds_g[0]],
        g_out,
        [ds_g[1]],
        dirichlet_bcs,
        kappa,
        alpha,
        eps,
    )

    md.nN = nM

    @fop.region_of(domain)
    def sub_dom_L(x):
        # x < 2*eps and 0.5 - a - eps < y < 0.5 + a + eps
        eps2 = a / 2.0
        return [2.0 * eps2 - x[0], x[1] - 0.5 + a + eps2, 0.5 + a + eps2 - x[1]]

    @fop.region_of(domain)
    def sub_dom_R(x):
        # x > 1.0 - 2*eps and 0.5 - a - eps < y < 0.5 + a + eps
        eps2 = a / 2.0
        return [x[0] - 1.0 + 2.0 * eps2, x[1] - 0.5 + a + eps2, 0.5 + a + eps2 - x[1]]

    @fop.region_of(domain)
    def sub_dom_T(x):
        eps = a / 2.0
        return [2.0 * eps - x[0], x[1] - 1.0 + b + eps]

    @fop.region_of(domain)
    def sub_dom_B(x):
        eps = a / 2.0
        return [2.0 * eps - x[0], b + eps - x[1]]

    md.sub = [sub_dom_L.expression(), sub_dom_R.expression()]

    centers = [(0.0, 0.25), (0.0, 0.75)]
    centers += [(i * 1.0 / 5.0, 1.0) for i in range(1, 6)]
    centers += [(0.1 + i * 0.2, 5.0 / 6.0) for i in range(5)]
    centers += [(i * 1.0 / 5.0, 2.0 / 3.0) for i in range(6)]
    centers += [(0.1 + i * 0.2, 0.5) for i in range(1, 4)]
    centers += [(i * 1.0 / 5.0, 1.0 / 3.0) for i in range(6)]
    centers += [(0.1 + i * 0.2, 1.0 / 6.0) for i in range(5)]
    centers += [(i * 1.0 / 5.0, 0.0) for i in range(1, 6)]

    centers = np.array(centers)
    radii = np.repeat(0.05, centers.shape[0])
    md.create_initial_level(centers, radii)
    md.save_initial_level(comm)

    md.runDP(
        niter=niter,
        dfactor=1e-2,
        lv_iter=(10, 20),
        lv_time=(1e-3, 0.1),
        cost_tol=1e-3,
        smooth=True,
        reinit_step=4,
        reinit_pars=(6, 0.01),
        random_pars=(26, 0.05),
    )


def KVdeformations(test_path, niter, forces, alpha):

    dim = 2
    rank_dim = 2
    dirB_idx, dirB_mkr = [10], 1
    dirT_idx, dirT_mkr = [6], 2
    neuL_idx, neuL_mkr = [8], 3
    neuR_idx, neuR_mkr = [3], 4

    filename = test_path / "domain.msh"
    domain, _, boundary_tags = fop.read_gmsh(filename, comm, 2)
    fop.all_connectivities(domain)

    phi = fop.read_level_set_function(test_path, domain, niter)
    space = fop.create_space(domain, "CG", rank_dim)

    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dirB_mkr, dirT_mkr], rank_dim
    )
    ds_g = fop.marked_ds(domain, boundary_tags, [neuL_mkr, neuR_mkr])
    force_in, force_out = forces
    g_in, g_out = [force_in], [force_out]

    elast_model = Hookecomponents()

    md1 = Compliance(
        dim,
        domain,
        space,
        test_path,
        elast_model,
        [g_in],
        [[ds_g[0]]],
        dirichlet_bcs,
        alpha,
    )

    md2 = Compliance(
        dim,
        domain,
        space,
        test_path,
        elast_model,
        [g_out],
        [[ds_g[1]]],
        dirichlet_bcs,
        alpha,
    )

    nN = 20

    names_in = [f"uin{i:02}" for i in range(1, nN)]
    names_out = [f"uout{i:02}" for i in range(1, nN)]
    factor = np.linspace(0.0, 1.0, nN)[1:]
    uhs = []

    for fc, nm in zip(factor, names_in):
        print(f"> Factor = {fc}")
        md1.update_gs([[(fc * force_in[0], fc * force_in[1])]])
        uhs.append(fop.SolveLinearProblem(space, md1.pde(phi)[0], nm))

    for fc, nm in zip(factor, names_out):
        print(f"> Factor = {fc}")
        md2.update_gs([[(fc * force_out[0], fc * force_out[1])]])
        uhs.append(fop.SolveLinearProblem(space, md2.pde(phi)[0], nm))

    fop.save_functions(comm, domain, [phi] + uhs, test_path / "deformations.xdmf")


test_functions = {
    "01": lambda: cantilever(
        "Hooke", Path("../results/Elasticity/t01/"), load=-10.0, alpha=0.25
    ),
    "02": lambda: cantilever(
        "SVK", Path("../results/Elasticity/t02/"), load=-10.0, alpha=0.25, nM=8
    ),
    "03": lambda: cantilever(
        "MR", Path("../results/Elasticity/t03/"), load=-10.0, alpha=0.25, nM=16
    ),
    "defo_cant_01": lambda: deformation_cantilever(
        "Hooke", Path("../results/Elasticity/t01/"), 119
    ),
    "defo_cant_02": lambda: deformation_cantilever(
        "SVK", Path("../results/Elasticity/t02/"), 115
    ),
    "defo_cant_03": lambda: deformation_cantilever(
        "MR", Path("../results/Elasticity/t03/"), 130
    ),
    "04": lambda: cantilever(
        "SVK",
        Path("../results/Elasticity/t04/"),
        load=-(10.0 * 1.633),
        alpha=(1.633**2) * 0.25,
        nM=8,
    ),
    "05": lambda: cantilever(
        "MR",
        Path("../results/Elasticity/t05/"),
        load=-(10.0 * 1.7588),
        alpha=(1.7588**2) * 0.25,
        nM=18,
    ),
    "11": lambda: inverter(
        mod="Hooke",
        test_path=Path("../results/Elasticity/t11/"),
        kappa=[4.0],
        alpha=0.05,
        eps=5e-3,
        niter=400,
        elastic_pars=(50.0, 0.4),
        forces=(2.0, 3.0),
    ),
    "12": lambda: inverter(
        mod="Hooke",
        test_path=Path("../results/Elasticity/t12/"),
        kappa=[1.0],
        alpha=0.3,
        eps=1e-2,
        niter=200,
        elastic_pars=(200.0, 0.3),
        forces=(8.0, 14.0),
    ),
    "13": lambda: inverter(
        mod="Hooke",
        test_path=Path("../results/Elasticity/t13/"),
        kappa=[2.0],
        alpha=0.25,
        eps=1e-2,
        niter=200,
        elastic_pars=(200.0, 0.3),
        forces=(8.0, 14.0),
    ),
    "14": lambda: inverter(
        mod="Hooke",
        test_path=Path("../results/Elasticity/t14/"),
        kappa=[3.0],
        alpha=0.25,
        eps=1e-2,
        niter=200,
        elastic_pars=(200.0, 0.3),
        forces=(10.0, 12.0),
    ),
    "15": lambda: inverter(
        mod="Hooke",
        test_path=Path("../results/Elasticity/t15/"),
        kappa=[18.0],
        alpha=0.1,
        eps=1e-2,
        niter=200,
        elastic_pars=(200.0, 0.3),
        forces=(2.0, 4.0),
    ),
    "16": lambda: inverter(
        mod="SVK",
        test_path=Path("../results/Elasticity/t16/"),
        kappa=[2.0],
        alpha=0.25,
        eps=1e-2,
        niter=200,
        elastic_pars=(200.0, 0.3),
        forces=(8.0, 14.0),
        nM=12,
    ),
    "41": lambda: KVtest(
        mod="Hooke",
        test_path=Path("../results/Elasticity/t41/"),
        kappa=1.0,
        alpha=0.075,
        eps=3e-3,
        niter=300,
        elastic_pars=(300.0, 0.4),
        forces=((6.0, 0.0), (-6.0, 0.0)),
    ),
    "42": lambda: KVtest(
        mod="Hooke",
        test_path=Path("../results/Elasticity/t42/"),
        kappa=1.0,
        alpha=0.075,
        eps=3e-3,
        niter=300,
        elastic_pars=(300.0, 0.4),
        forces=((6.0, 0.0), (0.0, -6.0)),
    ),
    "43": lambda: KVtest(
        mod="Hooke",
        test_path=Path("../results/Elasticity/t43/"),
        kappa=1.0,
        alpha=0.075,
        eps=3e-3,
        niter=300,
        elastic_pars=(300.0, 0.4),
        forces=((6.0, 0.0), (6.0, 0.0)),
    ),
    "defo_KV": lambda: KVdeformations(
        Path("../results/Elasticity/t41/"),
        206,
        ((15.0, 0.0), (-15.0, 0.0)),
        alpha=0.075,
    ),
}


def main():

    import sys

    if len(sys.argv) != 2:
        print("Usage: python Elasticity.py <test_id>")
        print("Example: python Elasticity.py 01")
        return

    test_id = sys.argv[1]
    func = test_functions.get(test_id)

    if func:
        func()
    else:
        print(f"Test '{test_id}' not recognized.")


if __name__ == "__main__":
    main()
