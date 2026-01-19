import formopt as fop
from Elasticity_models import (
    SVKcomponents,
    MRcomponents,
    Hookecomponents,
    Elasticity,
    Compliance,
    Gripping,
)

import numpy as np
from pathlib import Path
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


def compliance():

    test_path = Path("../results/t101/")
    test_name = "Compliance 2D"
    dim = 2
    rank_dim = 2
    mesh_size = 0.012
    vertices = np.array(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 0.45), (2.0, 0.55), (2.0, 1.0), (0.0, 1.0)]
    )
    dir_idx, dir_mkr = [6], 1
    neu_idx, neu_mkr = [3], 2
    boundary_parts = [(dir_idx, dir_mkr, "dir"), (neu_idx, neu_mkr, "neu")]
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )
    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    space = fop.create_space(domain, "CG", rank_dim)
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr])
    alpha = 0.25
    g = [(0.0, -10.0)]

    elast_model = Hookecomponents()
    cost_functs = Compliance(g, ds_g)
    # elast_model = SVKcomponents()
    # elast_model = MRcomponents()

    md = Elasticity(
        dim,
        domain,
        space,
        test_path,
        elast_model,
        cost_functs,
        g,
        ds_g,
        dirichlet_bcs,
        alpha,
    )

    md.nN = 10  # for MR
    md.nN = 5  # for SVK

    @fop.region_of(domain)
    def sub_domain(x):
        # 0.42 < x[1] < 0.58, 1.90 < x[0]
        return [x[1] - 0.42, 0.58 - x[1], x[0] - 1.90]

    md.sub = [sub_domain.expression()]

    centers = [(2.0, 0.35), (2.0, 0.65), (2.0, 0.0), (2.0, 1.0)]
    centers += [(0.0, 0.25), (0.0, 0.5), (0.0, 0.75)]
    centers += [(0.3 + i * 0.7, 0.0) for i in range(3)]
    centers += [(0.65 + i * 0.7, 0.25) for i in range(2)]
    centers += [(0.3 + i * 0.7, 0.5) for i in range(3)]
    centers += [(0.65 + i * 0.7, 0.75) for i in range(2)]
    centers += [(0.3 + i * 0.7, 1.0) for i in range(3)]
    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])
    md.create_initial_level(centers, radii)

    md.runDP(
        niter=150,
        dfactor=0.01,
        lv_iter=(10, 20),
        lv_time=(0.01, 0.1),
        cost_tol=1e-3,
        smooth=True,
        reinit_step=4,
        reinit_pars=(6, 0.01),
    )


def gripping():

    test_path = Path("../results/t106/")
    test_name = "Gripping mechanism 2D"
    dim = 2
    rank_dim = 2
    mesh_size = 0.008
    vertices = np.array(
        [
            (0.0, 0.0),
            (0.9, 0.0),
            (1.0, 0.0),
            (1.0, 0.48),
            (1.0, 0.52),
            (1.0, 1.0),
            (0.9, 1.0),
            (0.0, 1.0),
            (0.0, 0.6),
            (0.1, 0.6),
            (0.1, 0.52),
            (0.1, 0.48),
            (0.1, 0.4),
            (0.0, 0.4),
        ]
    )

    dirR_idx, dirR_mkr = [4], 1
    dirL_idx, dirL_mkr = [11], 2
    neuRT_idx, neuRT_mkr = [6], 3
    neuRB_idx, neuRB_mkr = [2], 4
    neuLT_idx, neuLT_mkr = [9], 5
    neuLB_idx, neuLB_mkr = [13], 6

    boundary_parts = [
        (dirR_idx, dirR_mkr, "dir_right"),
        (dirL_idx, dirL_mkr, "dir_left"),
        (neuRT_idx, neuRT_mkr, "neu_right_top"),
        (neuRB_idx, neuRB_mkr, "neu_right_bottom"),
        (neuLT_idx, neuLT_mkr, "neu_left_top"),
        (neuLB_idx, neuLB_mkr, "neu_left_bottom"),
    ]

    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    space = fop.create_space(domain, "CG", rank_dim)
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dirR_mkr, dirL_mkr], rank_dim
    )
    ds_g = fop.marked_ds(
        domain, boundary_tags, [neuRT_mkr, neuRB_mkr, neuLT_mkr, neuLB_mkr]
    )

    ff = 1.0
    gg = 1.0
    alpha = 0.25
    g = [(0.0, -ff * 10.0), (0.0, ff * 10.0), (0.0, ff * 1.0), (0.0, -ff * 1.0)]
    k = [(0.0, -gg * 1.0), (0.0, gg * 1.0), (0.0, gg * 4.0), (0.0, -gg * 4.0)]

    elast_model = Hookecomponents()
    cost_functs = Gripping(k, ds_g)

    md = Elasticity(
        dim,
        domain,
        space,
        test_path,
        elast_model,
        cost_functs,
        g,
        ds_g,
        dirichlet_bcs,
        alpha,
    )

    @fop.region_of(domain)
    def sub_domain1(x):
        # 0.9 < x[0] < 1.0
        # 0.95 < x[1]
        ineqs = [x[0] - 0.9, 1.0 - x[0], x[1] - 0.95]
        return ineqs

    @fop.region_of(domain)
    def sub_domain2(x):
        # 0.9 < x[0] < 1.0
        # x[1] < 0.05
        ineqs = [x[0] - 0.9, 1.0 - x[0], 0.05 - x[1]]
        return ineqs

    @fop.region_of(domain)
    def sub_domain3(x):
        # 0.0 < x[0] < 0.1
        # 0.35 < x[1] < 0.5
        ineqs = [x[0], 0.1 - x[0], x[1] - 0.35, 0.5 - x[1]]
        return ineqs

    @fop.region_of(domain)
    def sub_domain4(x):
        # 0.0 < x[0] < 0.1
        # 0.65 > x[1] > 0.5
        ineqs = [x[0], 0.1 - x[0], 0.65 - x[1], x[1] - 0.5]
        return ineqs

    md.sub = [
        sub_domain1.expression(),
        sub_domain2.expression(),
        sub_domain3.expression(),
        sub_domain4.expression(),
    ]

    centers = []
    centers += [(i * 0.5 / 7, 1.0) for i in range(8)]
    centers += [(i * 0.5 / 7, 0.0) for i in range(8)]
    centers += [(1.0, 0.2 + i * 0.6 / 9) for i in range(10)]
    centers += [(0.0, i * 0.2 / 3) for i in range(4)]
    centers += [(0.0, 1.0 - i * 0.2 / 3) for i in range(4)]
    centers += [(0.1, 0.5), (0.1, 0.55), (0.1, 0.45), (0.5, 0.5)]
    centers += [(0.0, 0.0), (0.0, 1.0), (1.0, 0.5)]

    centers = np.array(centers)
    radii = np.repeat(0.05, centers.shape[0])
    radii[-4:] = 0.1
    md.create_initial_level(centers, radii)
    md.save_initial_level(comm)

    md.runDP(
        niter=400,
        ctrn_tol=1e-3,
        dfactor=0.01,
        lv_iter=(10, 20),
        lv_time=(0.001, 0.01),
        reinit_step=4,
        reinit_pars=(10, 0.001),
        smooth=True,
    )


gripping()
