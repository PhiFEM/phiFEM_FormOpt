import formopt as fop
from Elasticity_models import Hookecomponents, SVKcomponents, MRcomponents, Compliance

import numpy as np
from pathlib import Path
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


def cantilever(mod, test_path, load, alpha, nM=0):
    """
    Cantilever with one load
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
        niter=150,
        dfactor=1e-2,
        lv_iter=(12, 24),
        lv_time=(1e-3, 1.0),
        cost_tol=1e-3,
        smooth=True,
        reinit_step=4,
        reinit_pars=(6, 0.01),
        random_pars=(26, 0.1),
    )


def multiload(mod, test_path, dd, nM=0):
    dim = 2
    rank_dim = 2
    mesh_size = 0.008

    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [1.0, 0.9], [1.0, 1.0], [0.0, 1.0]]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx_bot, neu_mkr_bot = [2], 2
    neu_idx_top, neu_mkr_top = [4], 3

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx_bot, neu_mkr_bot, "neu_bot"),
        (neu_idx_top, neu_mkr_top, "neu_top"),
    ]

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

    ds0 = fop.marked_ds(domain, boundary_tags, [neu_mkr_bot, neu_mkr_top])
    alpha = 0.5
    g = [[(0.0, -10.0)], [(0.0, -10.0)]]
    ds_g = [[ds0[0]], [ds0[1]]]

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
        g,
        ds_g,
        dirichlet_bcs,
        alpha,
        pty=False,
    )

    md.nN = nM

    centers = []
    centers += [(i * 0.2, 0.25) for i in range(5)]
    centers += [(0.1 + i * 0.2, 0.5) for i in range(5)]
    centers += [(i * 0.2, 0.75) for i in range(5)]
    centers = np.array(centers)
    radii = np.repeat(0.06, centers.shape[0])

    md.create_initial_level(centers, radii)
    md.save_initial_level(comm)

    md.runDP(
        niter=250,
        dfactor=1e-2,
        lv_iter=(12, 24),
        lv_time=(1e-3, 1.0),
        cost_tol=1e-3,
        lgrn_tol=1e-3,
        ctrn_tol=1e-3,
        smooth=True,
        reinit_step=4,
        reinit_pars=(8, 1e-3),
    )


# def gripping():

#     test_path = Path("../results/t106/")
#     test_name = "Gripping mechanism 2D"
#     dim = 2
#     rank_dim = 2
#     mesh_size = 0.008
#     vertices = np.array(
#         [
#             (0.0, 0.0),
#             (0.9, 0.0),
#             (1.0, 0.0),
#             (1.0, 0.48),
#             (1.0, 0.52),
#             (1.0, 1.0),
#             (0.9, 1.0),
#             (0.0, 1.0),
#             (0.0, 0.6),
#             (0.1, 0.6),
#             (0.1, 0.52),
#             (0.1, 0.48),
#             (0.1, 0.4),
#             (0.0, 0.4),
#         ]
#     )

#     dirR_idx, dirR_mkr = [4], 1
#     dirL_idx, dirL_mkr = [11], 2
#     neuRT_idx, neuRT_mkr = [6], 3
#     neuRB_idx, neuRB_mkr = [2], 4
#     neuLT_idx, neuLT_mkr = [9], 5
#     neuLB_idx, neuLB_mkr = [13], 6

#     boundary_parts = [
#         (dirR_idx, dirR_mkr, "dir_right"),
#         (dirL_idx, dirL_mkr, "dir_left"),
#         (neuRT_idx, neuRT_mkr, "neu_right_top"),
#         (neuRB_idx, neuRB_mkr, "neu_right_bottom"),
#         (neuLT_idx, neuLT_mkr, "neu_left_top"),
#         (neuLB_idx, neuLB_mkr, "neu_left_bottom"),
#     ]

#     output = fop.create_domain_2d_DP(
#         vertices, boundary_parts, mesh_size, path=test_path, plot=False
#     )

#     domain, nbr_tri, boundary_tags = output

#     if rank == 0:
#         print("\n\t" + test_name + "\n")
#         print(f"> Path = {test_path}")
#         print(f"> Nbr of triangles = {nbr_tri}")

#     space = fop.create_space(domain, "CG", rank_dim)
#     dirichlet_bcs = fop.homogeneous_dirichlet(
#         domain, space, boundary_tags, [dirR_mkr, dirL_mkr], rank_dim
#     )
#     ds_g = fop.marked_ds(
#         domain, boundary_tags, [neuRT_mkr, neuRB_mkr, neuLT_mkr, neuLB_mkr]
#     )

#     ff = 1.0
#     gg = 1.0
#     alpha = 0.25
#     g = [(0.0, -ff * 10.0), (0.0, ff * 10.0), (0.0, ff * 1.0), (0.0, -ff * 1.0)]
#     k = [(0.0, -gg * 1.0), (0.0, gg * 1.0), (0.0, gg * 2.0), (0.0, -gg * 2.0)]

#     elast_model = Hookecomponents()
#     cost_functs = Gripping(k, ds_g)

#     md = Elasticity(
#         dim,
#         domain,
#         space,
#         test_path,
#         elast_model,
#         cost_functs,
#         g,
#         ds_g,
#         dirichlet_bcs,
#         alpha,
#     )

#     @fop.region_of(domain)
#     def sub_domain1(x):
#         # 0.9 < x[0] < 1.0
#         # 0.95 < x[1]
#         ineqs = [x[0] - 0.9, 1.0 - x[0], x[1] - 0.95]
#         return ineqs

#     @fop.region_of(domain)
#     def sub_domain2(x):
#         # 0.9 < x[0] < 1.0
#         # x[1] < 0.05
#         ineqs = [x[0] - 0.9, 1.0 - x[0], 0.05 - x[1]]
#         return ineqs

#     @fop.region_of(domain)
#     def sub_domain3(x):
#         # 0.0 < x[0] < 0.1
#         # 0.35 < x[1] < 0.5
#         ineqs = [x[0], 0.1 - x[0], x[1] - 0.35, 0.5 - x[1]]
#         return ineqs

#     @fop.region_of(domain)
#     def sub_domain4(x):
#         # 0.0 < x[0] < 0.1
#         # 0.65 > x[1] > 0.5
#         ineqs = [x[0], 0.1 - x[0], 0.65 - x[1], x[1] - 0.5]
#         return ineqs

#     md.sub = [
#         sub_domain1.expression(),
#         sub_domain2.expression(),
#         sub_domain3.expression(),
#         sub_domain4.expression(),
#     ]

#     centers = []
#     centers += [(i * 0.5 / 7, 1.0) for i in range(8)]
#     centers += [(i * 0.5 / 7, 0.0) for i in range(8)]
#     centers += [(1.0, 0.2 + i * 0.6 / 9) for i in range(10)]
#     centers += [(0.0, i * 0.2 / 3) for i in range(4)]
#     centers += [(0.0, 1.0 - i * 0.2 / 3) for i in range(4)]
#     centers += [(0.1, 0.5), (0.1, 0.55), (0.1, 0.45), (0.5, 0.5)]
#     centers += [(0.0, 0.0), (0.0, 1.0), (1.0, 0.5)]

#     centers = np.array(centers)
#     radii = np.repeat(0.05, centers.shape[0])
#     radii[-4:] = 0.1
#     md.create_initial_level(centers, radii)
#     md.save_initial_level(comm)

#     md.runDP(
#         niter=600,
#         ctrn_tol=1e-3,
#         dfactor=0.01,
#         lv_iter=(10, 20),
#         lv_time=(0.001, 0.01),
#         reinit_step=4,
#         reinit_pars=(10, 0.001),
#         smooth=True,
#     )


# def symmetric_gripping():

#     test_path = Path("../results/t107/")
#     test_name = "Gripping mechanism sym 2D"
#     dim = 2
#     rank_dim = 2
#     mesh_size = 0.009
#     vertices = np.array(
#         [
#             (0.0, 0.0),
#             (1.0, 0.0),
#             (1.0, 0.06),
#             (1.0, 0.5),
#             (0.0, 0.5),
#             (0.0, 0.42),
#             (0.0, 0.06),
#         ]
#     )

#     dirB_idx, dirB_mkr = [1], 1
#     dirTL_idx, dirTL_mkr = [5], 2
#     neuBL_idx, neuBL_mkr = [7], 3
#     neuBR_idx, neuBR_mkr = [2], 4

#     boundary_parts = [
#         (dirB_idx, dirB_mkr, "dir_bottom"),
#         (dirTL_idx, dirTL_mkr, "dir_top_left"),
#         (neuBL_idx, neuBL_mkr, "neu_bottom_left"),
#         (neuBR_idx, neuBR_mkr, "neu_bottom_right"),
#     ]

#     output = fop.create_domain_2d_DP(
#         vertices, boundary_parts, mesh_size, path=test_path, plot=False
#     )

#     domain, nbr_tri, boundary_tags = output

#     if rank == 0:
#         print("\n\t" + test_name + "\n")
#         print(f"> Path = {test_path}")
#         print(f"> Nbr of triangles = {nbr_tri}")

#     space = fop.create_space(domain, "CG", rank_dim)
#     dirTL = fop.homogeneous_dirichlet(
#         domain, space, boundary_tags, [dirTL_mkr], rank_dim
#     )

#     dirB_y = fop.homogeneous_dirichlet_y_coord(domain, space, boundary_tags, [dirB_mkr])

#     ds_g = fop.marked_ds(domain, boundary_tags, [neuBL_mkr, neuBR_mkr])

#     ff = 9.0
#     alpha = 0.1
#     g = [(10.0, 0.0), (5.0, 0.0)]
#     k = [(10.0, 0.0), (ff * 5.0, 0.0)]

#     elast_model = Hookecomponents()
#     cost_functs = Gripping(k, ds_g)

#     md = Elasticity(
#         dim,
#         domain,
#         space,
#         test_path,
#         elast_model,
#         cost_functs,
#         g,
#         ds_g,
#         [dirTL[0], dirB_y[0]],
#         alpha,
#     )

#     @fop.region_of(domain)
#     def sub_domain1(x):
#         # x[0] < 0.08
#         # x[1] < 0.08
#         ineqs = [0.08 - x[0], 0.08 - x[1]]
#         return ineqs

#     @fop.region_of(domain)
#     def sub_domain2(x):
#         # 0.92 < x[0]
#         # x[1] < 0.08
#         ineqs = [x[0] - 0.94, 0.08 - x[1]]
#         return ineqs

#     @fop.region_of(domain)
#     def sub_domain3(x):
#         # x[0] < 0.08
#         # 0.42 < x[1]
#         ineqs = [0.08 - x[0], x[1] - 0.42]
#         return ineqs

#     md.sub = []
#     # md.sub = [
#     #     sub_domain1.expression(),
#     #     sub_domain2.expression(),
#     #     sub_domain3.expression(),
#     # ]

#     centers = [(0.0, 0.75), (2.0, 0.75), (0.0, 0.25), (2.0, 0.25)]
#     centers += [(0.3 + i * 0.7, 0.0) for i in range(3)]
#     centers += [(0.65 + i * 0.7, 0.25) for i in range(2)]
#     centers += [(0.3 + i * 0.7, 0.5) for i in range(3)]
#     centers += [(0.65 + i * 0.7, 0.75) for i in range(2)]
#     centers += [(0.3 + i * 0.7, 1.0) for i in range(3)]

#     centers = np.array(centers) / 2.0
#     radii = np.repeat(0.05, centers.shape[0])

#     md.create_initial_level(centers, radii)
#     md.save_initial_level(comm)

#     md.runDP(
#         niter=200,
#         cost_tol=1e-3,
#         dfactor=0.01,
#         lv_iter=(10, 20),
#         lv_time=(0.01, 0.1),
#         reinit_step=4,
#         reinit_pars=(10, 0.001),
#         smooth=True,
#     )


test_functions = {
    "01": lambda: cantilever("Hooke", Path("../results/t101/"), load=-10.0, alpha=0.2),
    "02": lambda: cantilever(
        "SVK", Path("../results/t102/"), load=-10.0, alpha=0.2, nM=8
    ),
    "03": lambda: cantilever(
        "MR", Path("../results/t103/"), load=-10.0, alpha=0.2, nM=16
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
