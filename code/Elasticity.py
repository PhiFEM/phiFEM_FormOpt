import formopt as fop
from Elasticity_models import (
    Hookecomponents,
    SVKcomponents,
    MRcomponents,
    Compliance,
    Mechanism,
)

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
        random_pars=(26, 0.075),
    )


def gripping(test_path, alpha=0.05, beta=0.005, kappa=4.5, nM=0):
    """
    Gripping mechanism
    """


def inverter2(test_path, kappa, alpha, beta, nM=0):

    dim = 2
    rank_dim = 2
    mesh_size = 0.009
    a, b = 0.05, 0.05
    vertices = np.array(
        [
            (0.0, 0.5),
            (1.0, 0.5),
            (1.0, 0.5 + a),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0 - b),
            (0.0, 0.5 + a),
        ]
    )

    dir_base_idx, dir_base_mkr = [1], 1  # base
    dirC_idx, dirC_mkr = [5], 2  # corner
    neuL_idx, neuL_mkr = [7], 3  # left
    neuR_idx, neuR_mkr = [2], 4  # right

    boundary_parts = [
        (dir_base_idx, dir_base_mkr, "dir_base"),
        (dirC_idx, dirC_mkr, "dir_corner"),
        (neuL_idx, neuL_mkr, "neu_left"),
        (neuR_idx, neuR_mkr, "neu_right"),
    ]

    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print(f"> Nbr of triangles = {nbr_tri}")

    space = fop.create_space(domain, "CG", rank_dim)
    dir_bc = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dirC_mkr], rank_dim
    )
    dir_bc_y = fop.homogeneous_dirichlet_y_coord(
        domain, space, boundary_tags, [dir_base_mkr]
    )
    ds_g = fop.marked_ds(domain, boundary_tags, [neuL_mkr, neuR_mkr])

    factor = 0.4
    g_in = [(factor * 10.0, 0.0)]
    g_out = [(factor * 5.0, 0.0)]

    elast_model = Hookecomponents(Ym=300.0, Pr=0.4)

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
        [dir_bc[0], dir_bc_y[0]],
        kappa,
        alpha,
        beta,
    )

    @fop.region_of(domain)
    def sub_dom_L(x):
        # x < 2*eps
        # 0.5 - a - eps < y < 0.5 + a + eps
        eps = a / 2.0
        return [2.0 * eps - x[0], x[1] - 0.5 + a + eps, 0.5 + a + eps - x[1]]

    @fop.region_of(domain)
    def sub_dom_R(x):
        # x > 1.0 - 2*eps
        # 0.5 - a - eps < y < 0.5 + a + eps
        eps = a / 2.0
        return [x[0] - 1.0 + 2.0 * eps, x[1] - 0.5 + a + eps, 0.5 + a + eps - x[1]]

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
        niter=500,
        dfactor=1e-2,
        lv_iter=(10, 20),
        lv_time=(1e-3, 0.1),
        cost_tol=1e-3,
        smooth=True,
        reinit_step=4,
        reinit_pars=(6, 0.01),
        random_pars=(26, 0.1),
    )


def inverter(test_path, kappa, alpha, beta, nM=0):
    """
    Inverter mechanism
    """
    dim = 2
    rank_dim = 2
    mesh_size = 0.009
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
        print(f"> Nbr of triangles = {nbr_tri}")

    space = fop.create_space(domain, "CG", rank_dim)
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dirB_mkr, dirT_mkr], rank_dim
    )
    ds_g = fop.marked_ds(domain, boundary_tags, [neuL_mkr, neuR_mkr])

    factor = 0.4
    g_in = [(factor * 10.0, 0.0)]
    g_out = [(factor * 5.0, 0.0)]

    elast_model = Hookecomponents(Ym=300.0, Pr=0.4)

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
        beta,
    )

    @fop.region_of(domain)
    def sub_dom_L(x):
        # x < 2*eps
        # 0.5 - a - eps < y < 0.5 + a + eps
        eps = a / 2.0
        return [2.0 * eps - x[0], x[1] - 0.5 + a + eps, 0.5 + a + eps - x[1]]

    @fop.region_of(domain)
    def sub_dom_R(x):
        # x > 1.0 - 2*eps
        # 0.5 - a - eps < y < 0.5 + a + eps
        eps = a / 2.0
        return [x[0] - 1.0 + 2.0 * eps, x[1] - 0.5 + a + eps, 0.5 + a + eps - x[1]]

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
        niter=400,
        dfactor=1e-2,
        lv_iter=(10, 20),
        lv_time=(1e-3, 0.1),
        cost_tol=1e-3,
        smooth=True,
        reinit_step=4,
        reinit_pars=(6, 0.01),
        random_pars=(26, 0.1),
    )


test_functions = {
    "01": lambda: cantilever("Hooke", Path("../results/t101/"), load=-10.0, alpha=0.25),
    "02": lambda: cantilever(
        "SVK", Path("../results/t102/"), load=-10.0, alpha=0.25, nM=8
    ),
    "03": lambda: cantilever(
        "MR", Path("../results/t103/"), load=-10.0, alpha=0.25, nM=16
    ),
    "04": lambda: inverter(Path("../results/t104/"), kappa=5.0, alpha=0.05, beta=1e-3),
    "05": lambda: inverter2(
        Path("../results/t105/"), kappa=5.0, alpha=0.05 / 2.0, beta=5e-3
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
