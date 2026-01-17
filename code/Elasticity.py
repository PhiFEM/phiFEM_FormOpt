import formopt as fop
from models import SVKcomponents, MRcomponents, Hookecomponents, Elasticity

import numpy as np
from pathlib import Path
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


def test_101():

    test_path = Path("../results/t103/")
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
    space = fop.create_space(domain, "CG", rank_dim)
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )

    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr])

    alpha = 0.25
    g = [(0.0, -10.0)]

    # elast_model = Hookecomponents()
    # elast_model = SVKcomponents()
    elast_model = MRcomponents()

    md = Elasticity(
        dim, domain, space, test_path, elast_model, g, ds_g, dirichlet_bcs, alpha
    )
    md.nN = 10

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


test_101()
