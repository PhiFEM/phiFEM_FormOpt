import numpy as np
from pathlib import Path
from mpi4py import MPI

import formopt as fop
from phiFEM_models import LaplacianEnergy, ComplianceVolConstraint, ComplianceVolPenalty


def test_01():
    # Path
    test_path = Path("../results/phiFEM/t01/")

    # Domain
    dim, rank_dim, mesh_size = 2, 1, 0.1
    vertices = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    domain, nbr_tri, _ = fop.create_domain_2d_DP(
        vertices, [], mesh_size, path=test_path
    )
    print(f"> Nbr of triangles = {nbr_tri}")

    # Space for the PDE solution
    space = fop.create_space(domain, "CG", rank_dim)
    # Model Problem
    md = LaplacianEnergy(dim, domain, space, test_path, rank_dim, 25.0)

    # Initial level set function
    centers = [(0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.5, 0.6), (0.7, 0.3), (0.4, 0.7)]
    centers = 10.0 * np.array(centers)
    radii = np.repeat(1.5, centers.shape[0])
    md.create_initial_level(centers, radii, factor=-1.0, ord=2)

    md.phifem_run(
        niter=150,
        ctrn_tol=1e-3,
        dfactor=1.0,
        lv_time=(0.001, 1.0),
        reinit_step=4,
        reinit_pars=(12, 0.05),
        smooth=True,
    )


def test_02():

    test_path = Path("../results/phiFEM/t02/")

    # Domain
    dim, rank_dim, mesh_size = 2, 2, 0.01
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
    print(f"> Nbr of triangles = {nbr_tri}")

    mix_space = fop.create_mixed_space(
        domain,
        ["CG", "CG", "DG"],
        [(rank_dim,), (rank_dim, rank_dim), (rank_dim,)],
        [1, 1, 0],
    )

    # Homogeneous Dirichlet boundary condition for the first subspace
    dir_bc = fop.homogeneous_dirichlet_mixed(
        domain, mix_space.sub(0), boundary_tags, [dir_mkr]
    )

    # Boundary to force application
    dsg = fop.marked_ds(domain, boundary_tags, [neu_mkr])[0]

    volume, g = 1.0, (0.0, -2.0)

    md = ComplianceVolConstraint(
        dim,
        domain,
        mix_space,
        test_path,
        rank_dim,
        g,
        dsg,
        dir_bc,
        volume,
    )

    md.biform_coefs = (0.1, 1.0)

    @fop.region_of(domain)
    def sub_domain(x):
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
    radii = np.repeat(0.1, centers.shape[0])
    md.create_initial_level(centers, radii)

    md.phifem_run(
        niter=250,
        ctrn_tol=1e-3,
        dfactor=1e-1,
        reinit_step=4,
        reinit_pars=(16, 0.01),
        smooth=True,
    )


def test_03():
    test_path = Path("../results/phiFEM/t03/")

    # Domain
    dim, rank_dim, mesh_size = 2, 2, 0.01
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
    print(f"> Nbr of triangles = {nbr_tri}")

    mix_space = fop.create_mixed_space(
        domain,
        ["CG", "CG", "DG"],
        [(rank_dim,), (rank_dim, rank_dim), (rank_dim,)],
        [1, 1, 0],
    )

    # Homogeneous Dirichlet boundary condition for the first subspace
    dir_bc = fop.homogeneous_dirichlet_mixed(
        domain, mix_space.sub(0), boundary_tags, [dir_mkr]
    )

    # Boundary to force application
    dsg = fop.marked_ds(domain, boundary_tags, [neu_mkr])[0]

    alpha, g = 1.663, (0.0, -2.0)

    md = ComplianceVolPenalty(
        dim,
        domain,
        mix_space,
        test_path,
        rank_dim,
        g,
        dsg,
        dir_bc,
        alpha,
    )

    md.biform_coefs = (0.1, 1.0)

    @fop.region_of(domain)
    def sub_domain(x):
        return [x[1] - 0.44, 0.56 - x[1], x[0] - 1.9]

    md.sub = [sub_domain.expression()]

    h = 1.4 / 2.0
    centers = [(2.0, 0.35), (2.0, 0.65), (2.0, 0.0), (2.0, 1.0)]
    centers += [(0.0, 0.25), (0.0, 0.5), (0.0, 0.75)]
    centers += [(0.3 + i * h, 0.0) for i in range(3)]
    centers += [(0.3 + h / 2.0 + i * h, 0.25) for i in range(2)]
    centers += [(0.3 + i * h, 0.5) for i in range(3)]
    centers += [(0.3 + h / 2.0 + i * h, 0.75) for i in range(2)]
    centers += [(0.3 + i * h, 1.0) for i in range(3)]

    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])
    md.create_initial_level(centers, radii)

    md.phifem_run(
        niter=250,
        cost_tol=1e-2,
        dfactor=1e-1,
        reinit_step=4,
        reinit_pars=(16, 0.01),
        smooth=True,
    )


def test_04():
    test_path = Path("../results/phiFEM/t04/")

    # Domain
    dim, rank_dim, mesh_size = 2, 2, 0.01
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
    print(f"> Nbr of triangles = {nbr_tri}")

    mix_space = fop.create_mixed_space(
        domain,
        ["CG", "CG", "DG"],
        [(rank_dim,), (rank_dim, rank_dim), (rank_dim,)],
        [1, 1, 0],
    )

    # Homogeneous Dirichlet boundary condition for the first subspace
    dir_bc = fop.homogeneous_dirichlet_mixed(
        domain, mix_space.sub(0), boundary_tags, [dir_mkr]
    )

    # Boundary to force application
    dsg = fop.marked_ds(domain, boundary_tags, [neu_mkr])[0]

    alpha, g = 40.0, (0.0, -5.6)

    md = ComplianceVolPenalty(
        dim,
        domain,
        mix_space,
        test_path,
        rank_dim,
        g,
        dsg,
        dir_bc,
        alpha,
    )

    md.biform_coefs = (1.0, 100.0)

    @fop.region_of(domain)
    def sub_domain(x):
        return [x[1] - 0.44, 0.56 - x[1], x[0] - 1.9]

    md.sub = [sub_domain.expression()]

    h = 1.4 / 2.0
    centers = [(2.0, 0.35), (2.0, 0.65), (2.0, 0.0), (2.0, 1.0)]
    centers += [(0.0, 0.25), (0.0, 0.5), (0.0, 0.75)]
    centers += [(0.3 + i * h, 0.0) for i in range(3)]
    centers += [(0.3 + h / 2.0 + i * h, 0.25) for i in range(2)]
    centers += [(0.3 + i * h, 0.5) for i in range(3)]
    centers += [(0.3 + h / 2.0 + i * h, 0.75) for i in range(2)]
    centers += [(0.3 + i * h, 1.0) for i in range(3)]

    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])
    md.create_initial_level(centers, radii)

    md.phifem_run(
        niter=150,
        cost_tol=1e-2,
        dfactor=10.0,
        reinit_step=4,
        reinit_pars=(16, 0.01),
        smooth=True,
    )


test_functions = {
    "01": test_01,
    "02": test_02,
    "03": test_03,
    "04": test_04,
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
