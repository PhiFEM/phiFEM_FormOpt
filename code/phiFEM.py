import numpy as np
from pathlib import Path
from mpi4py import MPI

import formopt as fop
from phiFEM_models import LaplacianEnergy


def test_01():
    # Path
    test_path = Path("../results/phiFem/t01/")

    # Domain
    dim, rank_dim, mesh_size = 2, 1, 0.12
    vertices = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    domain, _, _ = fop.create_domain_2d_DP(vertices, [], mesh_size, path=test_path)

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
        niter=700,
        ctrn_tol=1e-3,
        dfactor=0.5,
        lv_time=(0.001, 1.0),
        reinit_step=4,
        reinit_pars=(12, 0.05),
        smooth=True,
    )


def test_02():
    test_path = Path("../results/phiFem/t02/")

    # Domain
    dim, rank_dim, mesh_size = 2, 1, 0.08
    vertices = np.array([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)])
    domain, _, _ = fop.create_domain_2d_DP(
        vertices, [], mesh_size, path=test_path, plot=True
    )

    centers = [(i * 0.1, 0.5) for i in range(20)]
    centers = np.array(centers)
    radii = np.repeat(0.1, centers.shape[0])

    phi = fop.get_initial_level(domain, centers, radii)
    from mesh_scripts import compute_tags_measures

    print(domain.topology.cell_name())
    cells_tags, facets_tags, _, ds_out, _, _ = compute_tags_measures(
        domain, phi, 1, box_mode=True
    )


test_functions = {
    "01": test_01,
    "02": test_02,
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
