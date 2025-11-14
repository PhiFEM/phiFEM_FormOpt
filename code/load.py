import sys
import numpy as np

from plots import (
    plot_basic,
    see_initial_guess,
    plot_results_for_doc,
    plot_bars,
    plot_lag2,
)

from pathlib import Path


def test_01():

    print("\n\tSymmetric cantilever 2D (Data Parallelism)\n")
    test_path = Path("../results/t01/")

    vertices = np.array(
        [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 0.45),
            (2.0, 0.55),
            (2.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx, neu_mkr = [3], 2

    dir = np.array(vertices[(dir_idx[0] - 1) : (dir_idx[-1] + 1)])
    neu = np.array(vertices[(neu_idx[0] - 1) : (neu_idx[-1] + 1)])
    boundaries = [(dir, "red"), (neu, "deepskyblue")]

    # plot_basic(test_path)

    plot_results_for_doc(
        test_path, 0, [[0, 2], [0, 1]], 8, Path("../tex/compli1_0.png"), boundaries
    )

    last_iter = 58
    plot_results_for_doc(
        test_path,
        last_iter,
        [[0, 2], [0, 1]],
        8,
        Path("../tex/compli1_1.png"),
        boundaries,
    )


def test_02():
    print("\n\tSymmetric cantilever 3D\n")


def test_03():

    print("\n\tMultiple load cases (Data Parallelism)\n")
    test_path = Path("../results/t03/")

    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.1],
            [1.0, 0.9],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx_bot, neu_mkr_bot = [2], 2
    neu_idx_top, neu_mkr_top = [4], 3

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx_bot, neu_mkr_bot, "neu_bot"),
        (neu_idx_top, neu_mkr_top, "neu_top"),
    ]

    dir = np.array(vertices[(dir_idx[0] - 1) : (dir_idx[-1] + 1)])
    neu_bot = np.array(vertices[(neu_idx_bot[0] - 1) : (neu_idx_bot[-1] + 1)])
    neu_top = np.array(vertices[(neu_idx_top[0] - 1) : (neu_idx_top[-1] + 1)])
    boundaries = [(dir, "red"), (neu_bot, "deepskyblue"), (neu_top, "deepskyblue")]

    # plot_basic(test_path)

    plot_results_for_doc(
        test_path, 0, [[0, 1], [0, 1]], 8, Path("../tex/compli3_0.png"), boundaries
    )

    last_iter = 69
    plot_results_for_doc(
        test_path,
        last_iter,
        [[0, 1], [0, 1]],
        8,
        Path("../tex/compli3_1.png"),
        boundaries,
    )


def test_04():

    print("\n\tMultiple load cases (Task Parallelism)\n")
    test_path = Path("../results/t04/")

    plot_basic(test_path)


def test_05():

    print("\n\tMultiple load cases (Task Parallelism)\n")
    test_path = Path("../results/t05/")

    plot_basic(test_path)


def test_06():

    print("\n\tElasticity inverse problem (Data Parallelism)\n")
    test_path = Path("../results/t06/")

    def semi_ellipse(a, b, eps, npts):
        """
        Coordinates of a ellipse
        crossing the x-axis
        """
        t_ = np.arcsin((b - eps) / b)
        t = np.linspace(-t_, np.pi + t_, npts)
        x = a * np.cos(t)
        y = b * np.sin(t) + (b - eps)
        return x, y

    npts = 80
    part = npts // 4

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.2, npts))

    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part // 2 + 1), 2
    neu_idxA, neu_mkrA = part // 2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part // 2 + np.arange(part + 1, 2 * part + 1), 4
    neu_idxC, neu_mkrC = part // 2 + np.arange(2 * part + 1, 3 * part + 1), 5
    bL_idx, bL_mkr = np.arange(part // 2 + 3 * part + 1, npts), 6

    vertices = np.concatenate((vertices, [vertices[0]]))
    dir = np.array(vertices[(dir_idx[0] - 1) : (dir_idx[-1] + 1)])
    bR = np.array(vertices[(bR_idx[0] - 1) : (bR_idx[-1] + 1)])
    neuA = np.array(vertices[(neu_idxA[0] - 1) : (neu_idxA[-1] + 1)])
    neuB = np.array(vertices[(neu_idxB[0] - 1) : (neu_idxB[-1] + 1)])
    neuC = np.array(vertices[(neu_idxC[0] - 1) : (neu_idxC[-1] + 1)])
    bL = np.array(vertices[(bL_idx[0] - 1) : (bL_idx[-1] + 1)])

    boundaries = [
        (dir, "red"),
        (bR, "yellow"),
        (neuA, "blue"),
        (neuB, "green"),
        (neuC, "orange"),
        (bL, "yellow"),
    ]

    centers = np.array([(0.0, 0.4)])
    radii = np.array([0.15])

    see_initial_guess(
        (centers, radii, -1),
        test_path / "domain.xdmf",
        boundaries=boundaries,
        save_filename=test_path / "initial.png",
    )

    sd = np.load(test_path / "subdomain.npy")
    sd = np.vstack((sd, sd[0]))

    plot_basic(test_path, [(sd, "red")])


def test_07():

    print("\n\tElasticity inverse problem (Task Parallelism)\n")
    test_path = Path("../results/t07/")
    plot_basic(test_path)


def test_08():

    print("\n\tElasticity inverse problem (Mix Parallelism)\n")
    test_path = Path("../results/t08/")
    plot_basic(test_path)


def test_09():

    print("\n\tHeat conduction 1 (Data Parallelism)\n")
    test_path = Path("../results/t09/")

    vertices = [[0.0, 0.0], [0.4, 0.0], [0.6, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    dir_idx, dir_mkr = [2], 1
    vertices = vertices + [vertices[0]]
    dir = np.array(vertices[(dir_idx[0] - 1) : (dir_idx[-1] + 1)])
    boundaries = [(dir, "red")]

    # plot_basic(test_path, boundaries)


def test_10():

    print("\n\tHeat conduction 2 (Data Parallelism)\n")
    test_path = Path("../results/t10/")

    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    vertices = vertices + [vertices[0]]
    dir = np.array(vertices)
    boundaries = [(dir, "red")]

    # plot_basic(test_path, boundaries)

    # Plots for manuscript
    plot_results_for_doc(
        test_path, 0, [[0, 1], [0, 1]], 8, Path("../tex/heat2_0.png"), boundaries
    )
    plot_results_for_doc(
        test_path,
        172,
        [[0, 1], [0, 1]],
        8,
        Path("../tex/heat2_1.png"),
        boundaries,
    )


def test_11():

    print("\n\tHeat conduction 3 (Data Parallelism)\n")
    test_path = Path("../results/t11/")

    # plot_basic(test_path)


def test_12():

    print("\n\tHeat conduction with two sinks (Data Parallelism)\n")
    test_path = Path("../results/t12/")

    vertices = [
        [0.0, 0.0],
        [0.4, 0.0],
        [0.6, 0.0],
        [1.0, 0.0],
        [1.0, 0.4],
        [1.0, 0.6],
        [1.0, 1.0],
        [0.0, 1.0],
    ]

    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [5], 2

    dir1 = np.array(vertices[(dir1_idx[0] - 1) : (dir1_idx[-1] + 1)])
    dir2 = np.array(vertices[(dir2_idx[0] - 1) : (dir2_idx[-1] + 1)])

    boundaries = [(dir1, "red"), (dir2, "red")]

    # plot_basic(test_path, boundaries)


def test_13():

    print("\n\tHeat conduction with one load (Data Parallelism)\n")
    test_path = Path("../results/t13/")

    vertices = [
        [0.0, 0.0],
        [0.4, 0.0],
        [0.6, 0.0],
        [1.0, 0.0],
        [1.0, 0.4],
        [1.0, 0.6],
        [1.0, 1.0],
        [0.0, 1.0],
    ]
    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [5], 2
    dir1 = np.array(vertices[(dir1_idx[0] - 1) : (dir1_idx[-1] + 1)])
    dir2 = np.array(vertices[(dir2_idx[0] - 1) : (dir2_idx[-1] + 1)])
    boundaries = [(dir1, "red"), (dir2, "red")]

    # plot_basic(test_path, boundaries)

    # Plots for manuscript
    last_iter = 199
    plot_results_for_doc(
        test_path,
        last_iter,
        [[0, 1], [0, 1]],
        6,
        Path("../tex/heat4_3.png"),
        boundaries,
    )


def test_14():

    print("\n\tLogistic equation, r = 10 (Data Parallelism)\n")
    test_path = Path("../results/t14/")

    plot_basic(test_path)


def test_15():

    print("\n\tLogistic equation, r = 40 (Data Parallelism)\n")
    test_path = Path("../results/t15/")

    plot_basic(test_path)


def test_16():

    print("\n\tLogistic equation, r = 90 (Data Parallelism)\n")
    test_path = Path("../results/t16/")

    plot_basic(test_path)


def test_17():

    print("\n\t? (Data Parallelism)\n")
    test_path = Path("../results/t17/")

    plot_basic(test_path)


def test_18():

    print("\n\tCantilever with two loads II (Task Parallelism)\n")
    test_path = Path("../results/t18/")

    vertices = np.array(
        [
            [0.0, 0.0],
            [0.95, 0.0],
            [1.05, 0.0],
            [2.0, 0.0],
            [2.0, 0.45],
            [2.0, 0.55],
            [2.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    dir_idx, dir_mkr = [8], 1
    neu_idx_bot, neu_mkr_bot = [2], 2
    neu_idx_right, neu_mkr_right = [5], 3

    dir = np.array(vertices[(dir_idx[0] - 1) : (dir_idx[-1] + 1)])
    neu_bot = np.array(vertices[(neu_idx_bot[0] - 1) : (neu_idx_bot[-1] + 1)])
    neu_right = np.array(vertices[(neu_idx_right[0] - 1) : (neu_idx_right[-1] + 1)])
    boundaries = [(dir, "red"), (neu_bot, "deepskyblue"), (neu_right, "deepskyblue")]

    # plot_basic(test_path)

    plot_results_for_doc(
        test_path, 0, [[0, 2], [0, 1]], 8, Path("../tex/compli4_0.png"), boundaries
    )

    last_iter = 66
    plot_results_for_doc(
        test_path,
        last_iter,
        [[0, 2], [0, 1]],
        8,
        Path("../tex/compli4_1.png"),
        boundaries,
    )


def test_19():
    pass


def test_20():

    print("\n\tSymmetric Cantilever 2D (non-rectangular domain) - Data Parallelism\n")
    test_path = Path("../results/t20/")

    vertices = np.array(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.45),
            (2.0, 0.55),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx, neu_mkr = [3], 2

    dir = np.array(vertices[(dir_idx[0] - 1) : (dir_idx[-1] + 1)])
    neu = np.array(vertices[(neu_idx[0] - 1) : (neu_idx[-1] + 1)])
    boundaries = [(dir, "red"), (neu, "deepskyblue")]

    # plot_basic(test_path)

    plot_results_for_doc(
        test_path, 0, [[0, 2], [0, 1]], 8, Path("../tex/compli5_0.png"), boundaries
    )

    last_iter = 62
    plot_results_for_doc(
        test_path,
        last_iter,
        [[0, 2], [0, 1]],
        8,
        Path("../tex/compli5_1.png"),
        boundaries,
    )


def test_21():

    print("\n\t Heat conduction with four sources (single) - Data Parallelism\n")
    test_path = Path("../results/t21/")

    vertices = np.array(
        [
            [0.05, 0.0],
            [0.95, 0.0],
            [1.0, 0.05],
            [1.0, 0.95],
            [0.95, 1.0],
            [0.05, 1.0],
            [0.0, 0.95],
            [0.0, 0.05],
            [0.05, 0.0],
        ]
    )

    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [4], 2
    dir3_idx, dir3_mkr = [6], 3
    dir4_idx, dir4_mkr = [8], 4

    dir1 = np.array(vertices[(dir1_idx[0] - 1) : (dir1_idx[-1] + 1)])
    dir2 = np.array(vertices[(dir2_idx[0] - 1) : (dir2_idx[-1] + 1)])
    dir3 = np.array(vertices[(dir3_idx[0] - 1) : (dir3_idx[-1] + 1)])
    dir4 = np.array(vertices[(dir4_idx[0] - 1) : (dir4_idx[-1] + 1)])

    boundaries = [(dir1, "red"), (dir2, "red"), (dir3, "red"), (dir4, "red")]

    plot_results_for_doc(
        test_path, 0, [[0, 1], [0, 1]], 3, Path("../tex/heat5_0.png"), boundaries
    )
    plot_results_for_doc(
        test_path, 80, [[0, 1], [0, 1]], 3, Path("../tex/heat5_2.png"), boundaries
    )


def test_22():

    print("\n\t Heat conduction with four sources (multiple) - Task Parallelism\n")
    test_path = Path("../results/t22/")

    vertices = np.array(
        [
            [0.05, 0.0],
            [0.95, 0.0],
            [1.0, 0.05],
            [1.0, 0.95],
            [0.95, 1.0],
            [0.05, 1.0],
            [0.0, 0.95],
            [0.0, 0.05],
            [0.05, 0.0],
        ]
    )

    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [4], 2
    dir3_idx, dir3_mkr = [6], 3
    dir4_idx, dir4_mkr = [8], 4

    dir1 = np.array(vertices[(dir1_idx[0] - 1) : (dir1_idx[-1] + 1)])
    dir2 = np.array(vertices[(dir2_idx[0] - 1) : (dir2_idx[-1] + 1)])
    dir3 = np.array(vertices[(dir3_idx[0] - 1) : (dir3_idx[-1] + 1)])
    dir4 = np.array(vertices[(dir4_idx[0] - 1) : (dir4_idx[-1] + 1)])

    boundaries = [(dir1, "red"), (dir2, "red"), (dir3, "red"), (dir4, "red")]

    plot_results_for_doc(
        test_path, 63, [[0, 1], [0, 1]], 3, Path("../tex/heat5_3.png"), boundaries
    )


def test_32():
    """
    Performance of test 01 (mesh_size = 0.015, 20 946 triangles)
    """
    filename = Path("../tex/performance_1a.png")
    num_procs = np.arange(1, 11)
    resolution_times = [
        27.33817017199999,
        13.547148835999906,
        9.593779027999972,
        7.400951124999665,
        5.773146577999796,
        5.587280580999959,
        9.354751103000012,
        6.425889066999957,
        6.792038920999971,
        7.094126409999944,
    ]
    plot_bars(num_procs, resolution_times, filename)

    """
    Performance of test 01 (mesh_size = 0.0106, 42 035 triangles)
    """
    filename = Path("../tex/performance_1b.png")
    num_procs = np.arange(1, 11)
    resolution_times = [
        40.41690818799998,
        20.677432958999816,
        14.434316678999949,
        11.582552148000104,
        10.01717478400019,
        8.236337578999837,
        14.209770140999808,
        11.625784706000104,
        11.156894338999791,
        10.104777174999981,
    ]
    plot_bars(num_procs, resolution_times, filename)

    """
    Performance of test 03
    """
    filename = Path("../tex/performance_2.png")
    num_procs = np.arange(1, 11)
    resolution_times = [
        35.07935065799984,
        20.099389327000154,
        14.756963701000132,
        11.853981776999717,
        10.23874729499994,
        6.934067974999834,
        12.757291769999938,
        12.813011035000272,
        12.131966675000058,
        11.061559490000036,
    ]
    plot_bars(num_procs, resolution_times, filename)

    """
    Performance of test 21
    """
    filename = Path("../tex/performance_3.png")
    num_procs = np.arange(1, 11)
    resolution_times = [
        26.744602871000097,
        14.725885109999808,
        10.170385355000008,
        8.05341802199996,
        6.686465177999935,
        5.905231507000053,
        10.18865386199991,
        9.384790290999717,
        8.650962873000026,
        7.873313986000085,
    ]
    plot_bars(num_procs, resolution_times, filename)


test_functions = {
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
    "14": test_14,
    "15": test_15,
    "16": test_16,
    "17": test_17,
    "18": test_18,
    "19": test_19,
    "20": test_20,
    "21": test_21,
    "22": test_22,
    "32": test_32,
}


def main():

    if len(sys.argv) != 2:
        print("Usage: python load.py <test_id>")
        print("Example: python load.py 01")
        return

    test_id = sys.argv[1]
    func = test_functions.get(test_id)

    if func:
        func()
    else:
        print(f"Test '{test_id}' not recognized.")


if __name__ == "__main__":
    main()
