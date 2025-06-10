import sys
import numpy as np

from plots import (
    plot_basic,
    see_initial_guess
)

from pathlib import Path

def test_00():

    test_path = Path("../results/t00/")
    
    plot_basic(test_path)

def test_01():

    print("\n\tSymmetric cantilever 2D (Data Parallelism)\n")
    test_path = Path("../results/t01/")
    
    plot_basic(test_path)
    
def test_02():
    print("\n\tSymmetric cantilever 3D\n")

def test_03():

    print("\n\tMultiple load cases (Data Parallelism)\n")
    test_path = Path("../results/t03/")

    plot_basic(test_path)

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
        t_ = np.arcsin((b - eps)/b)
        t = np.linspace(-t_, np.pi + t_, npts)
        x = a*np.cos(t)
        y = b*np.sin(t) + (b - eps)
        return x, y
    
    npts = 80
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

    vertices = np.concatenate((vertices, [vertices[0]]))
    dir = np.array(vertices[(dir_idx[0]-1):(dir_idx[-1]+1)])
    bR = np.array(vertices[(bR_idx[0]-1):(bR_idx[-1]+1)])
    neuA = np.array(vertices[(neu_idxA[0]-1):(neu_idxA[-1]+1)])
    neuB = np.array(vertices[(neu_idxB[0]-1):(neu_idxB[-1]+1)])
    neuC = np.array(vertices[(neu_idxC[0]-1):(neu_idxC[-1]+1)])
    bL = np.array(vertices[(bL_idx[0]-1):(bL_idx[-1]+1)])

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
        boundaries = boundaries,
        save_filename = test_path / "initial.png"
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

    vertices = [
        [0.0, 0.0],
        [0.4, 0.0],
        [0.6, 0.0],
        [1.0, 0.0],
        [1., 1.0],
        [0.0, 1.0]
    ]

    dir_idx, dir_mkr = [2], 1
    vertices = vertices + [vertices[0]]
    dir = np.array(vertices[(dir_idx[0]-1):(dir_idx[-1]+1)])
    boundaries = [(dir, "red")]
    
    plot_basic(test_path, boundaries)

def test_10():
    
    print("\n\tHeat conduction 2 (Data Parallelism)\n")
    test_path = Path("../results/t10/")

    vertices = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ]

    vertices = vertices + [vertices[0]]
    dir = np.array(vertices)
    boundaries = [(dir, "red")]
    
    plot_basic(test_path, boundaries)

def test_11():
    
    print("\n\tHeat conduction 3 (Data Parallelism)\n")
    test_path = Path("../results/t11/")
    
    vertices = [
        [0.0, 0.0],
        [0.4, 0.0],
        [0.6, 0.0],
        [1.0, 0.0],
        [1.0, 0.4],
        [1.0, 0.6],
        [1.0, 1.0],
        [0.0, 1.0]
    ]

    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [5], 2

    dir1 = np.array(vertices[(dir1_idx[0]-1):(dir1_idx[-1]+1)])
    dir2 = np.array(vertices[(dir2_idx[0]-1):(dir2_idx[-1]+1)])

    boundaries = [(dir1, "red"), (dir2, "red")]

    plot_basic(test_path, boundaries)

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
        [0.0, 1.0]
    ]

    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [5], 2

    dir1 = np.array(vertices[(dir1_idx[0]-1):(dir1_idx[-1]+1)])
    dir2 = np.array(vertices[(dir2_idx[0]-1):(dir2_idx[-1]+1)])

    boundaries = [(dir1, "red"), (dir2, "red")]

    plot_basic(test_path, boundaries)

def test_13():

    print("\n\tHeat conduction with one load (Data Parallelism)\n")
    test_path = Path("../results/t13/")

    plot_basic(test_path)

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
    pass

def test_19():
    pass

def test_20():
    pass


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
    "14": test_14,
    "15": test_15,
    "16": test_16,
    "17": test_17,
    "18": test_18,
    "19": test_19,
    "20": test_20
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
