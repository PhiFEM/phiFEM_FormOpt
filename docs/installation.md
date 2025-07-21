# Installation

Below are some instructions to help you install **FEniCSx** and other required packages.

## 1. Install Anaconda

First, download and install [Anaconda](https://www.anaconda.com/) for Windows, Linux, or macOS.

Anaconda helps manage Python environments and packages easily.

## 2. Create a Conda Environment

A **Conda environment** provides an isolated workspace where you can install only the packages you need.

To create an environment named `fenicsx-env`, run:

```bash
conda create -n fenicsx-env
```

To activate the environment:

```bash
conda activate fenicsx-env
```

To deactivate it:

```bash
conda deactivate
```

Once the environment is active, any package you install or program you run will be isolated from the rest of your system.

## 3. Install Basic Packages

With the environment activated, install some essential packages such as **Python** (≥ 3.10), **NumPy**, **SciPy**, and **Matplotlib**.

You can find the appropriate installation commands at [conda-forge](https://anaconda.org/conda-forge), or simply run:

```bash
conda install -c conda-forge python numpy scipy matplotlib
```

## 4. Install FEniCSx

To install **FEniCSx** and its core dependencies, run:

```bash
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

This is the installation method recommended on the [FEniCSx download page](https://fenicsproject.org/download/).

## 5. Install Additional Packages

Finally, install other useful packages your module may depend on, such as:

- `pygmsh` (for mesh generation)
- `h5py` (for working with HDF5 files)

Install them with:

```bash
conda install -c conda-forge pygmsh h5py
```


