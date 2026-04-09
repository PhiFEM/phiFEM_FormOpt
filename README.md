# phiFEM-FormOpt

An implementation of the $\varphi$-FEM schemes in [FormOpt](https://github.com/JD26/FormOpt).

## Installation

### Conda

1. Download and install [Anaconda](https://www.anaconda.com/) for Windows, Linux, or macOS.
2. Create an Anaconda environment from the `explicit-env.txt` file:
   ```bash
   conda create -n phifem-formopt -f explicit-env.txt
   ```
3. Activate the environment:
   ```bash
   conda activate phifem-formopt
   ```

### Pixi

1. Download and install [Pixi](https://pixi.prefix.dev/latest/installation/).
2. Create and activate Pixi's workspace:
   ```bash
   pixi shell
   ```

## Contact

If you have any questions, suggestions, or need assistance with this project, feel free to reach out:
- **Raphaël Bulle:** raphael.bulle@inria.fr
- **Josué Diaz Avalos:** josue.diazavalos@uni-due.de
- **Louis Ducongé:** louis.duconge3111@gmail.com
- **Michel Duprez:** michel.duprez@inria.fr
- **Antoine Laurain:** antoine.laurain@uni-due.de

## Repository structure

```
├── code/       		# Python scripts and numerical tests
│ 	├── formopt.py		# Classes and functions
│ 	├── models.py		# Model problems
│ 	├── test.py			# Several tests
│ 	├── load.py			# Load and show numerical results
│ 	├── plots.py		# Functions for plotting
│ 	├── Examples.ipynb	# Examples for manuscript
│	└── Tutorial.ipynb	# Examples with detailed explanations
├── tex/        		# LaTeX source files of the article
├── results/    		# Directory structure for numerical results
└── docs/       		# Markdown documentation files
```

## License

`phiFEM-FormOpt` is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with `phiFEM-FormOpt`. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).