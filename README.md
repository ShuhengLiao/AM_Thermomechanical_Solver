# AM Thermomechanical Solver
A GPU-accelerated FEM solver for residual stress simulation in additive manufacturing using CuPy

**Authors:**
Shuheng Liao, Ashkan Golgoon

**Example results:**
<p align="middle">
  <img src="docs/files/L_contour.gif" width="600" />
</p>
<p align="middle">
  <img src="docs/files/L_zigzag.gif" width="600" />
</p>

# Installation Instructions

1) Install Conda according to instructions in [the Conda documentation](https://docs.conda.io/en/latest/miniconda.html).

2) Create a Conda environment initialized with Python 3.10:
   ```
   conda create -n gamma python=3.10
   ```

3) Activate the conda environment.
   ```
   conda activate gamma
   ```

4) Install required Conda packages:
   ```
   conda install python=3.10 numba pandas
   ```

5) Install required Pip packages:
   ```
   pip install cupy pyvista pyvirtualdisplay
   ```