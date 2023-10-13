# pyGraSPH

A Python, and super basic, version of GraSPH. Early stages of development. Only in 2D.

## Setup

Package requirements:

```bash
pip install -r requirements.txt
```

NumPy version 1.25 is used here as performance improvements are introduced which
improve speed of this code by 4x. See [NumPy Release Notes](https://numpy.org/doc/stable/release/1.25.0-notes.html#numpy-1-25-0-release-notes),
where the improvements to `np.ufunc.at` and `np.einsum` are most relevant.

## Run

Edit `main.py` as needed, then

```bash
mkdir output # make output directory if doesn't exist
python3 main.py
```

and wait awhile.

## Plotting

Edit `plot.py` as needed, then

```bash
python3 plot.py <timestep>
```

which reads data form `output/sph_*.h5` and creates `sphplot.png` in the current working directory
