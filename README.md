# pyGraSPH

A Python, and super basic, version of GraSPH. Early stages of development. Only in 2D.

## Setup

Package requirements:

```bash
pip/conda/mamba install scipy h5py
```

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