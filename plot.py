import h5py, numpy as np, matplotlib.pyplot as plt, sys

with h5py.File(f"output/sph_{sys.argv[1]}.h5", "r") as f:

    x = np.array(f.get("x"))
    type = np.array(f.get("type"))
    rho = np.array(f.get("rho"))
    sigma = np.array(f.get("sigma"))
    strain = np.array(f.get("strain"))

    fig = plt.scatter(x[:, 0], x[:, 1], s=0.5, c=strain[:,0]+strain[:,1]+strain[:,2])

    axs = plt.gca()
    axs.set_aspect('equal')

    plt.colorbar()

    plt.savefig('sphplot.png')