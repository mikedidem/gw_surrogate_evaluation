#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import shutil
import os


def tile(x, y):
    """Cartesian product of two coordinate arrays. Required by dataset.py."""
    X = np.tile(x, (y.shape[0], 1))
    Y = np.vstack([np.tile(y[i], (x.shape[0], 1)) for i in range(y.shape[0])])

    return np.hstack((X, Y))


# def show_surface(xyt, h, stage):
#     """
#     xyt: torch.tensor
#     h: torch.tensor
#     """
#     if stage == 1:
#         times = [0.25, 0.5, 0.75, 1]
#     elif stage == 2:
#         times = [5, 10, 15, 20]

#     xyt = xyt.cpu().numpy()
#     h = h.detach().cpu().numpy()

#     n = xyt.shape[0]
#     nx = 100
#     ny = 100

#     xyt0 = xyt[:n//4, :]
#     xyt1 = xyt[n//4:n//2, :]
#     xyt2 = xyt[n//2:3*n//4, :]
#     xyt3 = xyt[3*n//4:, :]

#     h0 = h[:n//4, :]
#     h1 = h[n//4:n//2, :]
#     h2 = h[n//2:3*n//4, :]
#     h3 = h[3*n//4:, :]

#     x = []
#     x.append(xyt0[:, 0].reshape(nx, ny))
#     x.append(xyt1[:, 0].reshape(nx, ny))
#     x.append(xyt2[:, 0].reshape(nx, ny))
#     x.append(xyt3[:, 0].reshape(nx, ny))

#     y = []
#     y.append(xyt0[:, 1].reshape(nx, ny))
#     y.append(xyt1[:, 1].reshape(nx, ny))
#     y.append(xyt2[:, 1].reshape(nx, ny))
#     y.append(xyt3[:, 1].reshape(nx, ny))

#     h = []
#     h.append(h0.reshape(nx, ny))
#     h.append(h1.reshape(nx, ny))
#     h.append(h2.reshape(nx, ny))
#     h.append(h3.reshape(nx, ny))

#     # Plot
#     fig = plt.figure(figsize=(10, 10))
#     mpl.rcParams['xtick.labelsize'] = 12
#     mpl.rcParams['font.size'] = 12
#     mpl.rcParams['axes.labelsize'] = 14
#     for i in range(0, 4):
#         ax = fig.add_subplot(2, 2, i+1, projection='3d')
#         surf = ax.plot_surface(x[i], y[i], h[i],
#                                cmap=cm.rainbow, linewidth=0, antialiased=False)
#         ax.set_title(f'$t={times[i]}$ d')
#         if i == 0:
#             ax.set_xlabel('x (m)')
#             ax.set_ylabel('y (m)')
#             ax.set_zlabel('h (m)')

#     # ax.set_zlim(2, 2.4)
#     # ax.set_zticks([2.0, 2.1, 2.2, 2.3, 2.4])

#     # Add a color bar which map values to colors
#     cax = fig.add_axes([0.92, 0.2, 0.01, 0.6])
#     fig.colorbar(surf, cax=cax)

#     return fig


def show_contours_2x2(validset, h, stage, well_type='confined_single_well'):

    if stage == 1:
        times = [0.25, 0.5, 0.75, 1]
    elif stage == 2:
        times = [5, 10, 15, 20]

    grid_x, grid_y = validset.spatial(grid=True)
    h = h.detach().cpu().numpy()

    n = h.shape[0]
    h0 = h[:n//4, :]
    h1 = h[n//4:n//2, :]
    h2 = h[n//2:3*n//4, :]
    h3 = h[3*n//4:, :]

    h = []
    h.append(h0.reshape(100, 100))
    h.append(h1.reshape(100, 100))
    h.append(h2.reshape(100, 100))
    h.append(h3.reshape(100, 100))

    fig = plt.figure(figsize=(10, 10))
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    if well_type == 'confined_single_well':
        levels = [65, 75, 80, 85, 90, 95]
    elif well_type == 'confined_multiple_wells':
        levels = [85, 88, 91, 94, 97]
    elif well_type == 'unconfined_single_well':
        levels = [85, 86, 87, 88, 89]
    else:
        levels = None

    # Reference snapshots. Default to this project's own modflow/ folder;
    # override with MODFLOW_REF_DIR if the data lives elsewhere.
    ref_dir = os.environ.get('MODFLOW_REF_DIR', './modflow')

    for i, t in enumerate(times):
        # files are named t0.25.txt but t1.txt (not t1.0.txt)
        ts = str(int(t)) if float(t).is_integer() else str(t)
        df = pd.read_csv(
            os.path.join(ref_dir, f't{ts}.txt'),
            sep=r'\s+',
            names=['x', 'y', 'h'])
        # MODFLOW writes 0..1000; the network works in -500..500
        points = df[['x', 'y']].values - 500.0
        values = df[['h']].values

        # Fall back to data-driven levels when the hardcoded set does not
        # COVER the field. Overlap is not enough: with a regional gradient the
        # old 85-89 levels overlap a field spanning 86-100 but would bunch
        # every contour at the bottom and hide the gradient entirely.
        span_data = float(values.max() - values.min())
        span_lv = (max(levels) - min(levels)) if levels else 0.0
        if span_lv < 0.5 * span_data:
            levels = list(np.linspace(float(values.min()),
                                      float(values.max()), 6)[1:-1])

        grid_h_modflow = griddata(
            points, values, (grid_x, grid_y), method='cubic')[:, :, 0]

        grid_h = h[i]

        ax = fig.add_subplot(2, 2, i+1)
        cs1 = ax.contour(grid_x, grid_y, grid_h_modflow,
                         colors='k',
                         levels=levels)
        cs2 = ax.contour(grid_x, grid_y, grid_h,
                         colors='r',
                         levels=levels)
        h1, _ = cs1.legend_elements()
        h2, _ = cs2.legend_elements()
        ax.legend([h1[0], h2[0]], ['MODFLOW', 'GW-PINN'])
        ax.clabel(cs1, fmt='%2d', inline=1, fontsize=10)
        ax.clabel(cs2, fmt='%2d', inline=1, fontsize=10)
        ax.set_title(f'$t={t}$ d')

        if i == 0:
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
        if i == 1:
            ax.text(350, 510, 'Unit (m)')

        ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    return fig


def save_checkpoints(state, is_best=None,
                     base_dir='checkpoints',
                     save_dir=None):
    if save_dir:
        save_dir = os.path.join(base_dir, save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    checkpoint = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint)
    if is_best:
        best_model = os.path.join(save_dir, 'best_model.pth.tar')
        shutil.copyfile(checkpoint, best_model)


def mae(h, h_pred):
    return np.mean(np.abs(h - h_pred))


def mse(h, h_pred):
    return np.mean(np.square(h - h_pred))


def rrmse(h, h_pred):
    return np.sqrt(mse(h, h_pred) / np.mean(h)**2)


if __name__ == '__main__':
    x = np.random.rand(5, 2)
    y = np.random.rand(3, 2)
    z = tile(x, y)
    print(z)
