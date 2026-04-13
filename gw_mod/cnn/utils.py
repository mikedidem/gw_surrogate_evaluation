#!/usr/bin/env python
import torch, re
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



def plot_test_cnn_vs_modflow(
    test_ds,
    cnn_pred_dir="outputs/test_predictions",
    shift_xy=500.0
):
    """
    Plot CNN vs MODFLOW contours using SAVED test predictions (no model calls).
    """



    # --- Select last 4 TEST timesteps ---
    test_files = test_ds.txt_files
    if len(test_files) < 4:
        raise ValueError("Need at least 4 test timesteps.")

    modflow_files = test_files[-4:]
    cnn_files = [
        os.path.join(cnn_pred_dir, f"cnn_pred_{os.path.basename(f)}")
        for f in modflow_files
    ]

    # --- Build grid from first MODFLOW file ---
    df0 = pd.read_csv(modflow_files[0], sep=r"\s+", engine="python", names=["x","y","h"])
    x = np.unique(df0["x"].values) 
    y = np.unique(df0["y"].values) 
    grid_x, grid_y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 10))
    mpl.rcParams["font.size"] = 12

    for i, (mf_file, cnn_file) in enumerate(zip(modflow_files, cnn_files)):

        # ----- MODFLOW ground truth -----
        df_mf = pd.read_csv(
            mf_file,
            sep=r"\s+",
            names=["x", "y", "h"]
            
)
        
        df_mf = df_mf.apply(pd.to_numeric, errors="coerce")
        pts_mf = df_mf[["x", "y"]].values - shift_xy
        h_mf = df_mf["h"].values

        grid_h_mf = griddata(
            pts_mf, h_mf, (grid_x, grid_y), method="cubic"
        )

        # ----- CNN prediction (already denormalized) -----
        df_cnn = pd.read_csv(
            cnn_file,
            sep=r"\s+",
            names=["x", "y", "h"],
            skiprows=1
        )
        df_cnn = df_cnn.apply(pd.to_numeric, errors="coerce")

        pts_cnn = df_cnn[["x", "y"]].values - shift_xy
        h_cnn = df_cnn["h"].values

        grid_h_cnn = griddata(
            pts_cnn, h_cnn, (grid_x, grid_y), method="linear"
        )

        # ----- Shared contour levels -----
        vmin = np.nanmin([grid_h_mf.min(), grid_h_cnn.min()])
        vmax = np.nanmax([grid_h_mf.max(), grid_h_cnn.max()])
        levels = np.linspace(vmin, vmax, 7)

        ax = fig.add_subplot(2, 2, i + 1)

        cs1 = ax.contour(
            grid_x, grid_y, grid_h_mf,
            colors="k", levels=levels
        )
        cs2 = ax.contour(
            grid_x, grid_y, grid_h_cnn,
            colors="r", levels=levels,
            linewidths=2.5, linestyles="--"
        )

        h1, _ = cs1.legend_elements()
        h2, _ = cs2.legend_elements()
        ax.legend([h1[0], h2[0]], ["MODFLOW", "CNN"])

        ax.clabel(cs1, fontsize=9)
        ax.clabel(cs2, fontsize=9)

        timestep = os.path.basename(mf_file).replace(".txt", "")
        ax.set_title(f"{timestep}")
        ax.set_aspect("equal", adjustable="box")

        if i == 0:
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

    plt.tight_layout()
    return fig



def mae(h, h_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(h - h_pred))


def mse(h, h_pred):
    """Mean Squared Error."""
    return np.mean(np.square(h - h_pred))


def rrmse(h, h_pred):
    """Relative Root Mean Squared Error."""
    return np.sqrt(mse(h, h_pred) / np.mean(h)**2)