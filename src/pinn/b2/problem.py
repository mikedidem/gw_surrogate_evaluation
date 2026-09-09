#!/usr/bin/env python
"""Benchmark B2: heterogeneous conductivity and a regional head gradient.

Transient Boussinesq flow in an unconfined aquifer on a square domain, with
Dirichlet conditions north and south, no-flow east and west, and a single
pumping well represented as a Gaussian source.

B2 differs from B1 in two respects:

  1. K is a field, not a scalar. A low-conductivity lens with a 10:1 contrast
     sits between the well and the upgradient (north) boundary. K_of() returns
     K together with its analytic gradient, because the Boussinesq operator
     gains a term h*(grad K . grad h) once K varies in space.

  2. The Dirichlet boundaries carry different heads, 100 m north and 90 m
     south. bc() therefore returns the analytic Dupuit profile rather than a
     constant. h^2 is linear in y for steady unconfined flow with no recharge,
     so this profile solves the homogeneous steady problem exactly and
     satisfies both boundary conditions identically.

h_initial() must match the initial head used to build the MODFLOW model.
MODFLOW takes it as strt and the network takes it as hstar. If the two differ
they are solving different problems.

Coordinates are centred, x, y in [-500, 500]. MODFLOW exports are shifted by
-500 on load, so the two frames agree.
"""
import numpy as np
import torch


class Problem(object):
    def __init__(self,
                 sigma=30.0,
                 domain=(-500, 500, -500, 500, 0.0, 30.0),
                 xi=(1.6667, -1.6667)):
        self.sigma = sigma
        self.domain = domain
        self.xi = xi
        self.mu = 0.10

        # ---- regional gradient (heads at y = +500 and y = -500) ----
        self.h_north = 100.0
        self.h_south = 90.0

        # ---- conductivity field: low-K lens in a uniform background ----
        self.K_hi = 33.33          # background, the published baseline value
        self.K_lo = 3.333          # lens, 10:1 contrast
        self.lens_x = (-300.0, 300.0)
        self.lens_y = (240.0, 390.0)
        self.lens_w = 10.0         # tanh edge scale, ~40 m transition

        # kept so anything reading problem.K as a scalar gets the background
        self.K = self.K_hi

    def __repr__(self):
        return (f'unconfined well problem, lens K={self.K_lo}/{self.K_hi} m/d, '
                f'regional gradient {self.h_south}->{self.h_north} m')

    # ------------------------------------------------------------------
    # source term
    # ------------------------------------------------------------------
    def f(self, x):
        return - 4.e4 * np.exp(-((x[:, [0]] - self.xi[0])**2 +
                                 (x[:, [1]] - self.xi[1])**2)
                               / (2*self.sigma**2)) / (2*np.pi*self.sigma**2)

    # ------------------------------------------------------------------
    # conductivity field and its analytic gradient
    # ------------------------------------------------------------------
    def K_of(self, xy):
        """K, dK/dx, dK/dy at the given points.

        Accepts a torch tensor (n, >=2) or a numpy array and returns the same
        type. The field is a product of tanh ramps, so it is smooth on every
        side and at every corner and grad K is bounded everywhere.
        """
        is_torch = torch.is_tensor(xy)
        lib = torch if is_torch else np
        x, y = xy[:, [0]], xy[:, [1]]
        w = self.lens_w
        x0, x1 = self.lens_x
        y0, y1 = self.lens_y

        tx0, tx1 = lib.tanh((x - x0) / w), lib.tanh((x - x1) / w)
        ty0, ty1 = lib.tanh((y - y0) / w), lib.tanh((y - y1) / w)

        sx = 0.5 * (tx0 - tx1)
        sy = 0.5 * (ty0 - ty1)
        # d/dz tanh(z) = 1 - tanh^2(z)
        dsx = 0.5 * ((1.0 - tx0**2) - (1.0 - tx1**2)) / w
        dsy = 0.5 * ((1.0 - ty0**2) - (1.0 - ty1**2)) / w

        dK = self.K_lo - self.K_hi
        K = self.K_hi + dK * sx * sy
        Kx = dK * dsx * sy
        Ky = dK * sx * dsy
        return K, Kx, Ky

    # ------------------------------------------------------------------
    # initial / boundary conditions
    # ------------------------------------------------------------------
    def h_initial(self, y):
        """Dupuit profile: h^2 linear in y, 90 m at y=-500, 100 m at y=+500."""
        ymin, ymax = self.domain[2], self.domain[3]
        lib = torch if torch.is_tensor(y) else np
        frac = (y - ymin) / (ymax - ymin)
        return lib.sqrt(self.h_south**2
                        + (self.h_north**2 - self.h_south**2) * frac)

    def bc(self, x, mode=0):
        """boundary/initial condition"""

        if mode == 0:                      # initial condition
            return self.h_initial(x[:, [1]])

        elif mode == 1:                    # Dirichlet, north and south
            return self.h_initial(x[:, [1]])

        elif mode == 2:                    # Neumann, east and west (no-flow)
            return 0.0 * x[:, [0]]


if __name__ == '__main__':
    p = Problem()
    print(p)

    # boundary values
    yb = np.array([[0.0, -500.0], [0.0, 500.0], [0.0, 0.0]])
    print('h at y=-500, +500, 0 :', p.bc(yb, mode=1).ravel())

    # K field spot checks
    pts = np.array([[0., 315.],      # lens centre
                    [0., 0.],        # at the well
                    [0., -500.],     # south boundary
                    [0., 500.],      # north boundary
                    [-500., 315.],   # west boundary, lens height
                    [500., 315.]])   # east boundary, lens height
    K, Kx, Ky = p.K_of(pts)
    for q, k, kx, ky in zip(pts, K.ravel(), Kx.ravel(), Ky.ravel()):
        print(f'  ({q[0]:7.1f},{q[1]:7.1f})  K={k:8.4f}  dK/dx={kx:9.5f}  dK/dy={ky:9.5f}')

    # torch and numpy must agree
    Kt, _, _ = p.K_of(torch.from_numpy(pts).float())
    print('torch/numpy max diff:', float(np.abs(Kt.numpy() - K).max()))
