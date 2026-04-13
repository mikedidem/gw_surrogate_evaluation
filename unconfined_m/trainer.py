#!/usr/bin/env python
from options import Options
from utils import save_checkpoints, show_surface, show_contours_2x2, mae, mse, rrmse
from model import Net, Net_Neumann, Net_PDE, PINN
from dataset import Trainset, Validset
from sampler import Sampler
from problem import Problem
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from dataset import ModflowDataset  
from torch.utils.data import DataLoader, Subset
import time
import os
import argparse
import shutil


class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.cuda_index = args.cuda_index
        self.problem = Problem(sigma=args.sigma)
        self.constraint = args.constraint
        self.stage = args.stage
        self.tau = args.tau

        # Model name
        name = f"{args.constraint}_{args.hidden_layers}x{args.hidden_neurons}_tau:{self.tau:.0f}_sigma:{args.sigma:.0f}_S:{args.spatial_strategy}"

        self.model_name = f"Stage{self.stage}_{name}_T:{args.temporal_strategy}_nt:{args.nt}"
        if self.stage > 1:
            self.model_name_prev = f"Stage{self.stage-1}_{name}_T:{args.temporal_strategy_prev}_nt:{args.nt_prev}"

        # Networks
        self.net = Net(self.args, stage=self.stage)
        self.net_neumann = Net_Neumann(self.net)
        self.net_pde = Net_PDE(self.net)
        self.pinn = PINN(self.net)

        if self.stage > 1:
            self.net_prev = Net(self.args, stage=self.stage-1)
            self.net_neumann_prev = Net_Neumann(self.net_prev)
            self.net_pde_prev = Net_PDE(self.net_prev)
            self.pinn_prev = PINN(self.net_prev)

            if self.device == torch.device(type='cuda', index=self.cuda_index):
                self.net_prev.to(self.device)
                self.net_neumann_prev.to(self.device)
                self.net_pde_prev.to(self.device)
                self.pinn_prev.to(self.device)

            self.net_prev.eval()
            self.net_neumann_prev.eval()
            self.net_pde_prev.eval()
            self.pinn_prev.eval()

            # Loading best model
            best_model = torch.load(
                f'checkpoints/{self.model_name_prev}/best_model.pth.tar')
            self.pinn_prev.load_state_dict(best_model['state_dict'])

        # Criterion
        self.criterion = nn.MSELoss()

        # Resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                print(f'Resuming training, loading {args.resume} ...')
                self.pinn.load_state_dict(
                    torch.load(args.resume)['state_dict'])
            else:
                print('input resume error', args.resume)

        # Trainset
        self.trainset = Trainset(self.problem, stage=self.stage, tau=self.tau,
                                 spatial_strategy=args.spatial_strategy,
                                 temporal_strategy=args.temporal_strategy,
                                 n=args.n, nx=args.nx, ny=args.ny, nt=args.nt,
                                 ratio=args.ratio, filename='./data/well.mat')

        # Validset
        self.validsize = (100, 100)
        if self.stage == 1:
            time_stamps = [0.25, 0.5, 0.75, 1]
        elif self.stage == 2:
            time_stamps = [5, 10, 15, 20]
        self.validset = Validset(
            self.problem, self.validsize[0], self.validsize[1], time_stamps)

        if self.stage > 1:
            xy, _, xy_bdy2 = self.trainset.spatial()

            xytau = self.trainset.spatial_temporal(xy, self.tau)
            xytau_bdy2 = self.trainset.spatial_temporal(xy_bdy2, self.tau)

            xytau = torch.from_numpy(xytau).float()
            xytau_bdy2 = torch.from_numpy(xytau_bdy2).float()

            xyt = self.validset()
            xytau_valid = Validset(
                self.problem, self.validsize[0], self.validsize[1], self.tau)()

            if self.device == torch.device(type='cuda', index=self.cuda_index):
                xytau = xytau.to(self.device)
                xytau_bdy2 = xytau_bdy2.to(self.device)
                xytau_valid = xytau_valid.to(self.device)

            ##########################################################################
            # Generate information of hstar, including hstar, hstar_diff, hstar_x_bdy2,
            # which is used for training the second stage
            ##########################################################################
            hstar = self.net_prev(xytau).detach()
            hstar_diff = self.net_pde_prev(xytau, out_diff=True).detach()
            hstar_x_bdy2 = self.net_neumann_prev(xytau_bdy2).detach()

            self.hstar = hstar.repeat(args.nt-1, 1)
            self.hstar_diff = hstar_diff.repeat(args.nt-1, 1)
            self.hstar_x_bdy2 = hstar_x_bdy2.repeat(args.nt-1, 1)

            ##########################################################################
            # Read information of hstar_valid, including hstar_valid, hstar_valid_diff,
            # which is used for validating the second stage
            ##########################################################################
            hstar_valid = self.net_prev(xytau_valid).detach()
            hstar_valid_diff = self.net_pde_prev(
                xytau_valid, out_diff=True).detach()

            self.hstar_valid = hstar_valid.repeat(4, 1)
            self.hstar_valid_diff = hstar_valid_diff.repeat(4, 1)

    def train_info(self, optimizer, epoch, train_loss, valid_loss, tt):
        result = f'{optimizer:5s} '
        result += f'{epoch+1:5d}/{self.epochs_Adam+self.epochs_LBFGS:5d} '
        result += f'train_loss: {train_loss:.4e} '
        result += f'valid_loss: {valid_loss:.4e} '
        result += f'time: {time.time()-tt:5.2f} '
        if optimizer == 'Adam':
            result += f'lr: {self.lr_scheduler.get_last_lr()[0]:.2e}'
        #   ReduceLROnPlateau doesn't have get_last_lr(), read from optimizer
        print(result)
      

    def _next_data_batch(self):
        if self.data_loader is None:
            return None, None
        try:
            xyt_b, h_b = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.data_loader)
            xyt_b, h_b = next(self._data_iter)
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            xyt_b = xyt_b.to(self.device)
            h_b   = h_b.to(self.device)
                    
        # ---- Align supervised data with PINN coordinate system ----
        # clone to avoid in-place edits on dataloader buffers
        xyt_b = xyt_b.clone()
        xyt_b[:, 0] -= 500.0   # shift x
        xyt_b[:, 1] -= 500.0   # shift y
        # -----------------------------------------------------------
        xyt_b.requires_grad = False  # Explicit
        h_b.requires_grad = False    # Explicit
        return xyt_b, h_b

    @torch.no_grad()
    def _hstar_for_batch(self, xyt_batch):
        """
        For HARD Stage-2: compute h*(x,y) at t = tau for the batch’s (x,y).
        This preserves the spatially varying h*.
        """
        if self.stage != 2:
            return None
        # Build (x,y,t=tau) with same x,y as batch
        x = xyt_batch[:, [0]]
        y = xyt_batch[:, [1]]
        t = torch.full_like(x, fill_value=self.tau)
        xytau = torch.cat([x, y, t], dim=1)
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            xytau = xytau.to(self.device)
        hstar_b = self.net_prev(xytau).detach()
        # hstar_diff_b = self.net_pde_prev(xytau, out_diff=True).detach()
        return hstar_b  #, hstar_diff_b


    def train(self):

        # Hyperparameters Setting
        self.epochs_Adam = self.args.epochs_Adam
        self.epochs_LBFGS = self.args.epochs_LBFGS
        self.lam = self.args.lam

        # Writer
        self.writer = SummaryWriter(comment=f'_{self.model_name}')

        # Optimizer
        self.lr = self.args.lr
        self.optimizer_Adam = optim.Adam(
            [param for param in self.pinn.parameters() if param.requires_grad == True],
            lr=self.lr)

        self.lr_scheduler = StepLR(self.optimizer_Adam,
                                   step_size=2000,
                                   gamma=0.1)

        self.optimizer_LBFGS = optim.LBFGS(
            [param for param in self.pinn.parameters() if param.requires_grad == True],
            max_iter=20, 
            lr=0.1,# Line search iterations per step
            history_size=50,       # Memory for quasi-Newton approximation
            line_search_fn="strong_wolfe"  # Robust line search
        )

        print(f'{self.trainset}')

        ###########################################################
        # Generate trainset, including xyt, and (xyt_bdy2, hx_bdy2).
        # if SOFT, also (xy0, u0) and (xyt_bdy1, h_bdy1).
        ###########################################################
        xyt = self.trainset()
        xyt_bdy2, hx_bdy2 = self.trainset(mode=2)

        if self.constraint == 'SOFT':
            xy0, u0 = self.trainset(mode=0)
            xyt_bdy1, h_bdy1 = self.trainset(mode=1)

        best_loss = 1.0e10
        tt = time.time()

        self.pinn.train()
        step = 0


        # --------------------------
        # Supervised MODFLOW data
        # --------------------------
        self.w_data = getattr(self.args, "w_data", 1)
        data_bs     = getattr(self.args, "data_bs", 1024)
        self.w_data0 = self.w_data  # Store initial weight for annealing
        data_pat    = getattr(self.args, "data_pattern", "./modflow/sdata/t*.txt")
        val_frac    = getattr(self.args, "data_val_frac", 0.2)

        # Create a DataLoader that yields mini-batches (x,y,t) -> h
        try:
            full_ds = ModflowDataset(pattern=data_pat, stage = self.stage, tau = self.tau)
            N = len(full_ds)
            seed = 1234
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            idx = torch.randperm(N)  # random spatial split
            n_val = int(val_frac * N)
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]
            
            self.mod_train_ds = Subset(full_ds, train_idx)
            self.mod_val_ds   = Subset(full_ds, val_idx)
            
            self.data_loader = DataLoader(self.mod_train_ds, batch_size=data_bs,
                                          shuffle=True, drop_last=False)
            self._data_iter  = iter(self.data_loader)
            
            val_loader = DataLoader(self.mod_val_ds, batch_size=len(self.mod_val_ds),
                            shuffle=False)
            val_xyt, val_h = next(iter(val_loader))
            # Shift x,y -> PINN coords [-500,500] (same as in _next_data_batch)
            val_xyt = val_xyt.clone()
            val_xyt[:, 0] -= 500.0
            val_xyt[:, 1] -= 500.0
            if self.device == torch.device(type='cuda', index=self.cuda_index):
                val_xyt = val_xyt.to(self.device)
                val_h   = val_h.to(self.device)

            self.val_xyt_data = val_xyt
            self.val_h_data   = val_h
        except Exception as e:
            print(f"[Supervised] Could not load MODFLOW data: {e}")
            self.mod_train_ds = None
            self.mod_val_ds   = None
            self.data_loader  = None
            self._data_iter   = None
            self.val_xyt_data = None
            self.val_h_data   = None


        ########################
        # Transfer them to GPU
        ########################
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            xyt = xyt.to(self.device)
            xyt_bdy2, hx_bdy2 = xyt_bdy2.to(
                self.device), hx_bdy2.to(self.device)

            if self.constraint == 'HARD':
                if self.stage == 2:
                    self.hstar = self.hstar.to(self.device)
                    self.hstar_diff = self.hstar_diff.to(self.device)
                    self.hstar_x_bdy2 = self.hstar_x_bdy2.to(self.device)

            elif self.constraint == 'SOFT':
                xy0, u0 = xy0.to(self.device), u0.to(self.device)
                xyt_bdy1, h_bdy1 = xyt_bdy1.to(
                    self.device), h_bdy1.to(self.device)

            self.net.to(self.device)
            self.net_pde.to(self.device)
            self.pinn.to(self.device)

        self.xyt = xyt
        self.xyt_bdy2, self.hx_bdy2 = xyt_bdy2, hx_bdy2
        if self.constraint == 'SOFT':
            self.xy0, self.u0 = xy0, u0
            self.xyt_bdy1, self.h_bdy1 = xyt_bdy1, h_bdy1

        # Training
        # Stage1: Training Process using Adam Optimizer
        
        # Adaptive gradient balancing (no curriculum — alpha_ema from epoch 0)
        self.alpha_ema = 0.5  # Start balanced, EMA will converge
        self.ema_momentum = 0.9
        
        for epoch in range(self.epochs_Adam):
            train_loss = self.train_Adam(epoch)

            if (epoch + 1) % 100 == 0:
                step += 1
                valid_loss = self.validate(step)
                self.train_info('Adam', epoch, train_loss, valid_loss, tt)
                tt = time.time()

                self.pinn.train()

                is_best = valid_loss < best_loss
                if is_best:
                    best_loss = valid_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.pinn.state_dict(),
                    'best_loss': best_loss
                }
                save_checkpoints(state, is_best, save_dir=self.model_name)

        train_loss_old = train_loss
        
        # Fix #3: LBFGS REFINEMENT after Adam stabilization
        # Now that curriculum + adaptive weighting + residual supervision are correct,
        # LBFGS will sharply reduce PDE residuals and tighten physics-data compromise
        print("\n=== Starting LBFGS Refinement ===")
        print(f"Final Adam PDE loss: {train_loss:.4e}")
        print(f"Final Adam alpha_ema: {self.alpha_ema:.4e}")
        print("======================================\n")
        
        # Stage2: Training Process using LBFGS Optimizer
        for epoch in range(self.epochs_Adam, self.epochs_Adam + self.epochs_LBFGS):
            train_loss = self.train_LBFGS(epoch)
            
            
            
        #     # Stage2: Training Process using LBFGS Optimizer
        # for epoch in range(self.epochs_Adam, self.epochs_Adam + self.epochs_LBFGS):
        #     train_loss = self.train_LBFGS(epoch)

            # =========================================================
            # NEW: EARLY STOPPING (Physics Convergence Check)
            # =========================================================
            # Calculate the L2 norm of all gradients in the network
            total_norm = 0.0
            for p in self.pinn.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            # If the gradient is extremely small, we found the bottom.
            if total_norm < 1e-7:
                print(f"\n[LBFGS] Converged (Grad Norm {total_norm:.2e} < 1e-7). Stopping early.")
                break
            # =========================================================

            

            if (epoch+1) % 20 == 0:

                step += 1
                valid_loss = self.validate(step)
                self.train_info('LBFGS', epoch, train_loss, valid_loss, tt)
                tt = time.time()
                
                
                # --- EARLY STOPPING 2: DIVERGENCE CHECK ---
                # If validation loss spikes > 20% worse than best, kill it.
                if valid_loss > best_loss * 1.2:
                    print(f"\n[LBFGS] Divergence detected! Valid Loss {valid_loss:.4f} > 1.2x Best ({best_loss:.4f}). Stopping.")
                    break
                
                self.pinn.train()

                is_best = valid_loss < best_loss
                if is_best:
                    best_loss = valid_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.pinn.state_dict(),
                    'best_loss': best_loss
                }
                save_checkpoints(state, is_best, save_dir=self.model_name)

            if abs(train_loss-train_loss_old) < 1.e-7:
                break
            train_loss_old = train_loss

        self.writer.close()

        print('Training finished successfully!!!\n')

    def train_Adam(self, epoch):
        """
        Training process using Adam optimizer
        """

        self.optimizer_Adam.zero_grad()
    
        eps = 1e-8  # for numerical stability
        
        # Fix #2: Minibatch PDE collocation points to avoid over-smoothing
        # Subsample ~10k points per step to balance with data batch size (~1k)
        N_pde_batch = min(10000, len(self.xyt))
        idx_pde = torch.randperm(len(self.xyt), device=self.xyt.device)[:N_pde_batch]
        xyt_batch = self.xyt[idx_pde]
        
        # Subsample boundary points to balance with PDE interior
        N_bdy = min(2000, len(self.xyt_bdy2))
        idx_bdy = torch.randperm(len(self.xyt_bdy2), device=self.xyt_bdy2.device)[:N_bdy]
        xyt_bdy2_batch = self.xyt_bdy2[idx_bdy]
        hx_bdy2_batch = self.hx_bdy2[idx_bdy]
        
        # Forward and backward propogate
        if self.constraint == 'HARD':
            if self.stage == 1:
                res, hx_bdy2_pred = self.pinn(xyt_batch,
                                              xyt_bdy2_batch)
              
            else:
                # For Stage-2, also subsample hstar and hstar_diff
                hstar_batch = self.hstar[idx_pde]
                hstar_diff_batch = self.hstar_diff[idx_pde]
                hstar_x_bdy2_batch = self.hstar_x_bdy2[idx_bdy]  # Match boundary indices
                
                res, hx_bdy2_pred = self.pinn(xyt_batch,
                                              xyt_bdy2_batch,
                                              hstar=hstar_batch,
                                              hstar_diff=hstar_diff_batch,
                                              hstar_x_bdy2=hstar_x_bdy2_batch)
                
              

            loss_pde = self.criterion(res, torch.zeros_like(res))
            loss_bdy2 = self.criterion(hx_bdy2_pred, hx_bdy2_batch)

            loss_total = loss_pde + self.lam * loss_bdy2
            
            
            

        elif self.constraint == 'SOFT':
            res, u0_pred, h_bdy1_pred, hx_bdy2_pred = self.pinn(self.xyt,
                                                                self.xyt_bdy2,
                                                                self.xy0,
                                                                self.xyt_bdy1)
            loss_pde = self.criterion(res, torch.zeros_like(res))
            loss0 = self.criterion(u0_pred, self.u0)
            loss_bdy1 = self.criterion(h_bdy1_pred, self.h_bdy1)
            loss_bdy2 = self.criterion(hx_bdy2_pred, self.hx_bdy2)
            loss_total = loss_pde + self.lam * (loss0 + loss_bdy1 + loss_bdy2)
        
            


        # --------------------------
        # Supervised batch 
        # --------------------------
        loss_data = torch.tensor(0.0, device =self.device)
        xyt_b, h_b = self._next_data_batch()
        if xyt_b is not None:
            if self.constraint == 'HARD' and self.stage == 2:
                # Fix #4: Supervise RESIDUALS (h - hstar) instead of absolute head
                # This aligns data loss with PDE formulation which constrains incremental evolution
                with torch.no_grad():
                    hstar_b = self._hstar_for_batch(xyt_b)
                h_pred_b = self.net(xyt_b, hstar=hstar_b)
                
                # Supervise the residual: (h_pred - hstar) vs (h_true - hstar)
                loss_data = self.criterion(h_pred_b - hstar_b, h_b - hstar_b)
            else:
                # stage-1 HARD or any SOFT: plain forward (no hstar)
                h_pred_b = self.net(xyt_b)
                loss_data = self.criterion(h_pred_b, h_b)
            
            # Check for NaN/Inf
            if not torch.isfinite(loss_data):
                print(f"Warning: Non-finite data loss at epoch {epoch}")
                print(f"  h_pred range: [{h_pred_b.min():.4f}, {h_pred_b.max():.4f}]")
                print(f"  h_true range: [{h_b.min():.4f}, {h_b.max():.4f}]")
                loss_data = torch.tensor(0.0, device=self.device)
            
            # Fix #1: Gradient-based loss normalization (adaptive weighting)
            # Compute gradient norms to balance PDE and data terms dynamically
            
            
                
        
            # Compute gradient norms at output layer only (comparable & fast)
            def last_layer_params(model):
                """Robustly find the last trainable module's parameters."""
                last = None
                for m in model.modules():
                    if any(p.requires_grad for p in m.parameters(recurse=False)):
                        last = m
                return list(last.parameters()) if last is not None else list(model.parameters())

            def grad_norm(loss):
                try:
                    params = last_layer_params(self.net)
                    g = torch.autograd.grad(loss, params,
                                          retain_graph=True, create_graph=False)
                    return torch.sqrt(sum(gi.norm()**2 for gi in g)).item()
                except:
                    return 0.0
            
            grad_pde = grad_norm(loss_pde)
            grad_data = grad_norm(loss_data) if loss_data.item() > 0 else 0.0
            
            # Balanced adaptive weighting (freeze for first 50 epochs)
            if epoch > 50 and grad_data > 1e-8:
                alpha_new = grad_pde / (grad_pde + grad_data + 1e-8)
                self.alpha_ema = self.ema_momentum * self.alpha_ema + (1 - self.ema_momentum) * alpha_new
                self.alpha_ema = max(0.05, min(0.4, self.alpha_ema))
            loss_total = loss_total + self.alpha_ema * loss_data
            
            # Diagnostics for monitoring (every 500 epochs)
            if (epoch + 1) % 500 == 0:
                grad_bdy = grad_norm(loss_bdy2) if self.constraint == 'HARD' else 0.0
                print(f"\n=== Gradient Diagnostics (Epoch {epoch+1}, last-layer) ===")
                print(f"  PDE grad norm:     {grad_pde:.4e}")
                if self.constraint == 'HARD':
                    print(f"  Boundary grad:     {grad_bdy:.4e}  (last-layer only)")
                if grad_data > 0:
                    print(f"  Data grad norm:    {grad_data:.4e}")
                    print(f"  Adaptive alpha:    {self.alpha_ema:.4e}")
                print(f"=======================================\n")
        else:
            # No data supervision yet
            pass
        
        loss_total.backward()
        self.optimizer_Adam.step()
     


        #Log individual losses to tensorboard
        self.writer.add_scalar('train_loss_total', loss_total.item(), epoch)
        if self.constraint == 'HARD':
            self.writer.add_scalar('train_loss_pde', loss_pde.item(), epoch)
            self.writer.add_scalar('train_loss_bdy2', loss_bdy2.item(), epoch)
        elif self.constraint == 'SOFT':
            self.writer.add_scalar('train_loss_pde', loss_pde.item(), epoch)
            self.writer.add_scalar('train_loss_ic', loss0.item(), epoch)
            self.writer.add_scalar('train_loss_bdy1', loss_bdy1.item(), epoch)
            self.writer.add_scalar('train_loss_bdy2', loss_bdy2.item(), epoch)
        
        if xyt_b is not None:
            self.writer.add_scalar('train_loss_data', loss_data.item(), epoch)
            self.writer.add_scalar('alpha_ema', self.alpha_ema, epoch)
    
        # Print diagnostics every N epochs
        if (epoch + 1) % 100 == 0:  # Print every 100 epochs
            if self.constraint == 'HARD':
                print(f"\nEpoch {epoch+1} Loss Breakdown:")
                print(f"  PDE Loss:      {loss_pde.item():.4e}")
                print(f"  Boundary Loss: {loss_bdy2.item():.4e}")
            elif self.constraint == 'SOFT':
                print(f"\nEpoch {epoch+1} Loss Breakdown:")
                print(f"  PDE Loss:      {loss_pde.item():.4e}")
                print(f"  IC Loss:       {loss0.item():.4e}")
                print(f"  Boundary1:     {loss_bdy1.item():.4e}")
                print(f"  Boundary2:     {loss_bdy2.item():.4e}")
            
            if xyt_b is not None:
                print(f"  Data Loss:     {loss_data.item():.4e}")
                print(f"  Data Weight:   {self.alpha_ema:.4e} (adaptive)")
            print(f"  Total Loss:    {loss_total.item():.4e}")
    
            
    
    
        return loss_total.item()


    def train_LBFGS(self, epoch):
        """
        LBFGS optimizer for convergence after Adam stabilization
        """
        
        # ============================================================
        # 1. FREEZE WEIGHTS (Correctly)
        # ============================================================
        # Use the weight Adam learned, not 0.0
        if hasattr(self, 'alpha_ema'):
            alpha_lbfgs = self.alpha_ema
        else:
            alpha_lbfgs = 0.1
            
        # Clamp to [0.05, 0.4] for stability (same range as Adam, physics-dominant)
        alpha_lbfgs = max(0.05, min(0.4, alpha_lbfgs))
            
        # ============================================================
        # 2. FIXED BATCHES & SINGULARITY FILTER
        # ============================================================
        xyt_b_static, h_b_static = self._next_data_batch()
        
        # --- SINGULARITY FILTER: Remove points near well (r < 5m) ---
        r_min = 5.0 
        N_pde_target = min(10000, len(self.xyt))
        
        # Get random indices
        torch.manual_seed(epoch)
        idx_all = torch.randperm(len(self.xyt), device=self.xyt.device)
        
        # Filter loop: keep only points far from the well
        safe_indices = []
        for idx in idx_all:
            pt = self.xyt[idx]
            # Calculate radius (assuming centered at 0,0)
            r = torch.sqrt(pt[0]**2 + pt[1]**2)
            if r > r_min:
                safe_indices.append(idx)
            if len(safe_indices) >= N_pde_target:
                break
        
        # Create the safe batch
        idx_pde = torch.tensor(safe_indices, device=self.xyt.device)
        xyt_pde_batch = self.xyt[idx_pde]
        # -------------------------------------------------------------
        
        # Subsample boundary
        N_bdy = min(4000, len(self.xyt_bdy2))
        idx_bdy = torch.randperm(len(self.xyt_bdy2), device=self.xyt_bdy2.device)[:N_bdy]
        xyt_bdy2_batch = self.xyt_bdy2[idx_bdy]
        hx_bdy2_batch = self.hx_bdy2[idx_bdy]
        
        # Precompute Stage-2 priors
        hstar_batch = None
        hstar_diff_batch = None
        hstar_b_static = None
        
        if self.constraint == 'HARD' and self.stage == 2:
            hstar_batch = self.hstar[idx_pde]
            hstar_diff_batch = self.hstar_diff[idx_pde]
            hstar_x_bdy2_batch = self.hstar_x_bdy2[idx_bdy]
            if xyt_b_static is not None:
                with torch.no_grad():
                    hstar_b_static = self._hstar_for_batch(xyt_b_static)
        
        loss_components = {}
        last_loss = [0.0]
    
        def closure():
            if torch.is_grad_enabled():
                self.optimizer_LBFGS.zero_grad()
    
            try:
                # --- PDE LOSS ---
                if self.constraint == 'HARD':
                    if self.stage == 1:
                        res, hx_bdy2_pred = self.pinn(xyt_pde_batch, xyt_bdy2_batch)
                    else:  # Stage-2
                        res, hx_bdy2_pred = self.pinn(xyt_pde_batch,
                                                      xyt_bdy2_batch,
                                                      hstar=hstar_batch,
                                                      hstar_diff=hstar_diff_batch,
                                                      hstar_x_bdy2=hstar_x_bdy2_batch)
    
                    loss_pde = self.criterion(res, torch.zeros_like(res))
                    loss_bdy2 = self.criterion(hx_bdy2_pred, hx_bdy2_batch)
                    loss_total = loss_pde + self.lam * loss_bdy2
                    
                    loss_components['pde'] = loss_pde.item()
                    loss_components['bdy2'] = loss_bdy2.item()
    
                elif self.constraint == 'SOFT':
                    res, u0_pred, h_bdy1_pred, hx_bdy2_pred = self.pinn(xyt_pde_batch,
                                                                        xyt_bdy2_batch,
                                                                        self.xy0,
                                                                        self.xyt_bdy1)
                    loss_pde = self.criterion(res, torch.zeros_like(res))
                    loss_total = loss_pde + self.lam * (self.criterion(u0_pred, self.u0) + 
                                                        self.criterion(h_bdy1_pred, self.h_bdy1) + 
                                                        self.criterion(hx_bdy2_pred, hx_bdy2_batch))
                    loss_components['pde'] = loss_pde.item()
    
                # --- DATA LOSS (With frozen alpha) ---
                if xyt_b_static is not None and h_b_static is not None:
                    if self.constraint == 'HARD' and self.stage == 2:
                        h_pred_b = self.net(xyt_b_static, hstar=hstar_b_static)
                        loss_data = self.criterion(h_pred_b - hstar_b_static, 
                                                   h_b_static - hstar_b_static)
                    else:
                        h_pred_b = self.net(xyt_b_static)
                        loss_data = self.criterion(h_pred_b, h_b_static)
                    
                    if torch.isfinite(loss_data):
                        loss_total = loss_total + alpha_lbfgs * loss_data
                        loss_components['data'] = loss_data.item()
                    else:
                        loss_components['data'] = float('inf')
                
                # --- SOFT FAILURE CHECK ---
                if not torch.isfinite(loss_total):
                    print(f"    [LBFGS] Non-finite loss detected. Backtracking...")
                    return torch.tensor(1e9, device=self.device, requires_grad=True)
    
                if loss_total.requires_grad:
                    loss_total.backward()
    
                last_loss[0] = loss_total.item()
                return loss_total
    
            except Exception as e:
                print(f"    [LBFGS] Math error in closure: {e}. Backtracking...")
                return torch.tensor(1e9, device=self.device, requires_grad=True)
    
        # Step
        self.optimizer_LBFGS.step(closure)
        train_loss = last_loss[0]
    
        # Logging
        self.writer.add_scalar('train_loss', train_loss, epoch)
        for key, value in loss_components.items():
            self.writer.add_scalar(f'train_loss_{key}', value, epoch)
    
        # Print diagnostics
        if (epoch + 1) % 20 == 0:
            print(f"\nEpoch {epoch+1} LBFGS Loss Breakdown:")
            for key, value in loss_components.items():
                if key == 'data':
                    print(f"  {key.upper():12s}: {value:.4e} (frozen α={alpha_lbfgs:.4e})")
                else:
                    print(f"  {key.upper():12s}: {value:.4e}")
            print(f"  {'TOTAL':12s}: {train_loss:.4e}\n")
    
        return train_loss

    def validate(self, step):
        """Validate process"""

        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()

        xyt_valid = self.validset()
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            xyt_valid = xyt_valid.to(self.device)

            if self.stage == 2:
                self.hstar_valid = self.hstar_valid.to(self.device)
                self.hstar_valid_diff = self.hstar_valid_diff.to(self.device)

        if self.stage == 1:
            res = self.net_pde(xyt_valid)
        elif self.stage == 2:
            res = self.net_pde(xyt_valid, self.hstar_valid_diff)

        loss_pde = self.criterion(res, torch.zeros_like(res))
        valid_loss_pde = loss_pde.item()
        self.writer.add_scalar('valid_loss_pde', valid_loss_pde, step)

        # plot
        if self.stage == 1:
            h_valid = self.net(xyt_valid)
        elif self.stage == 2:
            h_valid = self.net(xyt_valid, self.hstar_valid)

        fig = show_surface(xyt_valid, h_valid, stage=self.stage)
        self.writer.add_figure(tag='3D surface', figure=fig, global_step=step)

        fig = show_contours_2x2(self.validset, h_valid, stage=self.stage,
                                well_type='unconfined_single_well')
        self.writer.add_figure(tag='contour', figure=fig, global_step=step)
        
        
        
        # ============================================================
        # 2) ---------------------- DATA VALIDATION --------------------
        # ============================================================
        # If you have no MODFLOW validation data, fallback to PDE only.
        if self.val_xyt_data is None or self.val_h_data is None:
            # PDE only fallback
            self.writer.add_scalar('valid_loss', valid_loss_pde, step)
            return valid_loss_pde
    
        with torch.no_grad():
            xyt_data = self.val_xyt_data
            h_true   = self.val_h_data
    
            if self.stage == 2:
                # compute h*(x,y) at t = tau for data validation points
                x = xyt_data[:, [0]]
                y = xyt_data[:, [1]]
                t_tau = torch.full_like(x, self.tau)
                xyt_tau = torch.cat([x, y, t_tau], dim=1)
    
                if self.device == torch.device(type='cuda', index=self.cuda_index):
                    xyt_tau = xyt_tau.to(self.device)
    
                hstar_val = self.net_prev(xyt_tau).detach()
                h_pred_data = self.net(xyt_data, hstar=hstar_val)
                
                # Fix #1: Validate RESIDUALS (h - hstar) to match training objective
                # This prevents penalizing the model for Stage-1 inherited offsets
                loss_data_residual = self.criterion(h_pred_data - hstar_val, h_true - hstar_val)
                valid_loss_data = loss_data_residual.item()
                
                # Also compute absolute error for reporting (not used for checkpointing)
                loss_data_absolute = self.criterion(h_pred_data, h_true)
                self.writer.add_scalar('valid_loss_data_absolute', loss_data_absolute.item(), step)
            else:
                # Stage-1: no hstar, use absolute head
                h_pred_data = self.net(xyt_data)
                loss_data = self.criterion(h_pred_data, h_true)
                valid_loss_data = loss_data.item()
    
        self.writer.add_scalar('valid_loss_data', valid_loss_data, step)
    
        # ============================================================
        # 3) ---------------- FINAL VALIDATION METRIC -----------------
        # ============================================================
        # Fix #3: CONSISTENT VALIDATION LOSS
        # Use combined loss that matches training objective for checkpoint selection
        # This prevents the silent selection bug where model optimizes (PDE+data)
        # but checkpoint is selected based on data-only
        # Use adaptive alpha for consistent checkpoint selection
        alpha_val = getattr(self, 'alpha_ema', 0.5)
        valid_loss = valid_loss_pde + alpha_val * valid_loss_data
        
        # Log both for comparison
        self.writer.add_scalar('valid_loss_combined', valid_loss, step)
        self.writer.add_scalar('valid_loss', valid_loss, step)
        

        return valid_loss

    def test(self):
        print(f'{self.validset}')

        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()
        self.pinn.eval()

        if self.device == torch.device(type='cuda', index=self.cuda_index):
            self.net.to(self.device)
            self.net_neumann.to(self.device)
            self.net_pde.to(self.device)
            self.pinn.to(self.device)

        best_model = torch.load(
            f'checkpoints/{self.model_name}/best_model.pth.tar')
        self.pinn.load_state_dict(best_model['state_dict'])

        xyt_valid = self.validset()
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            xyt_valid = xyt_valid.to(self.device)
            if self.stage > 1:
                self.hstar_valid = self.hstar_valid.to(self.device)
                self.hstar_valid_diff = self.hstar_valid_diff.to(self.device)

        if self.stage == 1:
            res_valid = self.net_pde(xyt_valid)
            h_valid = self.net(xyt_valid)

        elif self.stage > 1:
            res_valid = self.net_pde(xyt_valid, self.hstar_valid_diff)
            h_valid = self.net(xyt_valid, self.hstar_valid)

        # Compute test loss
        loss = self.criterion(res_valid, torch.zeros_like(res_valid)).item()
        print(f'Test loss: {loss:.4e}')

        # Plot 3D surface
        fig = show_surface(xyt_valid, h_valid, self.stage)

        # Plot contour map for comparing the results between MODFLOW and GWPINN
        fig = show_contours_2x2(self.validset, h_valid, self.stage,
                                well_type='unconfined_single_well')

        plt.show()

        if self.stage == 2:
            df = pd.read_csv('./modflow/o1.csv',
                             delim_whitespace=True, names=['x', 'y', 't', 'h'])
                             
            xy_shift = df[['x','y']].values - 500     # shift x,y only
            t        = df[['t']].values               # keep t as-is
        
            xyt   = np.hstack([xy_shift, t])                              # (x',y',t)
            xytau = np.hstack([xy_shift, self.tau * np.ones_like(t)])     # (x',y',tau)
            # xyt = df[['x', 'y', 't']].values
            # xy = df[['x', 'y']].values - 500
            # xytau = np.hstack((xy, self.tau*np.ones_like(xy[:, [0]])))

            xyt = torch.from_numpy(xyt).float()
            xytau = torch.from_numpy(xytau).float()

            if self.device == torch.device(type='cuda', index=self.cuda_index):
                xyt = xyt.to(self.device)
                xytau = xytau.to(self.device)

            hstar = self.net_prev(xytau).detach()
            h_pred = self.net(xyt, hstar)
            h_pred = h_pred.detach().cpu().numpy()

            h = df[['h']].values
            print(np.hstack((h, h_pred)))

            print(
                f'mae = {mae(h, h_pred):.3f}, rrmse = {rrmse(h, h_pred) * 100:.3f} %')
                
                
                
            # Test on last 5 time steps (26-30) - temporal extrapolation
            test_times = [26, 27, 28, 29, 30]
            all_mae = []
            all_rrmse = []
            all_r2 = []
            all_rmse = []
            
            # For overall R² across all test times
            all_predictions_combined = []
            all_targets_combined = []
            
            print("\n=== Testing on Last 5 Time Steps (Extrapolation) ===")
            
            for t_val in test_times:
                try:
                    df = pd.read_csv(f'./modflow/t{t_val}.txt',
                                    sep=r'\s*,\s*|\s+', engine='python',
                                    header=None, names=['x', 'y', 'h'])
                    df = df.apply(pd.to_numeric, errors='coerce').dropna()
                    
                    if len(df) == 0:
                        print(f"  t={t_val}: No valid data, skipping")
                        continue
                    
                    xy_shift = df[['x','y']].values - 500     # shift to [-500, 500]
                    t = np.full((len(df), 1), float(t_val))   # constant time
                    xyt = np.hstack([xy_shift, t])
                    xytau = np.hstack([xy_shift, self.tau * np.ones_like(t)])
                    
                    xyt = torch.from_numpy(xyt.astype(np.float32))
                    xytau = torch.from_numpy(xytau.astype(np.float32))
                    
                    if self.device == torch.device(type='cuda', index=self.cuda_index):
                        xyt = xyt.to(self.device)
                        xytau = xytau.to(self.device)
                    
                    hstar = self.net_prev(xytau).detach()
                    h_pred = self.net(xyt, hstar).detach().cpu().numpy()
                    h_true = df[['h']].values
                    
                    # Calculate metrics
                    t_mae = mae(h_true, h_pred)
                    t_rmse = np.sqrt(np.mean((h_true - h_pred) ** 2))
                    t_rrmse = rrmse(h_true, h_pred) * 100
                    
                    # Calculate R² for this time step
                    ss_res = np.sum((h_true - h_pred) ** 2)
                    ss_tot = np.sum((h_true - np.mean(h_true)) ** 2)
                    t_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    
                    all_mae.append(t_mae)
                    all_rmse.append(t_rmse)
                    all_rrmse.append(t_rrmse)
                    all_r2.append(t_r2)
                    
                    # Store for combined R²
                    all_predictions_combined.append(h_pred)
                    all_targets_combined.append(h_true)
                    
                    print(f"  t={t_val:2d}: MAE={t_mae:.3f}m,RMSE={t_rmse:.3f}m, RRMSE={t_rrmse:.3f}%, R²={t_r2:.6f}")
                    
                except Exception as e:
                    print(f"  t={t_val}: Error loading/processing - {e}")
            
            if all_mae:
                # Calculate overall R² across all test times
                all_predictions_combined = np.vstack(all_predictions_combined)
                all_targets_combined = np.vstack(all_targets_combined)
                
                ss_res_total = np.sum((all_targets_combined - all_predictions_combined) ** 2)
                ss_tot_total = np.sum((all_targets_combined - np.mean(all_targets_combined)) ** 2)
                r2_overall = 1 - (ss_res_total / ss_tot_total) if ss_tot_total > 0 else 0.0
                
                print(f"\n--- FINAL TEST RESULTS (t=26-30) ---")
                print(f"Average MAE:    {np.mean(all_mae):.3f} m")
                print(f"Average RMSE:   {np.mean(all_rmse):.3f} m")
                print(f"Average RRMSE:  {np.mean(all_rrmse):.3f} %")
                print(f"Average R²:     {np.mean(all_r2):.6f}")
                print(f"Overall R²:     {r2_overall:.6f} (combined)")
                print(f"Std MAE:        {np.std(all_mae):.3f} m")
                print(f"Std RMSE:       {np.std(all_rmse):.3f} m")
                print(f"Std RRMSE:      {np.std(all_rrmse):.3f} %")
                print(f"Std R²:         {np.std(all_r2):.6f}")
                print(f"------------------------------------")
            else:
                print("WARNING: No test data found for t=26-30!")



        print('Testing finished successfully!!!\n')


if __name__ == '__main__':

    args = Options().parse()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.problem = Problem(sigma=args.sigma)
    print('****************************************************************')
    print(f'Confined Aquifer Containing A Single Well (Stage {args.stage})')
    print(f'domain={args.problem.domain}, tau={args.tau}')
    print(f'constraint={args.constraint}, sigma={args.sigma}')
    print(f'layers=args.layers')
    print('****************************************************************')
    trainer = Trainer(args)

    trainer.train()
