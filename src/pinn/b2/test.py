#!/usr/bin/env python
import torch
import numpy as np
from options import Options
from trainer import Trainer
from problem import Problem

args = Options().parse()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# numpy must be seeded as well as torch: the LHS temporal sampling (pyDOE.lhs)
# draws from numpy's global RNG, so the set of collocation time levels depends
# on it. Runs are reproducible from --seed.
np.random.seed(args.seed)

args.problem = Problem(sigma=args.sigma)
print('************************* TEST *********************************')
print(f'Unconfined aquifer, single pumping well (stage {args.stage})')
print(f'domain={args.problem.domain}')
print(f'tau={args.tau}')
print(f'sigma={args.sigma}')
print(f'constraint={args.constraint}')
print(f'seed={args.seed}')
print(f'layers={args.layers}')
print('****************************************************************\n')

trainer = Trainer(args)
trainer.test()
