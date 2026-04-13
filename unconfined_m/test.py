#!/usr/bin/env python
import torch
from options import Options
from trainer import Trainer
from problem import Problem

args = Options().parse()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

args.problem = Problem(sigma=args.sigma)
print('************************* TEST *********************************')
print(f'Confined Aquifer Containing A Single Well (Stage {args.stage})')
print(f'domain={args.problem.domain}')
print(f'tau={args.tau}')
print(f'sigma={args.sigma}')
print(f'constraint={args.constraint}')
print(f'layers={args.layers}')
print('****************************************************************\n')

trainer = Trainer(args)
trainer.test()
