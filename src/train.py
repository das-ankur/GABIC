# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import argparse
import math
import random
import shutil
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from comp.datasets import ImageFolder

#from compressai.datasets import ImageFolder
from comp.zoo import models

from utils.dataset import VimeoDatasets, TestKodakDataset, SquarePad
from utils.parser import parse_args, choose_model_args
from utils.utils import *

import wandb
import os

from training.loss import RateDistortionLoss
from training.step import train_one_epoch, test_one_epoch, compress_one_epoch
import numpy as np
import torch_geometric
from utils.lr import CustomStepLr


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
        #weight_decay=0.01
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
        #weight_decay=0.01,
    )
    return optimizer, aux_optimizer




def save_checkpoint(state, is_best, out_dir, filename='last_checkpoint.pth.tar'):
    
    torch.save(state, f'{out_dir}/{filename}')
    if is_best:
        shutil.copyfile(f'{out_dir}/{filename}', f'{out_dir}/model_best.pth.tar')





def main():
    args = parse_args()
    args = choose_model_args(args)
    
    print(args)
    

    wandb.init(
        project='GABIC',
        name=f'{args.project_name}_seed-{args.seed}',
        config=get_wandb_config(args)
    )


    if args.seed is not None:
        print(f'seed: {args.seed}\n')
        random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)

        args.save_dir = f'{args.save_dir}_seed_{args.seed}'

        
    if(args.save):
        os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f'n gpus: {torch.cuda.device_count()}')


    train_transforms = transforms.Compose(
        [SquarePad(args.patch_size), transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [SquarePad(args.patch_size), transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    val_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    # Kodak test set
    kodak_dataset = TestKodakDataset(data_dir= args.test_pt)

    

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    kodak_dataloader = DataLoader(
        kodak_dataset, 
        shuffle=False, 
        batch_size=1, 
        pin_memory=(device == "cuda"), 
        num_workers= args.num_workers 
    )


    print('train val and test sets loaded.')

    if(args.model in ['wacnn_cw', 'stf']):
        print(f'base / wa: {args.model}')
        net = models[args.model]()
    elif(args.model == 'wgrcnn_cw'):
        
        net = models[args.model](
            knn = args.knn,
            graph_conv = args.graph_conv,
            heads = args.local_graph_heads, 
            use_edge_attr = args.use_edge_attr,
            dissimilarity = args.dissimilarity 
        )
    else:
        raise RuntimeError(f'The used model: "{args.model}" is not a channel-wise and so not supported by train.py script')
    
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
    
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    counter = args.counter
    best_val_loss = float("inf")
    best_kodak_loss = float("inf")

    if args.checkpoint != '_':  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        best_kodak_loss = checkpoint["best_kodak_loss"]

        if counter == 0:
            counter = checkpoint["counter"] + 1
        net.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    
    for epoch in range(last_epoch, args.epochs):
        start = time.time()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        counter = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            lr_scheduler,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            counter
        )
        val_loss = test_one_epoch(epoch, val_dataloader, net, criterion, counter, label='val')
        lr_scheduler.step(val_loss)
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        if(args.save):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_kodak_loss":best_kodak_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "counter": counter
                },
                is_best, out_dir = args.save_dir
            )

        kodak_loss = test_one_epoch(epoch, kodak_dataloader, net, criterion, counter, label='kodak')
        is_best_kodak = kodak_loss < best_kodak_loss
        best_kodak_loss = min(kodak_loss, best_kodak_loss)
        if(args.save and is_best_kodak):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_kodak_loss":best_kodak_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "counter": counter
                },
                False, out_dir = args.save_dir, filename=f'best_kodak_checkpoint.pth.tar'
            )

        if epoch%10==0:
            print("make actual compression")
            net.update(force = True)
            print("model updated")
            _ = compress_one_epoch(net, kodak_dataloader, device, counter, label='kodak')

        if epoch%5==0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_kodak_loss":best_kodak_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "counter": counter
                },
                is_best= False, out_dir = args.save_dir, filename=f'ep_{epoch}_checkpoint.pth.tar'
            )


        print(f'--- time {time.time()-start}\n')

if __name__ == "__main__":

    wandb.login()


    main()
