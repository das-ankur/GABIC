import torch 
import os 
from torchvision import transforms
from PIL import Image, ImageChops
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.step import AverageMeter
from compressai.ops import compute_padding
import math 
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
import json
import argparse
from compressai.zoo import *
from utils.dataset import VimeoDatasets, TestKodakDataset
from torch.utils.data import DataLoader
from os.path import join 
from comp.datasets import ImageFolder
from comp.zoo import models
from torch.profiler import profile, record_function, ProfilerActivity
from training.loss import RateDistortionLoss
from torchvision.utils import save_image
import sys
import random
import numpy as np
import torch_geometric
from evaluate.bd_metrics import *
from comp.zoo.pretrained import load_pretrained
from compressai.zoo import cheng2020_attn, mbt2018_mean, bmshj2018_hyperprior
import time
from evaluate.colors_model import Colors, Colors_vs_base

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def set_seed(seed=66):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

def compute_metrics( org, rec, max_val: int = 255):
    metrics =  {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics

def load_models(models_path,model_checkpoint, device, model_type):
    res = {}
    for model_check in model_checkpoint:
    
        model_path = join(models_path, model_check)
        checkpoint = torch.load(model_path, map_location=device)
        name = (model_path.split(os.sep)[-1]).replace('.pth.tar','')

        N = 192
        M = 320

        if(model_type in ['wacnn_cw', 'stf']):
            print(f'base / wa: {model_type}')
            # model = models[model_type]()
            state_dict = load_pretrained(torch.load(model_path)['state_dict'])
            model = models[model_type].from_state_dict(state_dict)
            model = model.to(device)
            model.update(force = True)
            model.eval()
        elif(model_type == 'wgrcnn_cw'):
            print('Loading tranf conv')
            model = models[model_type](
                knn = 9,
                graph_conv = 'transf_custom', # '',
                heads = 8, 
                use_edge_attr = True,
                dissimilarity = False 
            )
        
        if(model_type not in ['wacnn_cw', 'stf']):
            model = model.to(device)
            model.load_state_dict(checkpoint["state_dict"]) 
            model.update(force = True)
            model.eval()
        
        lambd = 0.0
        if 'q1' in name:
            lambd = 0.0067
        elif 'q2' in name:
            lambd = 0.0130
        elif 'q3' in name:
            lambd = 0.0250
        elif 'q4' in name:
            lambd = 0.0483
        else:
            raise NotImplementedError(f'{name} does not contains qx')
        res[name] = {
            "model": model,
            "psnr": AverageMeter(),
            "ms_ssim": AverageMeter(),
            "bpps": AverageMeter(),
            "rate": AverageMeter(),
            "criterion": RateDistortionLoss(lmbda=lambd),
            "loss": AverageMeter()
            }
        print(f'{model_path} loaded')
    print()
    return res


def read_image(filepath):
    assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


@torch.no_grad()
def inference_entropy_estimation(model,x, x_padded, unpad, criterion):
    out  = model(x_padded)
    out_criterion = criterion(out, x_padded)
    out["x_hat"] = F.pad(out["x_hat"], unpad)
    metrics = compute_metrics(x, out["x_hat"], 255)
    size = out['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    
    y_bpp = torch.log(out["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
                        
    bpp = y_bpp
    rate = bpp.item()*num_pixels 
    
    return metrics, bpp, rate, out["x_hat"], out_criterion["loss"].item()

@torch.no_grad()
def inference(model,x, x_padded, unpad):
    out_enc = model.compress(x_padded)
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])

    out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
    metrics = compute_metrics(x, out_dec["x_hat"], 255)

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    rate = bpp*num_pixels 
    return metrics, torch.tensor([bpp]), rate, out_dec["x_hat"], 0.0


@torch.no_grad()
def eval_models(models, dataloader, device):
    
    print("Starting inferences")
    res_metrics = {}

    for j,(x,_) in enumerate(dataloader):
        res_metrics_img = {}
        print(j)
        x = x.to(device)
        h, w = x.size(2), x.size(3)
        pad, unpad = compute_padding(h, w, min_div=2**6) #min_div=2**6)  # pad to allow 6 strides of 2
        x_padded = F.pad(x, pad, mode="constant", value=0)

        if(j==0):
            print(f'unpad: {x.shape}')
            print(f'input: {x_padded.shape}')

        if(x.shape != x_padded.shape):
            print(f'{x.shape} --> {x_padded.shape}')

        for model_type in list(models.keys()):
            for qp in sorted(list(models[model_type].keys())):
                model = models[model_type][qp]['model']
                criterion = models[model_type][qp]['criterion']
                metrics, bpp, rate, x_hat, loss = inference(model,x,x_padded,unpad)
                x_hat = (255 * x_hat.permute(0, 2, 3, 1).detach().cpu().numpy()).astype(np.uint8)
                x_hat = x_hat[0]
                img = Image.fromarray(x_hat)
                img.save(os.path.join('/kaggle/working/GABIC/compressed_images', f'img_c{j}__m{qp}.png'))
                models[model_type][qp]['psnr'].update(metrics["psnr"])
                models[model_type][qp]['ms_ssim'].update(metrics["ms-ssim"])
                models[model_type][qp]['bpps'].update(bpp.item())
                models[model_type][qp]['rate'].update(rate)
                models[model_type][qp]['loss'].update(loss)

    for model_type in list(models.keys()):
        model_res = {}
        print(model_type)
        for qp in list(models[model_type].keys()):
            qp_name = str(qp).split('-')[0]
            model_res[qp_name] = {
                'psnr': models[model_type][qp]['psnr'].avg,
                'mssim': models[model_type][qp]['ms_ssim'].avg,
                'bpp': models[model_type][qp]['bpps'].avg,
                'rate': models[model_type][qp]['rate'].avg,
                'loss': models[model_type][qp]['loss'].avg
            }
            print(f'{qp}: {model_res[qp_name]}')
        res_metrics[model_type] = model_res
    return res_metrics


def extract_specific_model_performance(metrics, type):
    nms = list(metrics[type].keys())
    psnr = []
    mssim = []
    bpp = []
    rate = []
    for names in nms:
        psnr.append(metrics[type][names]["psnr"])
        mssim.append(metrics[type][names]["mssim"])
        bpp.append(metrics[type][names]["bpp"])
        rate.append(metrics[type][names]["rate"])
    
    return sorted(psnr), sorted(mssim), sorted(bpp), sorted(rate)


def plot_rate_distorsion_psnr(metrics, savepath, colors = Colors):
    print(f'plotting on {savepath}')
    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    for type_name in metrics.keys():
        psnr, mssim, bpp, rate = extract_specific_model_performance(metrics, type_name)      
        cols = colors[type_name]      
        axes.plot(bpp, psnr,cols[1],color = cols[0], label = type_name)
        axes.plot(bpp, psnr,'-',color = cols[0])
        axes.plot(bpp, psnr,cols[1],color =  cols[0])
    axes.set_ylabel('PSNR [dB]')
    axes.set_xlabel('Bit-rate [bpp]')
    axes.title.set_text(f'PSNR comparison')
    axes.grid()
    axes.legend(loc='best')
    # for ax in axes:
    axes.grid(True)
    plt.savefig(savepath)
    plt.close()      
     
def produce_metrics(configs, dataset_path, saved_checkpoint = True):

    # Loading dict of models
    models = {}
    if saved_checkpoint:
        for config in configs:

            with open(config) as f:
                args = json.load(f)
            model_name = args['model']
            models_path = config.replace('inference.json','')
            models_checkpoint = []
            for entry in os.listdir(models_path):
                if('pth.tar' in entry):
                    models_checkpoint.append(entry) # checkpoints models  q1-bmshj2018-sos.pth.tar, q2-....
            print(models_checkpoint)
            
            device = "cuda"
            res = load_models(models_path,models_checkpoint, device, args['model_type'])
            models[model_name] = res
    else:
        device = "cuda"
        for model_arch in configs:
            res = {}
            for qp in range(3,7):
                if model_arch == 'Cheng2020':
                    net = cheng2020_attn(quality=qp, pretrained=True).eval().to(device)
                elif model_arch == 'Minnen2018':
                    net = mbt2018_mean(quality=qp, pretrained=True).eval().to(device)
                elif model_arch == 'Ballé2018':
                    net = bmshj2018_hyperprior(quality=qp, pretrained=True).eval().to(device)
                
                res[f'q{qp}-model'] = {   
                    "model": net,
                    "psnr": AverageMeter(),
                    "ms_ssim": AverageMeter(),
                    "bpps": AverageMeter(),
                    "rate": AverageMeter(),
                    "criterion": None,
                    "loss": AverageMeter()
                }
            models[model_arch] = res

    # Test Set
    test_dataset = TestKodakDataset(data_dir= dataset_path, image_size=-1,  crop = False, random_crop=False, get_img_name = True)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=4)    
    metrics = eval_models(models, test_dataloader , device)
    return metrics


def produce_bd_metrics(metrics, baseline_name = 'local_attention', save_file = '/save/dir'):
    
    if(baseline_name not in metrics.keys()):
       print(f'{baseline_name} not found')
       sys.exit(1)
    psnr_base, _, _, rate_base = extract_specific_model_performance(metrics, 'local_attention')
    for type_name in metrics.keys():
        if(type_name == 'local_attention'):
            continue
        print(type_name)
        psnr, _, _, rate = extract_specific_model_performance(metrics, type_name)   
        print(f'DB-PSNR: {BD_PSNR(rate_base, psnr_base, rate, psnr)}')
        print(f'DB-RATE: {BD_RATE(rate_base, psnr_base, rate, psnr)}')
        with open(save_file, 'a') as f:
            f.write(f'## {type_name}\n')
            f.write(f'DB-PSNR: {BD_PSNR(rate_base, psnr_base, rate, psnr)} <br>\n')
            f.write(f'DB-RATE: {BD_RATE(rate_base, psnr_base, rate, psnr)} <br>\n\n')
        print('---------\n')


if __name__ == "__main__":
    set_seed()
    
    my_parser = argparse.ArgumentParser(description= "path to read the configuration of the evaluation")
    
    my_parser.add_argument("--metrics", default="none", type=str,
                      help='metrics json file')
    
    my_parser.add_argument("--dataset", default='kodak', type=str)
    my_parser.add_argument("--file-name", default='res', type=str)
    
    args = my_parser.parse_args()
    
    configs = [
        '/kaggle/input/gabic/pytorch/default/1/inference.json'
    ]
    # configs = [
    #     'Cheng2020',
    #     'Minnen2018',
    #     'Ballé2018'
    # ]
    new_metrics = {}
    if(args.metrics == 'none'):
        #print(config)
        new_metrics = produce_metrics(
            configs,
            args.dataset,
            saved_checkpoint=True)
        
        save_path = f'compression_results'
        
        os.makedirs(save_path, exist_ok=True)
        print(f'Results will be saved on {save_path}')
        file_path = join(save_path,f'{args.file_name}.json')
        with open(file_path, 'w') as outfile:
            json.dump(new_metrics, outfile)
        save_path_img = join(save_path,f'{args.file_name}.pdf')
        colors = Colors
    else:
        work_path = '/'.join(args.metrics.split('/')[:-1])
        with open(args.metrics) as json_file:
            new_metrics = json.load(json_file)
        save_path_img = join(work_path,f'{args.file_name}.pdf')
        save_path = work_path
        
        colors = Colors_vs_base
    
    plot_rate_distorsion_psnr(new_metrics,save_path_img, colors=colors)
    # produce_bd_metrics(new_metrics, save_file=join(save_path,f'bd_res.txt'))