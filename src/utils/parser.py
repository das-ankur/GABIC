import argparse
from comp.zoo import models

def parse_args():
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="stf",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument("--patience", default=4, type=int)
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=30,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--seed", type=int, help="Set random seed for reproducibility", default=42
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s)",
    )

    parser.add_argument("--save", action="store_true", help="Save checkpoint")
    parser.add_argument("--save-dir", type = str, help = "Save directory", default = "/scratch/GBIC_res/exp")
    parser.add_argument("--project-name", type = str, help = "project name of wandb directory", default = "delete")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default = "_")


    parser.add_argument("--test-pt", type = str, help = "test dataset", default = "/scratch/dataset/kodak")

    parser.add_argument("--counter", type = int, default = 0)

    parser.add_argument("-N","--N", type = int, default = 192)
    parser.add_argument("-M","--M", type = int, default = 320)
    
    args,_ = parser.parse_known_args()
    return args





def choose_model_args(args):
    dynamic_parser = argparse.ArgumentParser()
    
    if(args.model == 'wgrcnn_cw'):
        dynamic_parser = args_wgrcnn_cw(dynamic_parser)
    args2,_ =  dynamic_parser.parse_known_args()

    combined_args = argparse.Namespace(**vars(args), **vars(args2))
    return combined_args    





def args_wgrcnn_cw(parser):
    print('parsing args for wgrcnn_cw model')

    parser.add_argument("--knn", type = int, default = 0)
    parser.add_argument("--graph-conv", type = str, default = 'transf_custom')  # Also for Glocal One Graph and Local graph pyg
    parser.add_argument("--local-graph-heads", type=int, default=8)
    parser.add_argument("--use-edge-attr", action="store_true", help="Use Linear Heads to merge heads results")
    parser.add_argument("--dissimilarity", action="store_true", help="Knn based on dissimilarity")
    return parser

