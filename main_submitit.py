import os 
import wandb
import torch
import argparse
import numpy as np
from pathlib import Path
from trainer import Trainer
from omegaconf import OmegaConf
import torch.backends.cudnn as cudnn
from utils.dist import init_distributed_mode, get_rank, get_world_size

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_args_parser():
    parser = argparse.ArgumentParser('InstDiff training and evaluation script', add_help=False)
    parser.add_argument("--DATA_ROOT", type=str,  default="DATA", help="path to DATA")
    parser.add_argument("--OUTPUT_ROOT", type=str,  default="OUTPUT", help="path to OUTPUT")
    parser.add_argument("--name", type=str,  default="checkpoint-01", help="checkpoints and related files will be stored in OUTPUT_ROOT/name")
    parser.add_argument("--seed", type=int,  default=123, help="used in sampler")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--yaml_file", type=str,  default="configs/train_text_all_sd_v1_5.yaml", help="paths to base configs.")
    parser.add_argument("--base_learning_rate", type=float,  default=5e-5, help="")
    parser.add_argument("--weight_decay", type=float,  default=0.0, help="")
    parser.add_argument("--warmup_steps", type=int,  default=10000, help="")
    parser.add_argument("--scheduler_type", type=str,  default='constant', help="cosine or constant")
    parser.add_argument("--batch_size", type=int,  default=2, help="")
    parser.add_argument("--workers", type=int,  default=1, help="")
    parser.add_argument("--official_ckpt_name", type=str,  default="sd-v1-4.ckpt", help="SD ckpt name and it is expected in DATA_ROOT, thus DATA_ROOT/official_ckpt_name must exists")
    parser.add_argument("--ckpt", type=lambda x:x if type(x) == str and x.lower() != "none" else None,  default=None, 
        help=("If given, then it will start training from this ckpt"
              "It has higher prioty than official_ckpt_name, but lower than the ckpt found in autoresuming (see trainer.py) ")
    )
    
    # use exponential moving average or not
    parser.add_argument('--enable_ema', default=False, type=lambda x:x.lower() == "true")
    parser.add_argument("--ema_rate", type=float,  default=0.9999, help="")

    # checkpoint and logging
    parser.add_argument("--total_iters", type=int,  default=500000, help="")
    parser.add_argument("--save_every_iters", type=int,  default=10000, help="")
    parser.add_argument("--total_epochs", type=int,  default=40, help="")
    parser.add_argument("--disable_inference_in_training", type=lambda x:x.lower() == "true",  default=False, help="Do not do inference, thus it is faster to run first a few iters. It may be useful for debugging ")

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # wandb parameters
    parser.add_argument("--wandb_name", type=str,  default="instdiff", help="name for wandb run")

    # use fp32; use fp16 by default
    parser.add_argument('--fp32', type=lambda x:x.lower() == "true", default=False, help="use fp32")

    # text file for model learning
    parser.add_argument("--train_file", type=str,  default="train.txt", help="list of JSON files for model training")

    # count number of duplicated classes when making a sentence.
    parser.add_argument("--count_dup", type=lambda x:x.lower() == "true", default=False, help="count number of duplicated classes")

    # count number of duplicated classes when making a sentence.
    parser.add_argument("--re_init_opt", type=lambda x:x.lower() == "true", default=False, help="reinitialize optimizer and scheduler")

    # randomly use blip embeddings with a probability of random_blip
    parser.add_argument("--random_blip", type=float, default=0.0, help="randomly use blip embeddings")

    # use masked attention in the self-attention layer
    parser.add_argument("--use_masked_att", type=lambda x:x.lower() == "true", default=False, help="use masked attention given the bounding box or not")

    # more options
    parser.add_argument("--add_inst_cap_2_global", type=lambda x:x.lower() == "true", default=False, help="add instance captions to the global captions or not")
    parser.add_argument("--use_instance_sampler", type=lambda x:x.lower() == "true", default=False, help="using multi-instance sampler during training or not")
    parser.add_argument("--mis_ratio", type=float,  default=0, help="the percentage of timesteps using multi-instance-sampler")
    parser.add_argument("--use_crop_paste", type=lambda x:x.lower() == "true", default=False, help="using use_crop_paste for multi-instance sampler or not")
    parser.add_argument("--use_instance_loss", type=lambda x:x.lower() == "true", default=False, help="using instance loss")
    parser.add_argument("--instance_loss_weight", type=float, default=0.0, help="weights for instance loss")

    return parser


def main(args):
    init_distributed_mode(args)
    # print(args)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # set up the config
    n_gpu = get_world_size()
    config = OmegaConf.load(args.yaml_file) 

    # convert PosixPath to string as OmegaConf does not support PosixPath
    args.job_dir = str(args.output_dir)
    args.output_dir = str(args.output_dir)

    config.update( vars(args) )
    config.total_batch_size = config.batch_size * n_gpu
    # print("total_batch_size: ", config.total_batch_size)
    
    config.local_rank = args.gpu # assign local rank for each process
    # print("config: ", config)
    # print("args: ", args)

    # set up wandb
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    if get_rank() == 0:
        wandb.init(
            project="InstDiff",
            sync_tensorboard=True,
            name=args.wandb_name,
            entity="",
        )

    # start training
    trainer = Trainer(config)
    trainer.start_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('InstDiff training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
