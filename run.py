import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import argparse
from trainer import Trainier


parser = argparse.ArgumentParser()

# Train
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--max_epochs", default=100, type=int, help="max number of training epochs")
parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--device", default='cuda', type=str)

# Optimizer
parser.add_argument("--optim_lr", default=5e-4, type=float, help="optimization learning rate")
parser.add_argument("--betas", default=(0.9, 0.999))
parser.add_argument("--decay", default=0.0005, type=float, help="momentum")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--T_0", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--eta_min", default=1e-7, type=float)

# Loss
parser.add_argument("--alpha_lmse", default=1., type=float)
parser.add_argument("--alpha_lcon", default=0., type=float, help="max number of training epochs")
parser.add_argument("--alpha_lconsis", default=0., type=float, help="validation frequency")
parser.add_argument("--temperature", default=0.1, type=float, help="validation frequency")

# Data
parser.add_argument("--data_json", default='./mri_gen.json', type=str, help="dataset json file")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--batch_size", default=32, type=int, help="number of batch size")
parser.add_argument("--datasets", default="Imp2DBraTs2023", type=str, help="Seg3DBraTs2023, Seg2DBraTs2023, or Gen2DBraTs2023")

# Model
parser.add_argument("--img_size", default=256, type=int, help="feature size")
parser.add_argument("--patch_size", default=2, type=int, help="number of input channels")
parser.add_argument("--embed_dim", default=24, type=int, help="number of output channels")
parser.add_argument("--depths", default=(2, 2, 2, 2), type=str, help="feature size")
parser.add_argument("--num_heads", default=(3, 6, 12, 24), type=str, help="number of input channels")
parser.add_argument("--spatial_dims", default=2, type=int, help="drop path rate")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--attn_drop_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--normalize", default=True, type=bool, help="normalization name")
parser.add_argument("--downsample", default="merging", type=str, help="drop path rate")
parser.add_argument("--con_dim", default=512, type=int, help="normalization name")
parser.add_argument("--mask_ratio_spa", default=0.5, type=float, help="dropout rate")

# DataAug
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")

# Checkpoint
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint, including model, "
                                                       "optimizer, scheduler")
parser.add_argument("--logdir", default="./train_log", type=str, help="directory to save the tensorboard logs")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp

    trainer = Trainier(args)
    train_acc = trainer.train()
    
    return train_acc


if __name__ == "__main__":
    main()