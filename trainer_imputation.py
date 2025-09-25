import os
import time
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import random
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from data.brats2023 import get_loader
from models import ImputationModel

from losses import ImputationLoss
from metrics.syn_metrics import GenMetrics
from metrics.seg_metrics import SegMetrics


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)
        
        
class Trainier():
    def __init__(self, args) -> None:
        self.args = args
        
        if self.args.amp:
            self.scaler = GradScaler()
            
        self.trainer_initialized = False
        
        self.start_epoch = self.args.start_epoch

            
    def initial_trainer(self):
        self.trainer_initialized = False
        self.config_dataset()
        self.init_model()
        self.config_optimizer()
        self.config_losses_and_metrics()
        self.config_wirter()
        
        self.trainer_initialized = True
        
    def config_dataset(self):
        self.train_loader = get_loader(
            datalist_json=self.args.data_json,
            batch_size=self.args.batch_size,
            num_works=self.args.workers,
            phase='train',
            datasets=self.args.datasets
        )
    
        self.val_loader = get_loader(
            datalist_json=self.args.data_json,
            batch_size=self.args.batch_size,
            num_works=self.args.workers,
            phase='val',
            datasets=self.args.datasets
        )
        
        self.test_loader = get_loader(
            datalist_json=self.args.data_json,
            batch_size=self.args.batch_size,
            num_works=self.args.workers,
            phase='test',
            datasets=self.args.datasets
        )
        
        print('Train data number: {} | Val data number: {} | Test data number: {}'.format(
            len(self.train_loader) * self.args.batch_size,
            len(self.val_loader) * self.args.batch_size,
            len(self.test_loader) * self.args.batch_size))
        
    def init_model(self):
        self.model = ImputationModel(
            img_size=self.args.img_size,
            patch_size=self.args.patch_size,
            embed_dim=self.args.embed_dim,
            depths=self.args.depths,
            num_heads=self.args.num_heads,
            norm_name=self.args.norm_name,
            drop_rate=self.args.dropout_rate,
            attn_drop_rate=self.args.attn_drop_rate,
            dropout_path_rate=self.args.dropout_path_rate,
            normalize=self.args.normalize,
            downsample=self.args.downsample,
            con_dim=self.args.con_dim,
            mask_ratio_spa=self.args.mask_ratio_spa,
        ).to(self.args.device)
        
        pytorch_model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Model parameters count", pytorch_model_params)
        
    def config_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr=self.args.optim_lr, 
                                           betas=self.args.betas,
                                           weight_decay=self.args.decay)

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=self.args.T_0,
            T_mult=1,
            eta_min=self.args.eta_min)
        print("Config optimier")
        
    def config_losses_and_metrics(self):
        self.loss = ImputationLoss(
            alpha_lmse=self.args.alpha_lmse, 
            alpha_lcon=self.args.alpha_lcon, 
            alpha_lconsis=self.args.alpha_lconsis,
            temperature=self.args.temperature,  
            batch_size=self.args.batch_size,
            embed_dim=self.args.con_dim,
            device=self.args.device, 
        )
        
        self.metrics = GenMetrics(
            spatial_dims=2
        )
        print("Config losses and metrics")
        
    def config_wirter(self):
        self.writer = None
        if self.args.logdir is not None:
            self.writer = SummaryWriter(log_dir=f'{self.args.logdir}/tensorboard')
            print("Writing Tensorboard logs to ", f'{self.args.logdir}/tensorboard')
        print("Save model to ", self.args.logdir)

    def save_checkpoint(self, file_name, epoch, best_acc):
        state_dict = self.model.state_dict() 
        save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
        if self.optimizer is not None:
            save_dict["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            save_dict["scheduler"] = self.scheduler.state_dict()

        file_name = os.path.join(self.args.logdir, file_name)
        torch.save(save_dict, file_name)
        print("Saving checkpoint", file_name)
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        self.model.load_state_dict(new_state_dict, strict=False)
        print("=> loaded model checkpoint")

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        if "optimizer" in checkpoint.keys():
            for k, v in checkpoint["optimizer"].items():
                new_state_dict[k.replace("backbone.", "")] = v
            self.optimizer.load_state_dict(new_state_dict)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()      
            print("=> loaded optimizer checkpoint")
        if "scheduler" in checkpoint.keys():
            for k, v in checkpoint["scheduler"].items():
                new_state_dict[k.replace("backbone.", "")] = v
            self.scheduler.load_state_dict(new_state_dict)
            self.scheduler.step(epoch=start_epoch)
            print("=> loaded scheduler checkpoint")
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(self.args.checkpoint, start_epoch, best_acc))
    
    def get_mask_indice(self, epoch):
        self.mask = [
            # [1, 0, 0, 0],
            # [0, 1, 0, 0],
            # [0, 0, 1, 0],
            # [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
        ]
        
        self.tri_missing = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]

        self.double_missing = [
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 1]
        ]
        
        self.single_missing = [
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
        ]
        
        return random.choice(self.tri_missing)
        
        # if epoch < int(self.args.max_epochs // 5):
        #     return random.choice(self.single_missing)
        # elif int(self.args.max_epochs // 5) <= epoch < int(3 * self.args.max_epochs // 5):
        #     return random.choice(self.double_missing)
        # else:
        #     return random.choice(self.tri_missing)
        
    def train_one_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        run_mse_loss = AverageMeter()
        run_diff_loss = AverageMeter()
        run_con_loss = AverageMeter()
        
        for idx, batch_data in enumerate(self.train_loader):
            img = batch_data["image"].to(self.args.device)

            mask_indice = self.get_mask_indice(epoch)
                            
            with autocast(enabled=self.args.amp):
                x_rec, x_con, y_con = self.model(img, mask_indice)

                l_mse, l_diff, l_con = self.loss(x_rec, img, x_con, y_con)
                loss = l_mse + l_diff + l_con
                
            if self.args.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            run_mse_loss.update(l_mse.item(), n=self.args.batch_size)
            run_diff_loss.update(l_diff.item(), n=self.args.batch_size)
            run_con_loss.update(l_con.item(), n=self.args.batch_size)
                        
        return run_mse_loss.avg, run_diff_loss.avg, run_con_loss.avg
    
    def validata(self):
        self.model.eval()
        run_mse_loss = AverageMeter()
        run_diff_loss = AverageMeter()
        run_con_loss = AverageMeter()
        
        with torch.no_grad():
            for idx, batch_data in enumerate(self.val_loader):
                img = batch_data["image"].to(self.args.device)

                with autocast(enabled=self.args.amp):
                    x_rec, x_con, y_con = self.model(img)
                    l_mse, l_diff, l_con = self.loss(x_rec, img, x_con, y_con)

            run_mse_loss.update(l_mse.item(), n=self.args.batch_size)
            run_diff_loss.update(l_diff.item(), n=self.args.batch_size)
            run_con_loss.update(l_con.item(), n=self.args.batch_size)
            
        return run_mse_loss.avg, run_diff_loss.avg, run_con_loss.avg
    
    def train(self):
        if self.trainer_initialized is False:
            self.initial_trainer()
            
            val_loss_min = 10.0
    
        for epoch in range(self.args.start_epoch, self.args.max_epochs):

            print(time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            mse_loss, diff_loss, con_loss = self.train_one_epoch(epoch)

            print(
                "Train Epoch  {}/{}".format(epoch, self.args.max_epochs - 1),
                "MSE loss: {:.4f}".format(mse_loss),
                "Diff loss: {:.4f}".format(diff_loss),
                "Con loss: {:.4f}".format(con_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

            self.writer.add_scalar("train_mse_loss", mse_loss, epoch)
            self.writer.add_scalar("train_diff_loss", diff_loss, epoch)
            self.writer.add_scalar("train_con_loss", con_loss, epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch + 1) % self.args.val_every == 0:

                epoch_time = time.time()
                mse_loss, diff_loss, con_loss = self.validata()

                print(
                    "Final validation stats {}/{}".format(epoch, self.args.max_epochs - 1),
                    "MSE loss: {:.4f}".format(mse_loss),
                    "Diff loss: {:.4f}".format(diff_loss),
                    "Con loss: {:.4f}".format(con_loss),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )

                if self.writer is not None:
                    self.writer.add_scalar("val_mse_loss", np.mean(mse_loss), epoch)
                    self.writer.add_scalar("val_diff_loss", np.mean(diff_loss), epoch)
                    self.writer.add_scalar("val_con_loss", np.mean(con_loss), epoch)

                val_avg_acc = np.mean((mse_loss + diff_loss + con_loss) / 3)

                if val_avg_acc < val_loss_min:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_loss_min, val_avg_acc))
                    val_loss_min = val_avg_acc

                    self.save_checkpoint(file_name=f"ckpt_{self.args.embed_dim}_{self.args.patch_size}_{self.args.alpha_lmse}{self.args.alpha_lcon}{self.args.alpha_lconsis}_best.pt", epoch=epoch, best_acc=val_loss_min)

                self.save_checkpoint(file_name=f"ckpt_{self.args.embed_dim}_{self.args.patch_size}_{self.args.alpha_lmse}{self.args.alpha_lcon}{self.args.alpha_lconsis}_final.pt", epoch=epoch, best_acc=val_loss_min)

        print("Training Finished !, Best Accuracy: ", val_loss_min)
        return val_loss_min


