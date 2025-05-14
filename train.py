from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import PairedDataset
from models.unet import PBPUNet

import random
import itertools
from collections import defaultdict

@dataclass
class TrainOptions:
    lr: int
    batch_size: int
    train_root: str
    val_root: str
    run_name: str
    log_dir: str
    checkpoint_dir: str
    checkpoint: str | None = None
    val_interval: int = 10
    device: str = "cuda"

class Trainer:
    def __init__(self, opts: TrainOptions):
        self.opts = opts
        self.device = opts.device
        self.train_loader, self.val_loader = self._configure_loaders(infinite=True)
        self.model = self._init_model()
        self.optimizer = self._configure_optimizer()
        self.logger = SummaryWriter(log_dir=opts.log_dir)

    def _configure_loaders(self, infinite):
        self.train_ds = PairedDataset(self.opts.train_root)
        self.val_ds = PairedDataset(self.opts.val_root)
        train_loader = DataLoader(self.train_ds, batch_size=self.opts.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=self.opts.batch_size, shuffle=True)

        if infinite:
            train_loader = itertools.cycle(train_loader)

        return train_loader, val_loader

    def _init_model(self):
        model = PBPUNet(1,3).to(self.device)

        if self.opts.checkpoint is not None:
            ckpt = torch.load(self.opts.checkpoint, weights_only=False)
            self.global_step = ckpt["step"]
            model.load_state_dict(ckpt["state_dict"])
        else:
            self.global_step = 0

        return model

    def _configure_optimizer(self):
        return Adam(self.model.parameters(), lr = self.opts.lr)
    
    def _log_dict(self, tag, loss_dict, step):
        for k, v in loss_dict.items():
            self.logger.add_scalar(f"{tag}/{k}", v, step)

    def _log_image(self, x, y, y_hat, step):
        self.logger.add_image("image/input", x[0], step)
        self.logger.add_image("image/output", y_hat[0], step)
        self.logger.add_image("image/target", y[0], step)

    def calc_loss(self, y, y_hat):
        loss_dict = dict()

        l1_loss = F.l1_loss(y_hat, y, reduction='mean')
        loss_dict["l1"] = l1_loss.item()

        return l1_loss, loss_dict


    def train_step(self, step):
        self.model.train()

        x, y = next(self.train_loader)
        x, y = x.to(self.device), y.to(self.device)        

        y_hat = self.model(x)

        loss, loss_dict = self.calc_loss(y, y_hat)

        self._log_dict("train", loss_dict, step)

        return loss

    def validate(self, step):
        self.model.eval()

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)        

            total_loss = 0.0
            total_loss_dict = defaultdict(int)

            with torch.no_grad():
                y_hat = self.model(x)
                loss, loss_dict = self.calc_loss(y, y_hat)
                total_loss += loss.item()

                for k, v in loss_dict.items():
                    total_loss_dict[k] += v

            self._log_dict("val", total_loss_dict, step)

        #sample one batch and log one image 
        x, y = random.choice(self.val_ds)
        x, y = x.unsqueeze(0).to(self.device), y.unsqueeze(0).to(self.device)        
        with torch.no_grad():
            y_hat = self.model(x)

        self._log_image(x, y, y_hat, step)

    def train(self, num_steps):
        for step in range(num_steps):
            loss = self.train_step(self.global_step)

            #optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()                
            print(loss)
        
            if self.global_step % self.opts.val_interval == 0:
                self.validate(self.global_step)

            #TODO: add checkpointing

            self.global_step += 1

if __name__ == "__main__":
    opts = TrainOptions(
        1e-4,
        1,
        "/home/progs/dev/YWAI/demo-images/",
        "/home/progs/Pictures/maagi_potabo/",
        run_name="exp0",
        log_dir="logs",
        checkpoint_dir="checkpoints",
        device="cuda"
    )

    trainer = Trainer(opts)
    trainer.train(1000)