import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm


class PCa_Module(object):
    def __init__(self,
                 model,
                 train_data,
                 val_data,
                 max_epochs,
                 batch_size,
                 learning_rate,
                 optimizer_name,
                 optimizer_hparams,
                 save_dir,
                 class_weights=None,
                 binary=False):

        super().__init__()

        self.model = model
        self.tr_dataset = train_data
        self.val_dataset = val_data
        self.max_epochs = max_epochs
        self.batch_sz = batch_size
        self.lr = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.save_dir = self.create_save_dir(save_dir)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.binary = binary
        self.num_classes = 2 if binary else 4

        # initialize training utils
        if self.binary:
            self.loss_module = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.loss_module = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

        self.tr_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        self.optimizer, self.scheduler = self.configure_optimizers()

        self.writer = SummaryWriter(os.path.join(self.save_dir, 'logs'))
        self.n_iter = 0
        self.best_acc = 0

    def train_dataloader(self):
        return DataLoader(self.tr_dataset, batch_size=self.batch_sz, shuffle=True, drop_last=False, num_workers=20)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_sz, shuffle=False, drop_last=False, num_workers=20)

    def create_save_dir(self, path):
        dir_exists = True
        i = 0
        while dir_exists:
            save_dir = os.path.join(path, f"version_{str(i)}")
            dir_exists = os.path.exists(save_dir)
            i += 1
        os.makedirs(save_dir)
        return save_dir


    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.model.parameters(), **self.optimizer_hparams)
        elif self.optimizer_name == "SGD":
            optimizer = optim.SGD(self.model.parameters(), **self.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 50 and 75 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
        return optimizer, scheduler


    def _train_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        out = self.model(imgs)
        if self.binary:
            preds = torch.round(F.sigmoid(out))
            loss = self.loss_module(out.squeeze(), labels.float())
            acc = torch.sum(preds.squeeze(dim=1) == labels) / preds.size(dim=0)
        else:
            preds = F.one_hot(torch.argmax(out, dim=1), num_classes=self.num_classes)
            loss = self.loss_module(out, labels)
            acc = (preds == labels).all(dim=1).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        return loss, acc # Return tensor to call ".backward" on

    def train(self):
        for epoch in range(self.max_epochs):
            self.model.train()
            tr_loop = tqdm(self.tr_loader)
            for batch_idx, batch in enumerate(tr_loop):
                self.optimizer.zero_grad()
                loss, acc = self._train_step(batch, batch_idx)

                self.writer.add_scalar('train_loss', loss, global_step=self.n_iter)
                self.writer.add_scalar('train_acc', acc, global_step=self.n_iter)

                loss.backward()
                self.optimizer.step()
                self.n_iter += 1

                tr_loop.set_description(f"Epoch [{epoch}/{self.max_epochs}]")
                tr_loop.set_postfix(loss=loss.item(), acc=acc.item())
                # print("Iteration:[{0}][{1}] loss: {loss:.4f}".format(batch_idx, len(self.tr_loader), loss=loss.item()))

            self.validate(epoch)
            self.scheduler.step()

        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'last.pth'))

    def _val_step(self, batch):
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        out = self.model(imgs)
        if self.binary:
            preds = torch.round(F.sigmoid(out))
            acc = torch.sum(preds.squeeze(dim=1) == labels) / preds.size(dim=0)
        else:
            preds = F.one_hot(torch.argmax(out, dim=1), num_classes=self.num_classes)
            acc = (preds == labels).all(dim=1).float().mean()
        return acc

    def validate(self, ep):
        self.model.eval()

        accs = torch.zeros(len(self.val_loader))
        with torch.no_grad():
            val_loop = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(val_loop):
                acc = self._val_step(batch)

                accs[batch_idx] = acc.item()

                val_loop.set_description("Validation: ")
                val_loop.set_postfix(acc=acc.item())

        val_acc = torch.mean(accs)
        self.writer.add_scalar("val_acc", val_acc, global_step=ep)
        self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], global_step=ep)

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            print('____New best model____')
            torch.save(self.model.state_dict(), os.path.join(self.save_dir,'best.pth'))





