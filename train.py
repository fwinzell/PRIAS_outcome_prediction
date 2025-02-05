import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
import argparse
import datetime
from topk.svm import SmoothTop1SVM
import sys
import matplotlib.pyplot as plt
import yaml

from models import PRIAS_Model
from dataset import PRIAS_Generic_Dataset
from prias_file_reader import PRIAS_Data_Object
from losses import Longitudinal_Loss

class PRIAS_Trainer(object):
    """
    Trainer class for training and validating one model in PRIAS
    Arguments:
        model: The model to train (PRIAS_Model)
        tr_dataset: The training dataset
        val_dataset: The validation dataset
        max_epochs: The maximum number of epochs to train for
        batch_size: The batch size to use for training
        learning_rate: The learning rate to use for training
        optimizer_name: The name of the optimizer to use (Adam or SGD)
        optimizer_hparams: The hyperparameters for the optimizer
        **Optional**
        save_dir: The directory to save the model and logs to
        class_weights: The class weights to use for the loss function
        use_svmloss: Whether to use the SVM loss function (from CLAM paper)
        long_mode: Whether to use the longitudinal labels
        debug_mode: Whether to run in debug mode (no workers)
        log: Whether to log to tensorboard
        save: Whether to save the model
        model_name: The name of the model
    """
    def __init__(self,
                 model,
                 tr_dataset,
                 val_dataset,
                 max_epochs,
                 batch_size,
                 learning_rate,
                 optimizer_name,
                 optimizer_hparams,
                 save_dir=None,
                 class_weights=None,
                 use_svmloss=False,
                 long_mode=False,
                 debug_mode=False,
                 log=True,
                 save=True,
                 model_name=""):
        super().__init__()

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.model = model
        self.tr_dataset = tr_dataset
        self.val_dataset = val_dataset
        self.max_epochs = max_epochs
        self.batch_sz = batch_size
        self.lr = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        if class_weights is not None:
            self.weights = torch.tensor(class_weights).to(self.device)
        else:
            self.weights = None
        self.save = save
        self.log = log
        self.model_name = model_name
        self.long_mode = long_mode

        if save_dir is not None:
            self.save_dir = self.create_save_dir(os.path.join(save_dir, "models"))
        else:
            self.save = False
            self.log = False



        if use_svmloss:
            self.loss_fn = SmoothTop1SVM(n_classes = 1)
        else:
            if self.long_mode:
                #self.loss_fn = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=self.weights)
                self.loss_fn = Longitudinal_Loss(reduction='mean', pos_weight=self.weights)
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        if debug_mode:
            self.workers = 0
        else:
            self.workers = 20

        self.tr_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        self.optimizer, self.scheduler = self.configure_optimizers()

        if self.log:
            self.writer = SummaryWriter(self.create_save_dir(os.path.join(save_dir, "logs")))
        else:
            self.writer = None
        self.n_iter = 0
        self.best_acc = 0

    def train_dataloader(self):
        return DataLoader(self.tr_dataset, batch_size=self.batch_sz,
                          shuffle=True, drop_last=False, num_workers=self.workers)

    def save_config(self, config):
        with open(os.path.join(self.save_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(config), f)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_sz,
                          shuffle=False, drop_last=False, num_workers=self.workers)

    def get_model(self):
        return self.model

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

        # Linear warm up of learning rate
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
        # We will reduce the learning rate by 0.1 after 50 and 75 epochs
        decay = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75], gamma=0.1)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[10])
        return optimizer, scheduler

    def _step(self, inst, label):
        logits, y_prob, y_hat = self.model(inst)
        loss = self.loss_fn(logits.squeeze(), label.squeeze())
        if torch.isnan(loss):
            print("Loss is nan")
        if torch.isinf(loss):
            print("Loss is inf")
        #pred = torch.eq(y_hat, label).int()
        return loss, y_hat, y_prob


    def train(self):

        for epoch in range(self.max_epochs):
            self.model.train()
            tr_loop = tqdm(self.tr_loader)
            acc = 0
            losses = []
            for i, (pid, inst, label) in enumerate(tr_loop):
                inst, label = inst.squeeze().to(self.device), label.to(self.device).float()

                self.optimizer.zero_grad()
                loss, y_hat, _ = self._step(inst, label)
                pred = torch.eq(y_hat, label).float()
                pred = pred[~torch.isnan(label)]

                if self.log:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=self.n_iter)

                loss.backward()
                self.optimizer.step()
                self.n_iter += 1
                losses.append(loss.item())

                tr_loop.set_description(f"Epoch [{epoch}/{self.max_epochs}]")
                tr_loop.set_postfix(loss=loss.item(), pred=pred.cpu().numpy())
                if self.long_mode:
                    acc += int(torch.mean(pred))
                else:
                    acc += int(pred)

                for p in self.model.parameters():
                    if torch.any(torch.isnan(p)):
                        print("Model parameters are nan")
                        plt.plot(range(len(losses)), losses, 'ro-')
                        plt.show()


            acc /= len(self.tr_loader)
            print(f"\nEpoch {epoch}: Mean Accuracy: {acc}")
            if self.log:
                self.writer.add_scalar('train_acc', acc, global_step=epoch)
            self.validate(epoch)
            self.scheduler.step()

        if self.save:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, self.model_name + 'last.pth'))

    def validate(self, ep):
        self.model.eval()

        accs = torch.zeros(len(self.val_loader))
        losses = torch.zeros(len(self.val_loader))
        with torch.no_grad():
            val_loop = tqdm(self.val_loader)
            for i, (pid, inst, label) in enumerate(val_loop):
                inst, label = inst.squeeze().to(self.device), label.to(self.device).float()

                loss, y_hat, y_prob = self._step(inst, label)

                if self.long_mode:
                    accs[i] = int(torch.mean(torch.eq(y_hat, label).float()))
                else:
                    accs[i] = torch.eq(y_hat, label).int() #label.item() * y_prob.item() + (1-label.item()) * (1-y_prob.item())  # Cross-entropy ish
                losses[i] = loss.item()

                val_loop.set_description("Validation: ")
                val_loop.set_postfix(loss=loss.item(), acc=accs[i])

        val_acc = torch.mean(accs)
        val_loss = torch.mean(losses)
        if self.log:
            self.writer.add_scalar("val_acc", val_acc.item(), global_step=ep)
            self.writer.add_scalar("val_loss", val_loss.item(), global_step=ep)
            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], global_step=ep)

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            print('____New best model____')
            if self.save:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, self.model_name + "best.pth"))


def main(config):
    PDO = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)
    dataset = PRIAS_Generic_Dataset(
        path_to_dir=config.base_dir,
        patient_data_obj=PDO,
        xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx",
        xls_sheet_name=1,
        use_last_visit=config.use_last,
        use_features=config.use_features,
        use_long_labels=config.long_labels,
        p_augmentation=config.p_aug,
        top_k_features=config.top_k_patches,
    )

    tr_dataset, _ = dataset.return_splits(set_split=0.0)
    _, val_dataset = dataset.return_splits(set_split=0.1)

    prias_model = PRIAS_Model(config,
                              use_features=config.use_features,
                              long_mode=config.long_labels,
                              feature_size=config.num_features).to(device)

    save_name = "{}_{}".format(config.architecture, config.date)
    trainer = PRIAS_Trainer(
        model=prias_model,
        tr_dataset=tr_dataset,
        val_dataset=val_dataset,
        max_epochs=config.epochs,
        batch_size=1,
        learning_rate=config.lr,
        optimizer_name='Adam',
        optimizer_hparams={'lr': config.lr,
                           'betas': config.betas,
                           'eps': config.eps,
                           'weight_decay': config.weight_decay},
        save_dir=os.path.join(config.save_dir, save_name),
        use_svmloss=False,
        long_mode=config.long_labels,
        debug_mode=check_debug_mode()
        #class_weights=dataset.class_weights()
    )

    trainer.train()
    trainer.save_config(config)


def parse_config():
    parser = argparse.ArgumentParser("argument for run cnn pipeline")

    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--use_features", type=bool, default=True)

    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Python/PRIAS/prias_models")
    parser.add_argument("--base_dir", type =str, default="/home/fi5666wi/PRIAS_data/features_imagenet_256")
    parser.add_argument("--num_features", type=int, default=1920) # 1024 for UNI, 1920 for densenet
    parser.add_argument("--architecture", type=str, default="imagenet_gated", # densenet201_gated, imagenet201_gated, vitL_gated
                        help="Only for naming purposes")

    parser.add_argument("--use_last", type=bool, default=False,
                        help="To use only last visit, if False, all visits for active patients will be used")
    parser.add_argument("--long_labels", type=bool, default=False, help="Use longitudinal labels")

    parser.add_argument("--val_split", type=float, default=0.1)

    # Optimizer arguments, from CLAM paper
    parser.add_argument("--weight_decay", type=float, default=0.0005)  #0.0005
    parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.999])
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--loss_temperature", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0001) # innan: 0.001

    parser.add_argument("--top_k_patches", type=int, default=0,
                        help="Number of top k patches to use in the classifier, if zero all will be used")
    parser.add_argument("--hidden_layers", type=int, default=0, help="Number of hidden layers in the classifier")
    parser.add_argument("-a", "--use_augmentation", action='store_true')
    parser.add_argument("--p_aug", type=float, nargs=2, default=[0.5, 0.5],
                        help="p augmentation for (dropout, gaussian noise)")

    parser.add_argument("--dropout", type=bool, default=True)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--binary", type=bool, default=True)

    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed for reproducible experiment (new: 2)')

    args = parser.parse_args()
    return args

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def check_debug_mode():
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        print("Training")
        return False
    elif gettrace():
        print("Running in debug mode")
        return True
    else:
        return None


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("running on: {}".format(device))

    config = parse_config()
    seed_torch(config.seed)
    config.date = str(datetime.date.today())
    main(config)
