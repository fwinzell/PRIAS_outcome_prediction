import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn, optim
from torchsummary import summary
from tqdm import tqdm
import argparse
import datetime
import matplotlib.pyplot as plt

from models import RiskFactor_Model
from dataset import PRIAS_Generic_Dataset
from prias_file_reader import PRIAS_Data_Object
from train import check_debug_mode
from eval import get_eval_dataset

class RF_Trainer(object):
    """
    Trainer class for training a model to predict risk factors on features from the PRIAS dataset.
    Arguments:
        model: The model to train.
        tr_dataset: The training dataset.
        val_dataset: The validation dataset.
        max_epochs: The maximum number of epochs to train for.
        batch_size: The batch size to use for training.
        learning_rate: The learning rate to use for training.
        n_factors: The number of factors to predict.
        save_dir: The directory to save the model and logs to.
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
                 n_factors=2,
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
        self.n_factors = n_factors
        self.tr_dataset = tr_dataset
        self.val_dataset = val_dataset
        self.max_epochs = max_epochs
        self.batch_sz = batch_size
        self.lr = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams

        self.save = save
        self.log = log
        self.model_name = model_name

        if save_dir is not None:
            self.save_dir = self.create_save_dir(os.path.join(save_dir, "models"))
        else:
            self.save = False
            self.log = False

        self.loss_fn = nn.MSELoss(reduction='mean')

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
        self.best_loss = np.inf

    def train_dataloader(self):
        return DataLoader(self.tr_dataset, batch_size=self.batch_sz,
                          shuffle=True, drop_last=False, num_workers=self.workers)

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


    def filter_nana(self, inst, label):
        filter = torch.ones(self.batch_sz, dtype=torch.bool).to(device)
        for i in range(self.n_factors):
            filter = filter & ~torch.isnan(label[i])
        return inst[filter], [rf[filter] for rf in label]


    def _step(self, inst, label):
        y_hat = self.model(inst)
        loss = [self.loss_fn(y_hat.squeeze()[i], label[i].squeeze()) for i in range(self.n_factors)]
        #if torch.isnan(loss):
        #    print("Loss is nan")
        #if torch.isinf(loss):
        #    print("Loss is inf")
        #pred = torch.eq(y_hat, label).int()
        return loss, y_hat


    def train(self):

        for epoch in range(self.max_epochs):
            self.model.train()
            tr_loop = tqdm(self.tr_loader)

            losses = []
            for i, (pid, inst, _, factor) in enumerate(tr_loop):

                inst = inst.squeeze().to(self.device)
                label = [rf.to(self.device).float() for rf in factor]

                # filter out nans i.e missing factor values
                #inst, label = self.filter_nana(inst, label)
                # find if there are any nans in the labels
                isnans = [torch.any(torch.isnan(l)) for l in label]
                if torch.any(torch.tensor(isnans)):
                    continue

                self.optimizer.zero_grad()
                loss, y_hat = self._step(inst, label)

                if self.log:
                    self.writer.add_scalar('train_loss', sum(loss), global_step=self.n_iter)

                for l in loss:
                    l.backward(retain_graph=True)

                self.optimizer.step()
                self.n_iter += 1
                losses.append(sum(loss))

                tr_loop.set_description(f"Epoch [{epoch}/{self.max_epochs}]")
                tr_loop.set_postfix(loss=sum(loss), pred=y_hat.detach().cpu().numpy())


                for p in self.model.parameters():
                    if torch.any(torch.isnan(p)):
                        print("Model parameters are nan")
                        plt.plot(range(len(losses)), losses, 'ro-')
                        plt.show()

            acc = sum(losses) / len(losses)
            print(f"\nEpoch {epoch}: Mean Loss: {acc}")
            if self.log:
                self.writer.add_scalar('train_acc', acc, global_step=epoch)
            self.validate(epoch)
            self.scheduler.step()

        if self.save:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'last.pth'))

        return self.model

    def validate(self, ep):
        self.model.eval()

        losses = torch.zeros(len(self.val_loader))
        with torch.no_grad():
            val_loop = tqdm(self.val_loader)
            for i, (pid, inst, _, factor) in enumerate(val_loop):
                inst = inst.squeeze().to(self.device)
                label = [rf.to(self.device).float() for rf in factor]

                isnans = [torch.any(torch.isnan(l)) for l in label]
                if torch.any(torch.tensor(isnans)):
                    continue

                loss, y_hat = self._step(inst, label)

                losses[i] = sum(loss)

                val_loop.set_description("Validation: ")
                val_loop.set_postfix(loss=sum(loss), pred=y_hat.cpu().numpy())

        val_loss = torch.mean(losses)
        if self.log:
            self.writer.add_scalar("val_loss", val_loss.item(), global_step=ep)
            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], global_step=ep)

        if val_loss < self.best_loss:
            self.best_acc = val_loss
            print('____New best model____')
            if self.save:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best.pth"))

def main(config):
    """
    Main method for training a model to predict risk factors on features from the PRIAS dataset.
    Returns trained model
    """
    PDO = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)
    dataset = PRIAS_Generic_Dataset(
        path_to_dir=config.feature_dir,
        patient_data_obj=PDO,
        xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels_2.xlsx",
        xls_sheet_name=1,
        use_last_visit=False,
        use_features=True,
        use_long_labels=False,
        include_factors=True
    )

    tr_dataset, val_dataset = dataset.return_splits(set_split=0.1)

    model = RiskFactor_Model(input_size=config.num_features, num_factors=config.num_factors, hidden_layers=2).to(device)

    save_name = "{}_{}".format(config.architecture, config.date)
    trainer = RF_Trainer(
        model=model,
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
        n_factors=config.num_factors,
        save_dir=os.path.join(config.save_dir, save_name),
        debug_mode=check_debug_mode(),
        save=True,
        log=True,
        model_name=config.architecture
    )

    return trainer.train()


def parse_config():
    parser = argparse.ArgumentParser("argument for run cnn pipeline")

    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--use_features", type=bool, default=True)

    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Python/PRIAS/prias_models/risk_factor")
    parser.add_argument("--feature_dir", type=str, default="/home/fi5666wi/PRIAS_data/features_lower_densenet")
    parser.add_argument("--val_split", type=float, default=0.1)

    # Optimizer arguments, from CLAM paper
    parser.add_argument("--weight_decay", type=float, default=0.0005)  #0.0005
    parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--loss_temperature", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01) # innan: 0.001

    parser.add_argument("--architecture", type=str, default="risk_factor_model")
    parser.add_argument("--num_features", type=int, default=1920)

    parser.add_argument("--dropout", type=bool, default=True)
    parser.add_argument("--num_factors", type=int, default=2)
    parser.add_argument("--binary", type=bool, default=True)

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducible experiment (default: 1)')

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

def make_eval_plots(y_hat, label, n_factors=2, names=["PSA", "Volume"]):
    fig, axs = plt.subplots(1, n_factors, figsize=(10, 5))
    for i in range(n_factors):
        axs[i].set_title(names[i])
        axs[i].scatter(label[i], y_hat[i])

        # Regression line
        k, m = np.polyfit(label[i], y_hat[i], 1)
        axs[i].plot(label[i], k * label[i] + m, 'r-')

        # Plot diagonal line
        axs[i].plot(label[i], label[i], 'k-')

        axs[i].set_xlabel("True")
        axs[i].set_ylabel("Predicted")

    plt.show()

def eval_model(path_to_model, path_to_data, model=None, n_factors=2):
    """
    Method to evaluate a trained model on the test set. If model is not provided, it will load the model from the path
    Different sheets of test data can be used by changing the xls_sheet_name argument in get_eval_dataset:
        0: All patients
        1: train
        2: test

    """
    if model is None:
        model = RiskFactor_Model(input_size=1920, num_factors=n_factors)
        model.load_state_dict(torch.load(path_to_model))
        model.eval()
    else:
        model.eval()

    model = model.to(device)

    testset = get_eval_dataset(path_to_data,
                               xls_sheet_name=2,
                               use_features=True,
                               use_long_labels=False,
                               use_last_visit=False,
                               include_factors=True)

    testloader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    total_mean_loss = 0
    y_pred = np.zeros((n_factors, len(testloader)))
    y_true = np.zeros((n_factors, len(testloader)))
    with torch.no_grad():
        for i, (pid, inst, _, factor) in enumerate(testloader):
            inst = inst.squeeze().to(device)
            label = [rf.to(device).float() for rf in factor]

            isnans = [torch.any(torch.isnan(l)) for l in label]
            if torch.any(torch.tensor(isnans)):
                continue

            y_hat = model(inst)

            y_pred[:,i] = y_hat.cpu().numpy().squeeze()
            y_true[:,i] = [label[ii].cpu().numpy().squeeze() for ii in range(n_factors)]
            loss = [nn.functional.mse_loss(y_hat.squeeze()[i], label[i].squeeze()) for i in range(n_factors)]
            print(f"Patient: {pid.item()}, Loss: {sum(loss)}")
            print(f"True: {[l.item() for l in label]}, Pred: {y_hat.cpu().numpy()}")
            total_mean_loss += sum(loss)

    make_eval_plots(y_pred, y_true, n_factors=n_factors)
    print(f"Mean loss: {total_mean_loss/len(testloader)}")


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("running on: {}".format(device))

    config = parse_config()
    seed_torch(config.seed)
    config.date = str(datetime.date.today())
    model = main(config)
    path_to_model = "/home/fi5666wi/Python/PRIAS/prias_models/risk_factor/risk_factor_model_2024-01-12/models/version_2/last.pth"
    eval_model(path_to_model=path_to_model, path_to_data=config.feature_dir, model=model, n_factors=config.num_factors)