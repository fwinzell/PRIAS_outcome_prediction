import torch
import os
import numpy as np
from torch import nn
from torchsummary import summary
from torchvision import models
import argparse
import datetime

from image_loaders import get_datasets, get_weights
from modules import PCa_Module
from resnet import resnet18, resnet34, resnet50, resnet101, count_parameters
from densenet import densenet201, densenet169

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
# Benchmark useful when input sizes does not change -> faster runtime
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("running on: {}".format(device))


def parse_config():
    parser = argparse.ArgumentParser("argument for run cnn pipeline")

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("-e", "--epochs", type=int, default=99)
    parser.add_argument("--patch_size", type=int, default=299)

    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Python/PRIAS/supervised/saved_models")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--save_every_n_epochs", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.1)

    # Optimizer arguments
    parser.add_argument("--weight_decay", type=float, default=10e-6)
    parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--loss_temperature", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=0.01)  # innan 0.1

    parser.add_argument("--architecture", type=str, default="densenet201")
    parser.add_argument("--use_fcn", type=bool, default=True)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--binary", type=bool, default=False)

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
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def train_model(config):
    sz = config.patch_size

    # trainsets = ['/home/fi5666wi/Documents/Gleason/Data_1_SUS/20181120_SUS/train_' + str(sz)]
    # testsets = ['/home/fi5666wi/Documents/Gleason/Data_1_SUS/20181120_SUS/test_' + str(sz)]

    trainsets = ['/home/fi5666wi/Documents/Gleason/Data_1_SUS/20181120_SUS/train_' + str(sz),
                 '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_SUS/train_' + str(sz),
                 '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Helsingborg/train_' + str(sz),
                 '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Linkoping/train_' + str(sz),
                 '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Rotterdam/train_' + str(sz)]

    """
    testsets = ['/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_SUS/test_' + str(sz),
                '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Helsingborg/test_' + str(sz),
                '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Linkoping/test_' + str(sz),
                '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Rotterdam/test_' + str(sz)]
    """

    tr_dataset, val_dataset = get_datasets(trainsets, val_split=0.1, binary=False)
    # val_dataset, _ = get_datasets(testsets, val_split=0.0, binary=True)
    class_weights = get_weights(tr_dataset.targets, num_classes=config.num_classes, one_hot=True)

    print("Number of training images: {}".format(len(tr_dataset)))
    if config.architecture == 'resnet18':
        model = resnet18(config)
    elif config.architecture == 'resnet34':
        model = resnet34(config)
    elif config.architecture == 'resnet50':
        model = resnet50(config)
    elif config.architecture == 'resnet101':
        model = resnet101(config)
    elif config.architecture == 'densenet169':
        model = densenet169(num_classes=config.num_classes)
    elif config.architecture == 'densenet201':
        model = densenet201(num_classes=config.num_classes)
    else:
        NotImplementedError("Architecture not implemented")

    save_name = "{}_{}".format(config.architecture, config.date)
    module = PCa_Module(model,
                        train_data=tr_dataset, val_data=val_dataset, max_epochs=config.epochs,
                        batch_size=config.batch_size, learning_rate=config.learning_rate,
                        optimizer_name='Adam',
                        optimizer_hparams={'lr': config.learning_rate,
                                           'betas': config.betas,
                                           'eps': config.eps,
                                           'weight_decay': config.weight_decay},
                        save_dir=os.path.join(config.save_dir, save_name),
                        class_weights=None,
                        binary=config.binary)

    # ModelSummary(cnn, max_depth=-1)
    summary(model.to(device), (3, config.patch_size, config.patch_size), device=device)

    module.train()


if __name__ == '__main__':
    config = parse_config()
    seed_torch(config.seed)
    config.date = str(datetime.date.today())
    train_model(config)
