import pandas as pd
import torch
import numpy as np
import os

from torch.utils.data import DataLoader
import argparse
import datetime


from models import PRIAS_Model
from dataset import CrossValidation_Dataset
from prias_file_reader import PRIAS_Data_Object
from train import PRIAS_Trainer, check_debug_mode
from eval import eval

def parse_config():
    parser = argparse.ArgumentParser("Argument for running whole cross validation pipeline")

    parser.add_argument("--K", type=int, default=6)

    #parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--use_features", type=bool, default=True)

    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Python/PRIAS/prias_models")
    parser.add_argument("--base_dir", type =str, default="/home/fi5666wi/PRIAS_data/features_uni")
    parser.add_argument("--feature_size", type=int, default=1024)
    parser.add_argument("--last_visit_only", type=bool, default=False)

    # Optimizer arguments, from CLAM paper
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--loss_temperature", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0001) # innan: 0.001

    # Other training arguments
    parser.add_argument("--architecture", type=str, default="vitL_gated")
    parser.add_argument("--hidden_layers", type=int, default=0, help="Number of hidden layers in the classifier")

    parser.add_argument("--dropout", type=bool, default=True)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--binary", type=bool, default=True)
    parser.add_argument("--p_aug", type=float, nargs=2, default=(0.5, 0.5), help="p augmentation for (dropout, gaussian noise)")
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducible experiment (default: 1)')

    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--log", type=bool, default=True)

    args = parser.parse_args()
    return args

def create_dataframe(config, accs, scores):
    data = {
        'Date': config.date,
        'Architecture': config.architecture,
        'Seed': config.seed
    }

    for i in range(config.K):
        data[f"Accuracy Fold {i+1}"] = accs[i]
        data[f"Score Fold {i+1}"] = scores[i]

    data["MA"] = np.mean(accs)
    data["MS"] = np.mean(scores)

    return pd.DataFrame(data=data, index=[0])


def validation(model, val_dataset, verbose=True):
    loader = DataLoader(val_dataset, batch_size=1)
    acc = 0
    score = 0
    model.eval()
    with torch.no_grad():
        for i, (pid, inst, label) in enumerate(loader):
            inst, label = inst.squeeze().to(device), label.to(device).float()
            logits, y_prob, y_hat = model(inst)

            score += label.item() * y_prob.item() + (1 - label.item()) * (1 - y_prob.item())  # Cross-entropy ish
            acc += torch.eq(y_hat, label).int()
            if verbose:
                print(f"Label: {label.item()}, Prediction: {y_hat.item()} (p={y_prob.item()}")
    if verbose:
        print(f"Accuracy: {acc.item()/len(loader)}")

    return acc.item()/len(loader), score/len(loader)


def train(config, tr_dataset, val_dataset, save_path):
    prias_model = PRIAS_Model(config,
                              use_features=config.use_features,
                              feature_size=config.feature_size,
                              hidden_layers=config.hidden_layers).to(device)

    model_name = config.architecture + "_"
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
        save_dir=save_path,
        use_svmloss=False,
        debug_mode=check_debug_mode(),
        save=config.save,
        log=config.log,
        model_name=model_name
    )

    trainer.train()

    return trainer.get_model()

def run(config):
    PDO = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)
    cv_dataset = CrossValidation_Dataset(
        path_to_dir=config.base_dir,
        patient_data_obj=PDO,
        xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels_2.xlsx",
        xls_sheet_name=1,
        n_folds=config.K,
        seed=1,
        use_features=config.use_features,
        use_last_visit=config.last_visit_only,
        p_augmentation=config.p_aug,
        use_long_labels=False,
    )
    save_name = os.path.join(config.save_dir, "cross_val_{}".format(config.date))
    dir_exists = True
    i = 0
    while dir_exists:
        save_path = os.path.join(save_name, f"run_{str(i)}")
        dir_exists = os.path.exists(save_path)
        i += 1

    accs, scores = np.zeros(config.K), np.zeros(config.K)
    for i in range(config.K):
        print(f"##### Training on fold: {i} #####")
        tr_data_fold, val_data_fold = cv_dataset.return_splits(K=i)
        model = train(config, tr_data_fold, val_data_fold, save_path)

        accs[i], scores[i] = validation(model, val_data_fold, verbose=True)
        #accs[i], scores[i] = eval(model, device=device, base_dir=config.base_dir, verbose=True)  # on test data

    print("#### Finished ####")
    print(f"Mean Accuracy: {np.mean(accs)} +/- ({np.std(accs)})")
    print(f"Mean Likelihood Score: {np.mean(scores)} +/- ({np.std(scores)})")

    df = create_dataframe(config, accs, scores)
    #with pd.ExcelWriter("/home/fi5666wi/Documents/PRIAS sheets/Results.xlsx", mode='a', if_sheet_exists="overlay") as writer:
    #    df.to_excel(writer, sheet_name='treatment prediction', header=False, index=False, float_format="%.4f", startrow=1)
    csv_path = "/home/fi5666wi/Documents/PRIAS sheets/results_treatment_pred.csv"
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False)
    else:
        df.to_csv(csv_path)


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("running on: {}".format(device))

    config = parse_config()
    seed_torch(config.seed)
    config.date = str(datetime.date.today())
    run(config)

