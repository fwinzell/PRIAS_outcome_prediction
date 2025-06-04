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
from train import PRIAS_Trainer, PRIAS_Longitudinal_Trainer, PRIAS_Survival_Trainer, check_debug_mode

from torchmetrics import ROC, AUROC
from sklearn.metrics import balanced_accuracy_score

def parse_config():
    parser = argparse.ArgumentParser("Argument for running whole cross validation pipeline")

    parser.add_argument("--K", type=int, default=6, help="Number of folds")

    #parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--use_features", type=bool, default=True)

    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Python/PRIAS/prias_models")
    parser.add_argument("--base_dir", type =str, default="/home/fi5666wi/PRIAS_data/features_from_bengio")
    parser.add_argument("--num_features", type=int, default=1536) # 1024 for UNI, 1920 for densenet 1536 for UNIv2
    parser.add_argument("--last_visit_only", type=bool, default=False)
    parser.add_argument("--filter_old_slides", type=bool, default=True, help="Filter out slides from before 2011 in the dataset")

    #parser.add_argument("--label_path", type=str, default="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx")
    parser.add_argument("--prias_sheet_path", type=str, default="/home/fi5666wi/Documents/PRIAS sheets") #20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx")

    # Optimizer arguments, from CLAM paper
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.999])
    parser.add_argument("--eps", type=float, default=1e-8)

    # Loss arguments
    parser.add_argument("--loss_temperature", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.00001) # innan: 0.001 reduced again 2025-05-22
    parser.add_argument("--loss", type=str, default="bce", help="BCE : binary cross entropy, AEM : attention entropy loss, SVM : SVM loss")
    parser.add_argument("--aem_lambda", type=float, default=1.0, help="Lambda for attention entropy loss")

    # Other training arguments
    parser.add_argument("--architecture", type=str, default="vitH_gated", help="only for naming files")
    parser.add_argument("--hidden_layers", type=int, default=0, help="Number of hidden layers in the classifier")

    parser.add_argument("--dropout", type=bool, default=True)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--binary", type=bool, default=True)
    parser.add_argument("--p_aug", type=float, nargs=2, default=[0.5, 0.5], help="p augmentation for (dropout, gaussian noise)")
    parser.add_argument("--k_random_samples", type=int, default=-1, help="Number of random samples to use for training, If -1 use all samples")
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducible experiment (default: 1)')

    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--log", type=bool, default=True)

    # Longitudinal arguments
    parser.add_argument("--long_labels", type=bool, default=False, help="Use longitudinal labels")
    parser.add_argument("--survival_mode", type=bool, default=True)
    parser.add_argument("--time_intervals", type=int, nargs="+", default=[0, 3, 5], help="Time intervals for longitudinal labels")
    parser.add_argument("--n_month_mode", type=bool, default=True, help="Use n_month mode for survival analysis")
    parser.add_argument("--gg_mode", type=bool, default=False, help="Select the instances with the highest GG score for training")

    args = parser.parse_args()
    return args

def create_dataframe(config, accs, bal_accs, aucs):
    data = {
        'Date': config.date,
        'Architecture': config.architecture,
        'Seed': config.seed
    }

    for i in range(config.K):
        data[f"Accuracy Fold {i+1}"] = accs[i]
        data[f"BAc Fold {i+1}"] = bal_accs[i]
        data[f"AUC Fold {i+1}"] = aucs[i]

    data["Mean Acc"] = np.mean(accs)
    data["Mean BAc"] = np.mean(bal_accs)
    data["Mean AUC"] = np.mean(aucs)

    return pd.DataFrame(data=data, index=[0])


def validation(model, val_dataset, device, verbose=True):
    loader = DataLoader(val_dataset, batch_size=1)
    auc = AUROC(task="binary")
    probs = torch.zeros(len(loader))
    preds = torch.zeros(len(loader))
    true = torch.zeros(len(loader))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pid, feats, label, = batch
            feats, label = feats.to(device), label.to(device)
            logits, y_prob, y_hat, *a = model(feats.squeeze())
            probs[i] = y_prob
            preds[i] = y_hat
            true[i] = label
            if verbose:
                print(f"Patient: {pid.item()} \nProbability: {y_prob.item()} Predicition: {y_hat.item()} Label: {label.item()}")

    acc = torch.sum(torch.eq(preds, true))/len(loader)
    bal_acc = balanced_accuracy_score(true.cpu().numpy(), preds.cpu().numpy())

    auc_val = auc(probs, true.int())

    if verbose:
        print(f"Accuracy: {acc}, Balanced accuracy: {bal_acc}, AUC: {auc_val}")

    return acc, bal_acc, auc_val

def survival_validation(model, val_dataset, device, verbose=True):
    from lifelines.utils import concordance_index
    loader = DataLoader(val_dataset, batch_size=1)

    model.eval()
    events = []
    t_preds = []
    t_true = []

    with torch.no_grad():
        for i, (pid, inst, label) in enumerate(loader):
            inst = inst.squeeze().to(device)
            label = [l.to(device) for l in label]

            t_pred, avec = model(inst)
        
            events.append(label[0].item())  
            t_true.append(label[1].item())
            t_preds.append(t_pred.item())
            if verbose:
                print(f"Patient: {pid.item()} \nT (pred): {t_pred.item()} T: {label[1].item()} Event: {label[0].item()}")
                
    c_index = concordance_index(t_true, t_preds, events)
    mean_error = np.mean(np.abs(np.array(t_preds) - np.array(t_true)))
    return c_index, mean_error



def train(config, tr_dataset, val_dataset, save_path, device):
    prias_model = PRIAS_Model(config,
                              use_features=config.use_features,
                              long_mode=config.long_labels,
                              survival_mode=config.survival_mode,
                              n_follow_up=len(config.time_intervals)+1,
                              hidden_layers=config.hidden_layers,
                              feature_size=config.num_features,
                              return_attention=config.loss == "aem").to(device)

    model_name = config.architecture + "_"

    if config.long_labels:
        trainer = PRIAS_Longitudinal_Trainer(
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
            debug_mode=check_debug_mode(),
            survival_mode=config.survival_mode
        )
    elif config.survival_mode:
        trainer = PRIAS_Survival_Trainer(
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
            debug_mode=check_debug_mode(),
            model_name=model_name,
            device=device
        )
    else:
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
            loss=config.loss,
            aem_lambda=config.aem_lambda,
            debug_mode=check_debug_mode(),
            model_name="",
            device=device
        )

    
    trainer.save_config(config)
    trainer.train()
    
    return trainer.get_model()

def run(config, device):
    PDO = PRIAS_Data_Object(
        os.path.join(config.prias_sheet_path, "20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx"),
        sheet_name=0)
    
    cv_dataset = CrossValidation_Dataset(
        path_to_dir=config.base_dir,
        patient_data_obj=PDO,
        xls_path=os.path.join(config.prias_sheet_path, "PRIAS_labels.xlsx"),
        xls_sheet_name=1,
        n_folds=config.K,
        seed=1,
        use_features=config.use_features,
        use_last_visit=config.last_visit_only,
        p_augmentation=config.p_aug,
        use_long_labels=config.long_labels,
        survival_mode=config.survival_mode,
        long_time_intervals=config.time_intervals,
        filter_old=config.filter_old_slides,
        n_month_mode=config.n_month_mode,
        random_sampling=True,
        top_k_features=config.k_random_samples,
        gg_dataset=config.gg_mode
    )
    save_name = os.path.join(config.save_dir, "cross_val_{}".format(config.date))
    dir_exists = True
    i = 0
    while dir_exists:
        save_path = os.path.join(save_name, f"{config.architecture}_{str(i)}")
        dir_exists = os.path.exists(save_path)
        i += 1

    accs, bal_accs, aucs = np.zeros(config.K), np.zeros(config.K), np.zeros(config.K)
    for i in range(config.K):
        print(f"##### Training on fold: {i} #####")
        tr_data_fold, val_data_fold = cv_dataset.return_splits(K=i)
        model = train(config, tr_data_fold, val_data_fold, save_path, device)

        if config.survival_mode:
            cidx, t_err = survival_validation(model, val_data_fold, device=device, verbose=True)
            aucs[i] = cidx
            accs[i] = t_err
        else:
            accs[i], bal_accs[i], aucs[i] = validation(model, val_data_fold, device=device, verbose=True)
        #accs[i], scores[i] = eval(model, device=device, base_dir=config.base_dir, verbose=True)  # on test data

    print("#### Finished ####")
    print(f"Mean Accuracy: {np.mean(accs)} +/- ({np.std(accs)})")
    print(f"Mean Balanced Accuracy: {np.mean(bal_accs)} +/- ({np.std(bal_accs)})")
    print(f"Mean AUC: {np.mean(aucs)} +/- ({np.std(aucs)})")
    

    """df = create_dataframe(config, accs, bal_accs, aucs)
    csv_path = "/home/fi5666wi/Documents/PRIAS sheets/results_outcome_pred.csv"
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False)
    else:
        df.to_csv(csv_path)"""


def seed_torch(device, seed=7):
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
    seed_torch(device, config.seed)
    config.date = str(datetime.date.today())
    run(config, device)

