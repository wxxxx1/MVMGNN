import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import json
import pandas as pd
import pickle
from glob import glob
from tqdm import tqdm
from types import SimpleNamespace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_auc_score
from torch_geometric.utils import from_networkx
from utils.data_utils import (split_dataset_category_by_patient, get_split_deterministic,
                              load_graph_in_mem, filter_subject, load_graph_view_in_mem)

from utils.BrainGNN.braingnn import Network
from utils.gvae import MVGNN
from utils.admgnn import meatGCN
from utils.bagnn import Model


class GCN_Trainer:

    def __init__(self, params, device='cuda'):
        self.params = params
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.rand = np.random.RandomState(self.params.seed)
        self.__init_dataset__()
        self.__init_model__()

    def __graph_to_pyg__(self, g):
        """Convert networkx graph to PyG Data object."""
        data = from_networkx(g)
        data.x = torch.tensor(np.stack([g.nodes[n][self.params.feature_key] for n in g.nodes]), dtype=torch.float)
        return data

    def __init_model__(self):
        train_params = {"MOMENTUM": self.params.momentum,
                        "OPTIMIZER": self.params.optimizer,
                        "EPOCH_ITERS": self.params.n_iters,
                        "LR_DECAY": 0.1,
                        "LR_STEP": [50, 100, 200],
                        'LR': self.params.lr}

        if self.params.model == 'bagnn':
            self.model = (Model(in_dim=self.params.feature_channel,
                                node_num=self.params.feature_num,
                                hidden_dim=self.params.gnn_feat,
                                num_layers=self.params.gnn_layers,
                                dropout=self.params.dropout_p).to(self.device))

        elif self.params.model == 'braingnn':
            self.model = (Network(indim=self.params.feature_channel,
                                  ratio = 0.8,
                                  nclass = 2).to(self.device))
        elif self.params.model == 'MVGNN':
            self.model = (MVGNN(in_dim=self.params.feature_channel,
                                node_num=self.params.feature_num,
                                hidden_dim=self.params.gnn_feat,
                                num_layers=self.params.gnn_layers).to(self.device))
        elif self.params.model == 'meatGCN':
            self.model = meatGCN(in_dim=self.params.feature_channel,
                                    in_dim2=self.params.feature_channel2,
                                    node_num=self.params.feature_num,
                                    hidden_dim=self.params.gnn_feat,
                                    num_layers=self.params.gnn_layers,
                                    dropout=self.params.dropout_p, mlp_hidden=self.params.gnn_feat).to(self.device)

        if train_params['OPTIMIZER'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=train_params['LR'], momentum=train_params['MOMENTUM'],
                                  nesterov=True)
        elif train_params['OPTIMIZER'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=train_params['LR'])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params['LR_STEP'],
                                                   gamma=train_params['LR_DECAY'])

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.BCEWithLogitsLoss()

    def __build_dataloader__(self, dataset_dict, samples, batch_size=8, shuffle=True):
        """
        Build a PyG DataLoader from networkx graphs in dataset_dict for the given sample ids.
        Uses torch_geometric.loader.DataLoader for correct batching.
        Correctly maps labels using self.df[self.params.id_key] -> self.params.label_key,
        and treats the **first** entry in self.label_columns as the positive class (label=1).
        """
        from torch_geometric.utils import from_networkx
        from torch_geometric.loader import DataLoader as GeoDataLoader

        # build id->label map once (string labels)
        if self.params.id_key not in self.df.columns or self.params.label_key not in self.df.columns:
            raise ValueError(f"Dataframe missing required columns: {self.params.id_key} or {self.params.label_key}")

        data_list = []
        for sid in tqdm(samples, desc="[x] Build dataset"):
            g_item = dataset_dict[sid]
            g1, g2 = g_item

            # View 1
            node_list = list(g1.nodes())
            data = from_networkx(g1, group_edge_attrs=['weight'])  # View1作为data基础

            data.image_id = sid
            label = self.df[self.df[self.params.id_key] == sid][self.params.label_key].values[0]
            data.y = torch.tensor([label], dtype=torch.float)

            feats = np.stack([g1.nodes[n][self.params.feature_key] for n in node_list])
            data.x = torch.tensor(feats, dtype=torch.float32)
            data.edge_attr = data.edge_attr.to(torch.float32)
            data.edge_weight = data.edge_attr.view(-1)

            data2 = from_networkx(g2, group_edge_attrs=['weight'])
            feats = np.stack([g2.nodes[n][self.params.feature_key] for n in node_list])
            data.x_2 = torch.tensor(feats, dtype=torch.float32)
            data.edge_index_2 = data2.edge_index
            data.edge_attr_2 = data2.edge_attr.to(torch.float32)
            data.edge_weight_2 = data.edge_attr_2.view(-1)

            if self.params.model == 'braingnn':
                data.pos = torch.zeros((self.params.feature_num, 200), dtype=torch.float32)
            else:
                data.pos = None

            data_list.append(data)

        if len(data_list) == 0:
            raise RuntimeError("No data found to build dataloader — check dataset and dataframe filtering.")

        loader = GeoDataLoader(data_list, batch_size=batch_size, shuffle=shuffle, num_workers=self.params.n_workers)
        return loader

    def __init_dataset__(self):
        # determine feature_channel automatically
        if self.params.feature_channel == -1:
            graph_file_path = glob(f"{self.params.data_path}/*_norm.pkl")[0]
            g = pickle.load(open(graph_file_path, 'rb'))
            self.params.feature_channel = g.nodes[list(g.nodes())[0]][self.params.feature_key].shape[0]
            self.params.feature_num = len(g.nodes())
            print(self.params.feature_channel, self.params.feature_num)

            graph_file_path2 = glob(f"{self.params.data_path2}/*_norm.pkl")[0]
            g2 = pickle.load(open(graph_file_path2, 'rb'))
            self.params.feature_channel2 = g2.nodes[list(g2.nodes())[0]][self.params.feature_key].shape[0]
            self.params.feature_num2 = len(g2.nodes())
            print(self.params.feature_channel2, self.params.feature_num2)

        self.label_columns = self.params.label_columns.split("-")
        assert len(self.label_columns) == 2  # binary graph matching

        df_mri = pd.read_csv(f"{self.params.data_path}/mri_info.csv")
        df_mri = df_mri[df_mri[self.params.label_key].isin(self.label_columns)]

        # keep rows with existing graph data pkl_file_path = os.path.join(f"{data_path}/{patient_id}_norm.pkl")
        def __file_exists(row):
            filename = f"{self.params.data_path}/{row[self.params.id_key]}_norm.pkl"
            return os.path.isfile(filename)

        df_mri = df_mri[df_mri.apply(__file_exists, axis=1)]
        self.df = filter_subject(df_mri, self.params.label_key, self.params.label_columns)

        for label, count in self.df[self.params.label_key].value_counts().items():
            print(f"{label}: {count}")

        training_samples, template_samples = split_dataset_category_by_patient(df_mri,
                                                                               self.params.template_ratio,
                                                                               self.params.seed,
                                                                               label_key=self.params.label_key,
                                                                               id_key=self.params.id_key,
                                                                               subject_key=self.params.subject_key)
        train_sub = set(self.df[self.df[self.params.id_key].isin(training_samples)][self.params.subject_key])
        test_sub = set(self.df[self.df[self.params.id_key].isin(template_samples)][self.params.subject_key])
        print("Subject overlap:", len(train_sub & test_sub))

        training_samples_1, test_samples = get_split_deterministic(training_samples,
                                                                   self.params.cv,
                                                                   self.params.cv_max)
        self.sample_train = training_samples
        self.sample_template = template_samples
        self.sample_test = test_samples

        print(f"training samples {len(training_samples)}, "
              f"test samples {len(test_samples)}, "
              f"template_samples {len(template_samples)}")

        def _print_split_stats(name, ids):
            df_split = self.df[self.df[self.params.id_key].isin(ids)]
            print(f"\n[{name}] n = {len(df_split)}")
            for label, count in df_split[self.params.label_key].value_counts().items():
                print(f"  {label}: {count}")

        _print_split_stats("Train", self.sample_train)
        _print_split_stats("Test", self.sample_test)
        _print_split_stats("Template", self.sample_template)

        dataset = load_graph_view_in_mem(self.params.data_path, df_mri, id_key=self.params.id_key,
                                         data_path2=self.params.data_path2)
        dataset_train, dataset_template, dataset_test = {}, {}, {}
        for k in self.sample_train:
            dataset_train[k] = dataset[k]
        for k in self.sample_template:
            dataset_template[k] = dataset[k]
        for k in self.sample_template:
            dataset_test[k] = dataset[k]

        self.dataset = {"train": dataset_train, "test": dataset_test, "template": dataset_template, "all": dataset}
        # dataloader_train = GMDataset(dataset_train, self.sample_train, self.df,
        #                              id_key=self.params.id_key, label_key=self.params.label_key, feature_key=self.params.feature_key, rand=self.rand)
        # self.dataloaders = {}
        # self.dataloaders['train'] = build_dataloader(dataloader_train, self.params.batch_size, self.params.n_workers, fix_seed=True, shuffle=True)

        self.train_loader = self.__build_dataloader__(self.dataset['train'], self.sample_train,
                                                      batch_size=self.params.batch_size, shuffle=True)
        self.test_loader = self.__build_dataloader__(self.dataset['template'], self.sample_template, batch_size=1,
                                                     shuffle=False)
        return training_samples, test_samples, template_samples

    def train(self):
        print("[x] Training started...")
        self.model.train()

        best_f1, wait = 0., 0
        for epoch in range(self.params.n_iters):
            total_loss = 0.0
            preds, gts = [], []

            # training loop
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.params.n_iters}"):
                batch = batch.to(self.device)
                # ensure labels are float vector shaped (N,)
                y = batch.y.view(-1).to(self.device).float()

                out, pro_loss = self.model(batch)  # expected shape (batch_size, 1)
                out = out.view(-1)  # shape (batch_size,)

                loss = self.criterion(out, y) + pro_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * y.size(0)

                probs = torch.sigmoid(out).detach().cpu().numpy()
                preds.extend(probs.tolist())
                gts.extend(y.cpu().numpy().tolist())

            # mean loss
            avg_loss = total_loss / (len(self.train_loader.dataset) if len(self.train_loader.dataset) > 0 else 1)
            # metrics (guard AUC if single class)
            preds_bin = np.round(preds)
            acc = accuracy_score(gts, preds_bin)

            print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f}, ACC: {acc:.4f}")

            self.scheduler.step()

            # periodic validation
            if epoch % self.params.n_eval == 0:
                acc, f1, precision, sensitivity, specificity = self.test(self.params.exp, epoch)
                print(
                    f"[x] test @ epoch {epoch}, acc = {acc}, f1 = {f1}, prec = {precision}, sen = {sensitivity}, spec = {specificity}")
                # save best
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(self.model.state_dict(), f"{self.params.exp}/model.pt")
                    df_perf = pd.DataFrame([{'acc': acc, 'f1': f1, 'precision': precision, 'sensitivity': sensitivity,
                                             'specificity': specificity}])
                    df_perf.to_csv(os.path.join(self.params.exp, "best_performance.csv"), index=False)

                    wait = 0
                    print(f"[!] New stopping patience {wait}/{self.params.patience}")
                else:
                    wait += 1
                    print(f"[!] Early stopping patience {wait}/{self.params.patience}")
                    if wait >= self.params.patience:
                        break

    def test(self, result_save_path=None, epoch=0, save_file=True):
        """
        Evaluate model on test set and save predictions + performance summary.
        - Saves three files:
            1. test_preds.csv  → image_id, pred (probability), gt
            2. performance.csv → acc, f1, precision, sensitivity, specificity, auc
            3. classification_report.txt and confusion_matrix.txt
        """
        self.model.eval()
        if result_save_path is None or not isinstance(result_save_path, str):
            result_save_path = os.path.join(self.params.exp, "results")
        os.makedirs(result_save_path, exist_ok=True)

        preds_prob, preds_label, gts, image_ids = [], [], [], []

        with (torch.no_grad()):
            for batch in tqdm(self.test_loader, desc="[Testing]"):
                batch = batch.to(self.device)

                y = batch.y.view(-1).float()
                out, _ = self.model(batch)
                out = out.view(-1)

                # convert logits to probabilities
                probs = torch.sigmoid(out).detach().cpu().numpy()
                preds_prob.extend(probs.tolist())

                # convert probabilities to binary categorical predictions
                bin_preds = (probs >= 0.5).astype(int)
                preds_label.extend(bin_preds.tolist())

                gts.extend(y.cpu().numpy().tolist())

                # record image IDs if available
                if hasattr(batch, 'image_id'):
                    image_ids.extend(batch.image_id)
                elif hasattr(batch, 'idx'):
                    image_ids.extend(batch.idx.cpu().numpy().tolist())
                else:
                    image_ids.extend(list(range(len(y))))

        acc = accuracy_score(gts, preds_label)
        f1 = f1_score(gts, preds_label)
        precision = precision_score(gts, preds_label)
        sensitivity = recall_score(gts, preds_label)  # same as recall
        cm = confusion_matrix(gts, preds_label)
        try:
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp)
        except Exception:
            specificity = float('nan')

        # --- print summary ---
        print(
            f"[x] Test ACC: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

        if save_file:
            # --- save predictions ---
            df_preds = pd.DataFrame({'image_id': image_ids, 'pred': preds_label, 'gt': gts})
            df_preds.to_csv(os.path.join(result_save_path, f"{epoch:04d}_test_preds.csv"), index=False)

            # --- save classification report ---
            clf_report = classification_report(gts, preds_label, digits=4)
            with open(os.path.join(result_save_path, f"{epoch:04d}_classification_report.txt"), "w") as f:
                f.write("Classification Report:\n")
                f.write(clf_report)

            # --- save confusion matrix ---
            with open(os.path.join(result_save_path, f"{epoch:04d}_confusion_matrix.txt"), "w") as f:
                f.write("Confusion Matrix (TN FP; FN TP):\n")
                np.savetxt(f, cm, fmt="%d", delimiter=" ")

            # --- save performance summary ---
            df_perf = pd.DataFrame([{'acc': acc, 'f1': f1, 'precision': precision, 'sensitivity': sensitivity,
                                     'specificity': specificity}])
            df_perf.to_csv(os.path.join(result_save_path, f"{epoch:04d}_performance.csv"), index=False)
        return acc, f1, precision, sensitivity, specificity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('--feature_channel', type=int, default=-1)
    parser.add_argument('--feature_num', type=int, default=-1)
    parser.add_argument('--gnn_feat', type=int, default=64)
    parser.add_argument('--gnn_layers', type=int, default=3)
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--model', type=str, default="GCN")

    # data parameters
    parser.add_argument('--label_columns', type=str, default="CN-MCI")
    parser.add_argument('--id_key', type=str, default="image_id")
    parser.add_argument('--subject_key', type=str, default='Subject')
    parser.add_argument('--label_key', type=str, default="DIAGNOSIS")
    parser.add_argument('--feature_key', type=str, default="feature")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--data_path', type=str,
                        default=r"data/View_1")
    parser.add_argument('--data_path2', type=str,
                        default=r"data/View_2")

    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--cv_max', type=int, default=5)
    parser.add_argument('--template_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=6)

    # training parameters
    parser.add_argument('--momentum', type=float, default=0.9)  # Momentum for optimizer
    parser.add_argument('--optimizer', type=str, default="adam")  # optimizer
    parser.add_argument('--n_iters', type=int, default=1000)  # number of epoches to be trained
    parser.add_argument('--n_eval', type=int, default=10)  # number of epoches for evaluation
    parser.add_argument('--lr_decay', type=float, default=0.1)  # learning rate decay
    parser.add_argument('--lr', type=float, default=1e-4)  # learning rate
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--patience', type=int, default=5)

    parser.add_argument('--flow', type=str, default="train")
    # exp
    parser.add_argument('--exp', type=str, default="exp_baseline")
    args = parser.parse_args()


    for ii in ["1", "2", "3", "4", "5"]:
        for feat in [32, 64, 96, 128]:
            for layer in [2, 3, 4, 5]:
                for i in ['bagnn', 'braingnn', 'MVGNN', 'meatGCN']:
                    for j in ["AD-CN", "AD-MCI", "CN-MCI"]:
                        print(f"------------------------   Start_{ii}_{i}_{j}   --------------------------------")
                        args.exp = os.path.join("xiaorong", f"{feat}_{layer}", ii, i, j)

                        if os.path.exists(args.exp):
                            print(f" {args.exp} : continue")
                            continue

                        args.model = i
                        args.label_columns = j

                        if args.label_columns == "AD-CN":
                            args.gnn_feat = 128
                            args.gnn_layers = 3
                        elif args.label_columns == "AD-MCI":
                            args.gnn_feat = 64
                            args.gnn_layers = 3
                        elif args.label_columns == "CN-MCI":
                            args.gnn_feat = 128
                            args.gnn_layers = 3

                        args.gnn_feat = feat
                        args.gnn_layers = layer

                        config_path = os.path.join(args.exp, "config.json")
                        os.makedirs(args.exp, exist_ok=True)
                        with open(config_path, "w") as f:
                            json.dump(vars(args), f, indent=4)
                        print(args)

                        if args.flow == "train":
                            gcn_train = GCN_Trainer(args)
                            gcn_train.train()
