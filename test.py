import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from glob import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader as GeoDataLoader

from utils.mvgnn import MVMGNN
from utils.data_utils import get_split_deterministic,load_graph_view_in_mem

def filter_subject(df_mri, label_key, label_column):
    # df_mri.map({'CN': 0, 'AD': 1})
    if label_column == "AD-CN":
        df_mri[label_key] = df_mri[label_key].map({'CN': 0, 'AD': 1})
    elif label_column == "AD-MCI":
        df_mri[label_key] = df_mri[label_key].map({'MCI': 0, 'AD': 1})
    elif label_column == "CN-MCI":
        df_mri[label_key] = df_mri[label_key].map({'CN': 0, 'MCI': 1})

    return df_mri

def build_test_loader(dataset_dict, samples, df, params):
    data_list = []

    for sid in tqdm(samples, desc="[x] Build test dataset"):
        g1, g2 = dataset_dict[sid]

        data = from_networkx(g1, group_edge_attrs=['weight'])
        data.image_id = sid

        label = df[df[params.id_key] == sid][params.label_key].values[0]
        data.y = torch.tensor([label], dtype=torch.float)

        node_list = list(g1.nodes())
        feats1 = np.stack([g1.nodes[n][params.feature_key] for n in node_list])
        data.x = torch.tensor(feats1, dtype=torch.float32)
        data.edge_attr = data.edge_attr.to(torch.float32)
        data.edge_weight = data.edge_attr.view(-1)

        data2 = from_networkx(g2, group_edge_attrs=['weight'])
        feats2 = np.stack([g2.nodes[n][params.feature_key] for n in node_list])
        data.x_2 = torch.tensor(feats2, dtype=torch.float32)
        data.edge_index_2 = data2.edge_index
        data.edge_attr_2 = data2.edge_attr.to(torch.float32)
        data.edge_weight_2 = data.edge_attr_2.view(-1)

        data_list.append(data)

    return GeoDataLoader(data_list, batch_size=1, shuffle=False, num_workers=params.n_workers)

def split_dataset_category_by_patient(df: pd.DataFrame,template_ratio: float,seed: int,
    label_key: str = "DIAGNOSIS", id_key: str = "image_id", subject_key: str = "Subject",):

    if label_key not in df.columns:
        raise ValueError(f"df 中缺少 label 列: {label_key}")
    if id_key not in df.columns:
        raise ValueError(f"df 中缺少 id 列: {id_key}")
    if subject_key not in df.columns:
        raise ValueError(f"df 中缺少 subject 列: {subject_key}")

    rng = np.random.RandomState(seed)

    training_samples: List[str] = []
    template_samples: List[str] = []

    labels = sorted(df[label_key].unique())

    for lab in labels:
        df_lab = df[df[label_key] == lab]

        subjects = df_lab[subject_key].dropna().unique()
        if len(subjects) == 0:
            continue

        subjects = np.array(sorted(subjects))
        perm = rng.permutation(len(subjects))
        subjects = subjects[perm]

        if template_ratio <= 0.0:
            n_temp = 0
        else:
            n_temp = int(np.floor(len(subjects) * template_ratio))
            if n_temp == 0 and len(subjects) > 0:
                n_temp = 1

        template_subj = set(subjects[:n_temp])
        train_subj = set(subjects[n_temp:])

        df_temp = df_lab[df_lab[subject_key].isin(template_subj)]
        df_train = df_lab[df_lab[subject_key].isin(train_subj)]

        template_samples.extend(df_temp[id_key].astype(str).tolist())
        training_samples.extend(df_train[id_key].astype(str).tolist())

    training_samples = list(dict.fromkeys(training_samples))
    template_samples = list(dict.fromkeys(template_samples))

    return training_samples, template_samples


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(f"{args.data_path}/mri_info.csv")

    df = df[df[args.label_key].isin(args.label_columns.split("-"))]
    df = filter_subject(df, args.label_key, args.label_columns)

    train_ids, template_ids = split_dataset_category_by_patient(
        df, args.template_ratio, args.seed,
        label_key=args.label_key,
        id_key=args.id_key,
        subject_key=args.subject_key
    )

    train_pairs = set(map(tuple, df.loc[df[args.id_key].isin(train_ids), [args.subject_key, args.label_key]].astype(str).values))
    test_pairs = set(map(tuple, df.loc[df[args.id_key].isin(template_ids), [args.subject_key, args.label_key]].astype(str).values))

    overlap = train_pairs & test_pairs
    print("Subject overlap:", len(overlap))
    print("无 Subject 泄露" if not overlap else ">>> 泄露名单 <<<\n" + "\n".join(map(str, sorted(overlap))))


    g_tmp = pickle.load(open(glob(f"{args.data_path}/*_norm.pkl")[0], "rb"))
    args.feature_channel = g_tmp.nodes[list(g_tmp.nodes())[0]][args.feature_key].shape[0]
    args.feature_num = len(g_tmp.nodes())

    g_tmp2 = pickle.load(open(glob(f"{args.data_path2}/*_norm.pkl")[0], "rb"))
    args.feature_channel2 = g_tmp2.nodes[list(g_tmp2.nodes())[0]][args.feature_key].shape[0]


    dataset_all = load_graph_view_in_mem(
        args.data_path, df,
        id_key=args.id_key,
        data_path2=args.data_path2
    )
    dataset_test = {k: dataset_all[k] for k in template_ids}
    test_loader = build_test_loader(dataset_test, template_ids, df, args)


    model = (MVMGNN(in_dim=args.feature_channel,
                        in_dim2=args.feature_channel2,
                        node_num=args.feature_num,
                        hidden_dim=args.gnn_feat,
                        num_layers=args.gnn_layers,
                        dropout=args.dropout_p,
                        mlp_hidden=args.gnn_feat).to(device))

    ckpt = os.path.join(args.exp, "model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    print(f"[✓] Loaded model: {ckpt}")


    preds, gts = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Testing]"):
            batch = batch.to(device)
            y = batch.y.view(-1)

            out, _ = model(batch)
            prob = torch.sigmoid(out.view(-1)).cpu().numpy()

            preds.extend((prob >= 0.5).astype(int).tolist())
            gts.extend(y.cpu().numpy().tolist())


    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds)
    prec = precision_score(gts, preds)
    sen = recall_score(gts, preds)

    tn, fp, fn, tp = confusion_matrix(gts, preds).ravel()
    spec = tn / (tn + fp)

    result = {
        "task": args.label_columns,
        "fold": getattr(args, "fold_name", ""),
        "acc": acc,
        "f1": f1,
        "pre": prec,
        "sen": sen,
        "spec": spec
    }

    print("\n========== Test Results ==========")
    print(f"ACC : {acc:.4f}")
    print(f"F1  : {f1:.4f}")
    print(f"PRE : {prec:.4f}")
    print(f"SEN : {sen:.4f}")
    print(f"SPEC: {spec:.4f}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--id_key', type=str, default="image_id")
    parser.add_argument('--subject_key', type=str, default="Subject")
    parser.add_argument('--label_key', type=str, default="DIAGNOSIS")
    parser.add_argument('--feature_key', type=str, default="feature")

    parser.add_argument('--gnn_feat', type=int, default=128)
    parser.add_argument('--gnn_layers', type=int, default=4)
    parser.add_argument('--dropout_p', type=float, default=0.3)

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--cv_max', type=int, default=5)
    parser.add_argument('--template_ratio', type=float, default=0.2)
    parser.add_argument('--n_workers', type=int, default=4)

    parser.add_argument('--label_columns', type=str, default="AD-CN")
    parser.add_argument('--exp', type=str, default="")
    parser.add_argument('--data_path', type=str,
                        default=r"data/View_1")
    parser.add_argument('--data_path2', type=str,
                        default=r"data/View_2")

    args = parser.parse_args()
    all_results = []

    for i in ["1","2","3","4","5"]:
        for j in ["AD-CN", "AD-MCI", "CN-MCI"]:
            print(f"------------------   Start_fold_{i}_task_{j}   ----------------------")
            args.exp = os.path.join("model_path/MVMGNN", i, j)
            args.fold_name = i
            args.label_columns = j

            config_path = os.path.join(args.exp, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            args.gnn_feat = int(config["gnn_feat"])
            args.gnn_layers = int(config["gnn_layers"])

            result = main(args)
            all_results.append(result)

    result_df = pd.DataFrame(all_results)

    summary = []
    for (task), g in result_df.groupby(["task"], sort=False):
        acc_mean = g["acc"].mean()
        acc_std = g["acc"].std(ddof=0)

        f1_mean = g["f1"].mean()
        f1_std = g["f1"].std(ddof=0)

        pre_mean = g["pre"].mean()
        pre_std = g["pre"].std(ddof=0)

        sen_mean = g["sen"].mean()
        sen_std = g["sen"].std(ddof=0)

        spec_mean = g["spec"].mean()
        spec_std = g["spec"].std(ddof=0)

        summary.append({
            "task": task,
            "acc": f"{acc_mean:.4f}±{acc_std:.4f}",
            "f1": f"{f1_mean:.4f}±{f1_std:.4f}",
            "pre": f"{pre_mean:.4f}±{pre_std:.4f}",
            "sen": f"{sen_mean:.4f}±{sen_std:.4f}",
            "spec": f"{spec_mean:.4f}±{spec_std:.4f}",
        })

    summary_df = pd.DataFrame(summary)
    print("\n\n========== Summary Results ==========")
    print(summary_df.to_string(index=False))

