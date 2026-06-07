import os
import pickle
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd


def filter_subject(df_mri, label_key, label_column):
    # df_mri.map({'CN': 0, 'AD': 1})
    if label_column == "AD-CN":
        df_mri[label_key] = df_mri[label_key].map({'CN': 0, 'AD': 1})
    elif label_column == "AD-MCI":
        df_mri[label_key] = df_mri[label_key].map({'MCI': 0, 'AD': 1})
    elif label_column == "CN-MCI":
        df_mri[label_key] = df_mri[label_key].map({'CN': 0, 'MCI': 1})

    return df_mri

def split_dataset_category_by_patient(
    df: pd.DataFrame,
    template_ratio: float,
    seed: int,
    label_key: str = "DIAGNOSIS",
    id_key: str = "image_id",
    subject_key: str = "Subject",
) -> Tuple[List[str], List[str]]:
    """
    按【类别(label_key) + 患者(subject_key)】分层，把样本划成：
      - training_samples: 用于后续 CV 再划分 (train/test)
      - template_samples: 作为“模板图”的样本

    注意：
    - df 通常已经过滤为只包含感兴趣的类别，例如 AD / CN；
    - 以 subject 为单位划分，保证同一个病人的多个扫描不会分到两个集合。

    Returns
    -------
    training_samples : list[str]
        用于训练/交叉验证的 image_id 列表。
    template_samples : list[str]
        用于模板的 image_id 列表。
    """
    if label_key not in df.columns:
        raise ValueError(f"df 中缺少 label 列: {label_key}")
    if id_key not in df.columns:
        raise ValueError(f"df 中缺少 id 列: {id_key}")
    if subject_key not in df.columns:
        raise ValueError(f"df 中缺少 subject 列: {subject_key}")

    rng = np.random.RandomState(seed)

    training_samples: List[str] = []
    template_samples: List[str] = []

    # label 一般是 ["AD", "CN"]
    labels = sorted(df[label_key].unique())

    for lab in labels:
        df_lab = df[df[label_key] == lab]

        # 每个 label 内的 unique subject
        subjects = df_lab[subject_key].dropna().unique()
        if len(subjects) == 0:
            continue

        subjects = np.array(sorted(subjects))  # 固定排序 + perm 保证可复现
        perm = rng.permutation(len(subjects))
        subjects = subjects[perm]

        # 决定 template subject 数量
        if template_ratio <= 0.0:
            n_temp = 0
        else:
            n_temp = int(np.floor(len(subjects) * template_ratio))
            # 比例>0 但计算下来是 0，就至少取 1 个
            if n_temp == 0 and len(subjects) > 0:
                n_temp = 1

        template_subj = set(subjects[:n_temp])
        train_subj = set(subjects[n_temp:])

        # 按 subject 拿 image_id
        df_temp = df_lab[df_lab[subject_key].isin(template_subj)]
        df_train = df_lab[df_lab[subject_key].isin(train_subj)]

        template_samples.extend(df_temp[id_key].astype(str).tolist())
        training_samples.extend(df_train[id_key].astype(str).tolist())

    # 保险去重
    training_samples = list(dict.fromkeys(training_samples))
    template_samples = list(dict.fromkeys(template_samples))

    return training_samples, template_samples


def get_split_deterministic(
    samples: List[str],
    cv: int,
    cv_max: int,
) -> Tuple[List[str], List[str]]:
    """
    在给定的样本列表上做【确定性】K 折切分：
      - 假定 samples 的顺序已经打乱过；
      - 将 samples 按顺序平均分成 cv_max 份；
      - 取第 cv 份为 test，其余合并为 train。

    支持：
      - cv 为 0-based (0..cv_max-1)
      - 或 cv 为 1-based (1..cv_max)，内部会自动减 1。
    """
    rng = np.random.RandomState(42)
    samples = list(samples)
    rng.shuffle(samples)
    n = len(samples)

    if cv_max <= 1 or n == 0:
        # 不做 CV，全部用作训练
        return samples, []

    # 处理 cv 合法性和 0/1-based
    if 0 <= cv < cv_max:
        cv_idx = cv
    elif 1 <= cv <= cv_max:
        cv_idx = cv - 1
    else:
        raise ValueError(f"cv={cv} 不在合法范围内（0~{cv_max-1} 或 1~{cv_max}）。")

    # 每折大小：前 (n % cv_max) 折多 1 个
    fold_sizes = [n // cv_max] * cv_max
    for i in range(n % cv_max):
        fold_sizes[i] += 1

    start = sum(fold_sizes[:cv_idx])
    end = start + fold_sizes[cv_idx]

    test_samples = samples[start:end]
    train_samples = samples[:start] + samples[end:]

    return train_samples, test_samples


def load_graph_in_mem(
    data_path: str,
    df_mri: pd.DataFrame,
    id_key: str = "image_id",
    use_norm: bool = True,
) -> Dict[str, object]:
    """
    把 data_path 下对应的图 pkl 一次性加载到内存，返回 {image_id: graph} 字典。

    约定：
      - 图文件名为 "{image_id}_norm.pkl" 或 "{image_id}.pkl"
      - df_mri 至少有一列 id_key（默认 'image_id'）

    Parameters
    ----------
    data_path : str
        图数据所在的目录，例如 'data/aibl/radiomics_50'。
    df_mri : pd.DataFrame
        至少包含 id_key 列。
    id_key : str
        df 中的 ID 列名，例如 'image_id'。
    use_norm : bool
        若为 True，优先尝试 "{id}_norm.pkl"，找不到再尝试 "{id}.pkl"。

    Returns
    -------
    dataset : dict
        {image_id: networkx.Graph} 字典。
    """
    if id_key not in df_mri.columns:
        raise ValueError(f"df_mri 中缺少 id 列: {id_key}")

    dataset: Dict[str, object] = {}

    for _, row in df_mri.iterrows():
        sid = str(row[id_key])

        if use_norm:
            candidates = [
                os.path.join(data_path, f"{sid}_norm.pkl"),
                os.path.join(data_path, f"{sid}.pkl"),
            ]
        else:
            candidates = [
                os.path.join(data_path, f"{sid}.pkl"),
                os.path.join(data_path, f"{sid}_norm.pkl"),
            ]

        g_path = None
        for p in candidates:
            if os.path.isfile(p):
                g_path = p
                break

        if g_path is None:
            # 通常你前面已经过滤过，这里只做 silent skip 或打印 warning
            # print(f"[warn] 未找到 {sid} 对应的 pkl，已跳过。")
            continue

        with open(g_path, "rb") as f:
            g = pickle.load(f)

        dataset[sid] = g

    return dataset


def load_graph_view_in_mem(
    data_path: str,
    df_mri: pd.DataFrame,
    id_key: str = "image_id",
    use_norm: bool = True,
    data_path2: str = None,   # 新增：第二视图路径，默认为 None
):
    """
    把图一次性加载到内存。

    - 如果 data_path2 is None       -> 单视图：返回 {sid: g}
    - 如果 data_path2 不为 None     -> 双视图：返回 {sid: (g1, g2)}

    约定：
      - 图文件名为 "{sid}_norm.pkl" 或 "{sid}.pkl"
      - df_mri 至少有一列 id_key（默认 'image_id'）
    """
    if id_key not in df_mri.columns:
        raise ValueError(f"df_mri 中缺少 id 列: {id_key}")

    dataset: Dict[str, Any] = {}
    skipped = 0

    for _, row in df_mri.iterrows():
        sid = str(row[id_key])

        # ----------- 视图1 -----------
        if use_norm:
            candidates1 = [
                os.path.join(data_path, f"{sid}_norm.pkl"),
                os.path.join(data_path, f"{sid}.pkl"),
            ]
        else:
            candidates1 = [
                os.path.join(data_path, f"{sid}.pkl"),
                os.path.join(data_path, f"{sid}_norm.pkl"),
            ]

        g1_path = None
        for p in candidates1:
            if os.path.isfile(p):
                g1_path = p
                break

        if g1_path is None:
            skipped += 1
            continue

        # 单视图情况，直接加载 g1
        if data_path2 is None:
            with open(g1_path, "rb") as f:
                g1 = pickle.load(f)
            dataset[sid] = g1
            continue

        # ----------- 视图2（只有在 data_path2 不为空时才走到这里） -----------
        if use_norm:
            candidates2 = [
                os.path.join(data_path2, f"{sid}_norm.pkl"),
                os.path.join(data_path2, f"{sid}.pkl"),
            ]
        else:
            candidates2 = [
                os.path.join(data_path2, f"{sid}.pkl"),
                os.path.join(data_path2, f"{sid}_norm.pkl"),
            ]

        g2_path = None
        for p in candidates2:
            if os.path.isfile(p):
                g2_path = p
                break

        if g2_path is None:
            skipped += 1
            continue

        with open(g1_path, "rb") as f1:
            g1 = pickle.load(f1)
        with open(g2_path, "rb") as f2:
            g2 = pickle.load(f2)

        dataset[sid] = (g1, g2)

    mode = "dual-view" if data_path2 is not None else "single-view"
    print(f"[load_graph_in_mem] Loaded {len(dataset)} subjects ({mode}). Skipped {skipped}.")

    return dataset


