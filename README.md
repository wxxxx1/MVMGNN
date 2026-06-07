# MVMGNN: Multi-View Masked Graph Neural Network for Alzheimer’s Disease Diagnosis using Structural MRI

## Environment

The required environment is provided in `environment.yml`.

## Project Structure

```text
MVMGNN/
├── data/
│   ├── View_1/
│   └── View_2/
├── model_path/
│   └── MVMGNN/
├── utils/
├── related.py
├── baselines.py
├── train.py
├── test.py
├── environment.yml
└── README.md
```

where:

* `data`: data directory.
* `model_path`: directory used to save trained model weight files.
* `utils`: network structure definitions and data loading tools.
* `baselines.py`: baseline training code.
* `related.py`: related work code.
* `test.py`: model testing entry.
* `train.py`: model training entry.
* `environment.yml`: environment configuration file.
* `README.md`: project description file.

## Data

The `data` folder is recommended to be organized as follows:

```text
data/
├── View_1/
│   ├── mri_info.csv
│   ├── Ixxxxx.pkl
│   └── ...
├── View_2/
│   ├── mri_info.csv
│   ├── Ixxxxx.pkl
│   └── ...
```

where:

* `mri_info.csv`: MRI image information file, which should contain the `Subject` column and the `DIAGNOSIS` column.
* `Ixxxxx.pkl`: preprocessed graph file.

## PKL Graph Data Structure

In this project, each `.pkl` file is saved as a `networkx.Graph` object, which is used to represent the brain network constructed from a single subject or a single MRI image.

The graph information includes:

```python
{
    "name": roi_name,
    "centroid_voxel": [x, y, z],
    "centroid_mm": [x, y, z],
    "feature": feature_vector,
    "edge_index": edge_index,
    "edge_weight": edge_weight
}
```

where:

* `name`: name of the brain region of interest (ROI).
* `centroid_voxel`: centroid coordinate of the ROI in voxel space.
* `centroid_mm`: centroid coordinate of the ROI in physical space.
* `feature`: feature vector of the ROI.
* `edge_index`: edge connection index of the graph.
* `edge_weight`: edge weight of the graph.

## model_path

The `model_path` folder is used to save trained model weight files.

```text
model_path/
├── MVMGNN/
│   ├── fold/
│   │   ├── AD-CN/
│   │   │   ├── best_performance.csv
│   │   │   ├── config.json
│   │   │   ├── model.pt
```

where:

* `best_performance.csv`: used to save the best model performance.
* `config.json`: used to save the training parameters.
* `model.pt`: used to save the model weights.

## Train

Set the data path:

```python
parser.add_argument('--data_path', type=str, default=r"data/View_1")
parser.add_argument('--data_path2', type=str, default=r"data/View_2")
```

Set the classification task:

```python
parser.add_argument('--label_columns', type=str, default="AD-CN")
```

## Test

Set the data path:

```python
parser.add_argument('--data_path', type=str, default=r"data/View_1")
parser.add_argument('--data_path2', type=str, default=r"data/View_2")
```

Set the classification task:

```python
parser.add_argument('--label_columns', type=str, default="AD-CN")
```

Set the model weight path:

```python
args.exp = os.path.join("model_path/model_name/fold_name/task")
```
