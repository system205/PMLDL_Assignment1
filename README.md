# Author
- Name: Arsen Mutalapov
- Email: a.mutalapov@innopolis.university
- Group: BS21-DS02


# How to

First of all you should **```pip install -r requirements.txt```**

Secondly, make sure you execute any of the following command being in the repo's root directory

P.s: you can find all the code usage with example outputs in _```notebooks/final_solution.ipynb```_

## Use - make predictions

```py
from src.models.predict_model import predict

texts =  ["Write your sentence", "Here is another sentence"]

# It will print the paraphrased versions
translations = predict('./', texts) # First arg - root path
```

## Transform data

1. Preprocess dataset: **```python ./src/data/preprocess_dataset.py --root=./```**
1. Prepare the dataset for training: **```python ./src/models/filter_split_train_val.py --root=./```**

## Train models

1. Check baseline metric: **```python ./src/models/metric/compute_metric.py --dataframe_file=./data/interim/val.csv --predictions_column=reference```**

1. Train simple model with swearing removal: **```python ./src/models/train_simple_model.py --project_path=<full-path like /content/PMLDL_Assignment1> --root=./```**

    a. Predict on your dataset: **```python ./src/models/predict_simple_model.py --dataframe_file=./data/interim/val.csv --root=./```**

1. Trainig T5 with: **```python ./src/models/train_model.py --train_path=./data/interim/train.csv --val_path=./data/interim/val.csv --save_dir=./models/output_dir/```**

1. Comput metric on your dataset specifying predictions column ('predictions' for swearing removal model): **```python ./src/models/metric/compute_metric.py --dataframe_file=./data/interim/val.csv --predictions_column=predictions```**
