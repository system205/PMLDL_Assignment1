import zipfile
import os
import pandas as pd

def unzip_dataset(root, zip_file_path = 'data/raw/filtered_paranmt.zip', extracted_dir = 'data/raw/'):
    with zipfile.ZipFile(root+zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(root+extracted_dir)

def load_dataframe(root, extracted_dir = 'data/raw/', tsv_file_name = 'filtered.tsv' ):
    # Construct the full path to the TSV file
    tsv_file_path = os.path.join(root+extracted_dir, tsv_file_name)

    # Check file existence
    if not os.path.exists(tsv_file_path):
        unzip_dataset(root)

    # Read the TSV file into a DataFrame
    dataframe = pd.read_csv(tsv_file_path, delimiter='\t')

    return dataframe.drop(columns=['Unnamed: 0'])


def load_train(root, train_filename='data/interim/train.csv'):
    return pd.read_csv(root+train_filename)

def load_validation(root, train_filename='data/interim/val.csv'):
    return pd.read_csv(root+train_filename)