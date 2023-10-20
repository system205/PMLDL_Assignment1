import zipfile
import os
import pandas as pd

def unzip_dataset(zip_file_path = '../../data/raw/filtered_paranmt.zip', extracted_dir = '../../data/raw/'):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)

def load_dataframe(extracted_dir = '../../data/raw/', tsv_file_name = 'filtered.tsv' ):
    # Construct the full path to the TSV file
    tsv_file_path = os.path.join(extracted_dir, tsv_file_name)

    # Check file existence
    if not os.path.exists(tsv_file_path):
        unzip_dataset()

    # Read the TSV file into a DataFrame
    dataframe = pd.read_csv(tsv_file_path, delimiter='\t')

    return dataframe.drop(columns=['Unnamed: 0'])
