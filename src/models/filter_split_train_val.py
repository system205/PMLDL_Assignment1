import argparse
from sklearn.model_selection import train_test_split
import pandas as pd

def filter(root, input_file='data/interim/preprocessed.csv'):
    dataframe = pd.read_csv(root+input_file)
    dataframe = dataframe[dataframe['length_difference'] < 15]
    dataframe = dataframe[dataframe['trn_length'] < 70]
    dataframe = dataframe[dataframe['trn_tox'] < 0.002]
    dataframe = dataframe[dataframe['ref_tox'] > 0.95]
    dataframe = dataframe[dataframe['similarity'] > 0.8]
    return dataframe

def split(root, dataframe, train_proportion=0.8, output_path='data/interim/'):
    train, val = train_test_split(dataframe, train_size=train_proportion, random_state=42)
    train.to_csv(f"{root+output_path}train.csv", index=False)
    val.to_csv(f"{root+output_path}val.csv", index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    args = parser.parse_args()
    
    if(args.root is None):
        print('Specify root')
        exit(1)
    
    split(args.root, filter(args.root))
