'''Main module combined all data preparation'''
import argparse
from dataframe_preprocessing import preprocess, remove_unknowns
from text_preprocessing import simple_row_preprocessing
from make_dataset import load_dataframe

def preprocess_dataset(root):
    '''Load raw data, preprocess it and save the interim data'''
    dataframe = load_dataframe(root)

    df_processed = preprocess(dataframe)
    df_processed = df_processed.apply(simple_row_preprocessing, axis=1)
    df_processed = remove_unknowns(df_processed)

    df_processed.to_csv(f"{root}data/interim/preprocessed.csv", index=False, )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    args = parser.parse_args()

    if (args.root is None): 
        print("You have to specify the root of your project")
        exit(1)

    preprocess_dataset(args.root)
