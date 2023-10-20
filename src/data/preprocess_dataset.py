'''Main module combined all data preparation'''
from dataframe_preprocessing import preprocess, remove_unknowns
from text_preprocessing import simple_row_preprocessing
from make_dataset import load_dataframe

def preprocess_dataset():
    '''Load raw data, preprocess it and save the interim data'''
    dataframe = load_dataframe()

    df_processed = preprocess(dataframe)
    df_processed = df_processed.apply(simple_row_preprocessing, axis=1)
    df_processed = remove_unknowns(df_processed)

    df_processed.to_csv("../../data/interim/preprocessed.csv", index=False, )

preprocess_dataset()
