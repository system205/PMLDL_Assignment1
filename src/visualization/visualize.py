''' This module contains a single function to visualize the data of a dataframe '''

import warnings
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.option_context('mode.use_inf_as_na', True)
warnings.simplefilter(action='ignore', category=FutureWarning)


def visualize(dataframe, output_file_path="./"):
    '''Plots histograms of lenght_diff, similarity and ref_tox of a dataframe
        reference_length, translation_length, difference_length'''
    nrows, ncols = 3,3
    
    _, axis = plt.subplots(nrows, ncols, figsize=(20, 10))

    columns = ['lenght_diff', 'similarity', 'ref_tox', 'ref_length',
            'trn_length', 'length_difference', 'trn_tox', 'tox_diff']
    labels = ['Length difference', 'Similarity', 'Reference toxicity Rating',
            'Reference length', 'Translation length', 
            'Difference in translation and reference lengths',
            'Translation toxicity',
            'Difference of toxicity'
            ]
    
    
    for i in range(len(columns)):
        ax = axis[i//ncols][i%ncols]
        sns.histplot(dataframe[columns[i]], bins=40, kde=True, ax=ax)
        ax.set_xlabel(labels[i])
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of ' + labels[i])

    
    plt.tight_layout()
    plt.savefig(output_file_path + 'dataset_visualization.png')
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataframe")
    parser.add_argument("--output_path")
    args = parser.parse_args()

    if (args.input_dataframe is None): 
        print("You have to specify dataframe path")
        exit(1)

    df = pd.read_csv(args.input_dataframe)
    if (args.output_path is not None):
        visualize(df, output_file_path=args.output_path)
    else:
        visualize(df)

