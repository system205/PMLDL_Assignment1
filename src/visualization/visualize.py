''' This module contains a single function to visualize the data of a dataframe '''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.option_context('mode.use_inf_as_na', True)


def visualize(dataframe):
    '''Plots histograms of lenght_diff, similarity and ref_tox of a dataframe
        reference_length, translation_length, difference_length'''
    nrows, ncols = 3,3
    
    fig, axis = plt.subplots(nrows, ncols, figsize=(20, 10))

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
    plt.show()

