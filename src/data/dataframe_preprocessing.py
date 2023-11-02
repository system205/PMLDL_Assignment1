'''This module contains statistical preprocessing for a dataframe'''

import re


def swap_ref_trn(row):
    ''' It swaps the translation text and score with the reference one if 
        the toxicity of reference is lower the one of translation'''
    if row['ref_tox'] < row['trn_tox']:
        row['reference'], row['translation'] = row['translation'], row['reference']
        row['ref_tox'], row['trn_tox'] = row['trn_tox'], row['ref_tox']
    return row

def add_stat_columns(dataframe):
    ''' This function adds a new column "tox_diff" calculated as
        the difference between the toxicity scores of reference and translation
        rounded to 3 decimal points'''
    dataframe['tox_diff'] = round(dataframe['ref_tox'] - dataframe['trn_tox'], 3)
    return dataframe

def round_columns(dataframe):
    ''' This function rounds the dataframe columns "similarity" and "lenght_diff"
        to 2 decimal points'''
    dataframe['similarity'] = round(dataframe['similarity'], 2)
    dataframe['lenght_diff'] = round(dataframe['lenght_diff'], 2)
    return dataframe

def add_ref_trn_length(dataframe):
    dataframe['ref_length'] = dataframe['reference'].apply(len)
    dataframe['trn_length'] = dataframe['translation'].apply(len)

    return dataframe

def calculate_ref_trn_length(dataframe):
    dataframe['length_difference'] = (dataframe['ref_length'] - dataframe['trn_length']).apply(abs)

    return dataframe

def preprocess(dataframe):
    '''Applies the preprocessing functions on a dataframe:
        swap reference and translation
        add difference column
        round similarity and lenght_diff'''
    dataframe = dataframe.apply(swap_ref_trn, axis=1)
    dataframe = add_stat_columns(dataframe)
    dataframe = round_columns(dataframe)
    dataframe = add_ref_trn_length(dataframe)
    dataframe = calculate_ref_trn_length(dataframe)
    return dataframe


def remove_unknowns(dataframe):
    '''It removes rows that has not english letters'''
    pattern = re.compile(r'^[a-z ]+$')
    bools = []

    for row in dataframe.values:
        ref, trn = row[0], row[1]
        if re.fullmatch(pattern, ref) and re.fullmatch(pattern, trn):
            bools.append(True)
        else:
            bools.append(False)

    print(len(dataframe[bools]))
    return dataframe[bools]
