''' This module contains functions to preprocess text data
Especially: clean text from numbers/punctuations/spaces, lowering it
Additionally: tokenize, stem, remove stop words'''

import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')
stopwords_set = set(stopwords.words('english'))
ps = PorterStemmer()

''' Simple preprocessing without loose of meaning '''
def lower_text(text: str):
    return text.lower()

def remove_numbers(text: str):
    without_numbers = re.sub(r'\d+', ' ', text)
    return without_numbers

def remove_punctuation(text: str):
    without_punctuation = re.sub(r'[^a-z|\s]+', ' ', text)
    return without_punctuation

def remove_multiple_spaces(text: str):
    without_doublespace = re.sub('\s+', ' ', text).strip()
    return without_doublespace

def simple_text_preprocessing(text: str):
    return remove_multiple_spaces(remove_punctuation(remove_numbers(lower_text(text))))

def simple_row_preprocessing(row):
    row['translation'] = simple_text_preprocessing(row['translation'])
    row['reference'] = simple_text_preprocessing(row['reference'])
    return row

''' Hard part - with loosing of meaning '''
def tokenize_text(text: str):
    return word_tokenize(text)

def remove_stop_words(tokenized_text: list[str]):
    return [
        w for w in tokenized_text
        if w not in stopwords_set
    ]

def stem_words(tokenized_text: list[str]):
    return [
        ps.stem(w)
        for w in tokenized_text
    ]

def clean_data(sentence):
    preprocessed = simple_text_preprocessing(sentence)
    _tokenized = tokenize_text(preprocessed)
    _without_sw = remove_stop_words(_tokenized)
    _stemmed = stem_words(_without_sw)
    return _stemmed

# def clean_dataframe(df, filename='cleaned.csv', dir='../data/interim/', override=False):
#     file = os.path.join(dir, filename)

#     # Check cache
#     if (os.path.exists(file) and not override):
#         return pd.read_csv(file)
    
#     df['reference'] = df['reference'].apply(lambda s: clean_data(s))
#     df['translation'] = df['translation'].apply(lambda s: clean_data(s))

#     # Cache version
#     df.to_csv(file, index=False)

#     return df
