import pandas as pd


def predict(dataframe_file, root='../../'):
    text_file = open(f"{root}/data/external/toxic_words.txt", "r")
    bad_words = text_file.read().split('\n')[:-1]
    text_file.close()
    
    print(f'Number of bad words to be filtered: {len(bad_words)}')
    
    test_dataframe = pd.read_csv(dataframe_file)
    
    translations = []
    for ref in test_dataframe['reference']:
        translations.append('')
        for word in ref.split(' '):
            if word not in bad_words: 
                translations[-1] += word + " "
                
    
    test_dataframe['predictions'] = translations
    test_dataframe.to_csv(dataframe_file, index=False)
    
    print(pd.read_csv(dataframe_file))
    
    
    
predict( '../data/interim/val.csv', '../')