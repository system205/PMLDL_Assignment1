from collections import Counter

def get_toxic_words(dataframe, log=True) -> Counter:
    toxic_words = Counter()
    for sent in dataframe['reference']:
        toxic_words.update(sent.split(" "))

    translated_words = Counter()
    for sent in dataframe['translation']:
        translated_words.update(sent.split(" "))
        
    words_difference = set(toxic_words).difference(set(translated_words))
    
    if log:
        print(f'Number of words in toxic dataset: {len(toxic_words)}')
        print(f'Number of words in translated dataset: {len(translated_words)}')
        print(f'Number of toxic words that are not in translation: {len(words_difference)}')

    # Filter toxic counter removing words that are in translation
    for word in list(toxic_words):
        if (word not in words_difference): # not only toxic
            del toxic_words[word]
            
    return toxic_words, translated_words