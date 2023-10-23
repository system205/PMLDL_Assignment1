import sys
sys.path.append('/workspaces/PMLDL_Assignment1') # TODO get project source path
from src.data.make_dataset import load_train
from urllib.request import urlopen
from src.data.analysis.analyze import get_toxic_words

def train(root="../../"):
    train_dataset = load_train(root)
    toxic_counter = get_toxic_words(train_dataset, log=False)[0]
    
    # List of bad words
    response = urlopen('https://raw.githubusercontent.com/snguyenthanh/better_profanity/master/better_profanity/profanity_wordlist.txt')
    data = str(response.read()).split('\\n')[1:-1]
    
    all_toxic_words = list(set(toxic_counter).union(set(data)))
    print(f'Total number of toxic words to be filtered: {len(all_toxic_words)}')
    
    # Save for evaluation
    file = open(f'{root}data/external/toxic_words.txt','w+',encoding='UTF-8')
    for word in all_toxic_words:
        file.write(word+'\n')
    file.close()


train('../') # TODO specify root
    