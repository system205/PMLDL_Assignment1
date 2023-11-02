'''Train simple hypothesis of swearing removal'''
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root")
parser.add_argument('--project_path')
args = parser.parse_args()

sys.path.append(args.project_path)

from src.data.make_dataset import load_train
from urllib.request import urlopen
from src.data.analysis.analyze import get_toxic_words

def train(root="../../"):
    '''Loads the train dataset and composes a list of toxic words combined with the ones known from the Internet'''
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
    
if __name__ == "__main__":
    if (args.root is None):
        print('You have to provide root path')
        exit(1)
        
    train(args.root)
    