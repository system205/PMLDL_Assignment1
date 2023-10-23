import evaluate
import pandas as pd
import argparse


def evaluate_dataframe(dataframe_file):
    metric = evaluate.load("sacrebleu")
    dataframe = pd.read_csv(dataframe_file)
    translations = list(dataframe['translation'])
    predictions = list(dataframe['predictions'])
    
    print('\n(Translation, Prediction)', *zip(translations[-5:], predictions[-5:]), sep='\n')
    
    score = metric.compute(predictions=predictions, references=translations)
    print(f'Computed sacrebleu: {score}')
    print(f'Score: {score["score"]:.2f}%')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataframe_file")
    args = parser.parse_args()

    if (args.dataframe_file is None): 
        print("You have to specify dataframe_file")
        exit(1)

    evaluate_dataframe(args.dataframe_file)
