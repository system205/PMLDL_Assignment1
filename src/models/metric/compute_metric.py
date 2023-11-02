'''Contains the evaluation function on dataframe'''
import evaluate
import pandas as pd
import argparse


def evaluate_dataframe(dataframe_file, predictions_column='predictions', separator=','):
    '''Using sacreBLEU metric computes the score of
    prediction and reference columns in dataframe'''
    
    metric = evaluate.load("sacrebleu")
    dataframe = pd.read_csv(dataframe_file, sep=separator)
    references = list(dataframe['reference'])
    translations = list(dataframe['translation'])
    predictions = list(dataframe[predictions_column])
    
    print('\n(Translation (target), Prediction (output), Reference (input))', *zip(translations[-5:], predictions[-5:], references[-5:]), sep='\n')
    
    score = metric.compute(predictions=predictions, references=translations)
    print(f'Computed sacrebleu: {score}')
    print(f'Score: {score["score"]:.2f}%')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataframe_file")
    parser.add_argument("--predictions_column")
    parser.add_argument("--separator")
    args = parser.parse_args()

    if (args.dataframe_file is None): 
        print("You have to specify dataframe_file")
        exit(1)
        
    if (args.predictions_column is not None): 
        evaluate_dataframe(args.dataframe_file, args.predictions_column, separator=args.separator)
    else:
        evaluate_dataframe(args.dataframe_file, separator=args.separator)
