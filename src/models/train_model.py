import argparse
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
import evaluate
import numpy as np
import pandas as pd

PREFIX = "paraphrase from toxic to neutral: "

CHECKPOINT = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=CHECKPOINT)
metric = evaluate.load("sacrebleu")

def preprocess_function(examples):
    inputs = [PREFIX + example for example in examples["reference"]]
    targets = examples["translation"]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def train(train_dataframe, valid_dataframe, save_dir="output_dir/"):
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        # fp16=True,
        report_to='tensorboard',
    )

    train_dataset = Dataset.from_pandas(train_dataframe).map(preprocess_function, batched=True)
    val_dataset = Dataset.from_pandas(valid_dataframe).map(preprocess_function, batched=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(f"{save_dir}trained_model")
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path")
    parser.add_argument("--val_path")
    parser.add_argument("--save_dir")

    args = parser.parse_args()
    
    if(args.train_path is None or args.val_path is None):
        print('You have to specify train_path and val_path')
        exit(1)
        
        
    train_dataframe = pd.read_csv(args.train_path)
    val_dataframe = pd.read_csv(args.val_path)
    
    if (args.save_dir is None):
        train(train_dataframe, val_dataframe)
    else: train(train_dataframe, val_dataframe, args.save_dir)