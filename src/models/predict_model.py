import argparse
import sys
import os
import requests, zipfile, os
import transformers
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
    
parent_dir = os.path.dirname(os.path.dirname(__file__))
print(parent_dir)
sys.path.append(parent_dir)

from data.text_preprocessing import simple_text_preprocessing

def download_weights(root='../../'):
    url = 'https://github.com/system205/PMLDL_Assignment1/releases/download/final-solution/trained_model.zip'

    response = requests.get(url, timeout=100000)
    if response.status_code == 200:
        with open(f"{root}data/external/weights.zip", "wb") as f:
            f.write(response.content)
    else:
        print("Failed to download the zip file.")

    with zipfile.ZipFile(f"{root}data/external/weights.zip", "r") as zip_ref:
        zip_ref.extractall(f'{root}models/')

    os.remove(f"{root}data/external/weights.zip")

def predict(root, texts: list[str]):
    if not os.path.exists(f"{root}output_dir/trained_model/pytorch_model.bin"):
        download_weights(root)
    
    texts = [simple_text_preprocessing(text) for text in texts]
    
    prefix = "paraphrase from toxic to neutral: "

    checkpoint = f'{root}models/trained_model'

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    for text in texts:
        t = prefix + text
        inputs = tokenizer(t, return_tensors="pt").input_ids
        transformers.set_seed(42)
        outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95, )

        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f'Translation: {translation} | Input: {text}')
