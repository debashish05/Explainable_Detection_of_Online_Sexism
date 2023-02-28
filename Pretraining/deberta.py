# %%
import multiprocessing
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import transformers
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoConfig
from transformers import BertForMaskedLM, DistilBertForMaskedLM
from transformers import BertTokenizer, DistilBertTokenizer
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from tokenizers import BertWordPieceTokenizer
import os
import math
from transformers import DebertaTokenizer, DebertaModel,DebertaForMaskedLM
import glob

# %%
# HYPERPARAMS
SEED_SPLIT = 0
SEED_TRAIN = 0

MAX_SEQ_LEN = 64
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5 
LR_WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# %%
# load data
dtf_mlm1 = pd.read_csv('~/gab_1M_unlabelled.csv')
dtf_mlm2 = pd.read_csv('~/reddit_1M_unlabelled.csv')
dtf_mlm=dtf_mlm1.append(dtf_mlm2)
print(dtf_mlm)

# %%
texts=dtf_mlm['text'].tolist()

freq={}
for sentence in texts:
    num_text=len(sentence.split())
    if num_text not in freq:
        freq[num_text]=0
    freq[num_text]+=1

types = list(freq.keys())
frequency = list(freq.values())

fig = plt.figure(figsize = (10, 5))
plt.bar(types, frequency, color ='maroon',width = 0.4)
plt.xlabel("Number of words in sentence")
plt.ylabel("Frequency")
plt.title("Num of Words v/s Frequency")
plt.show()

# %%
# Train/Valid Split
df_train, df_valid = train_test_split(dtf_mlm, test_size=0.15, random_state=SEED_SPLIT)
len(df_train), len(df_valid)

# Convert to Dataset object
train_dataset = Dataset.from_pandas(df_train[['text']].dropna())
valid_dataset = Dataset.from_pandas(df_valid[['text']].dropna())

# %%
MODEL = 'deberta'
bert_type = 'microsoft/deberta-base'

TokenizerClass = DebertaTokenizer
ModelClass = DebertaForMaskedLM

# %%

tokenizer = TokenizerClass.from_pretrained(
            bert_type, use_fast=True, do_lower_case=False, max_len=MAX_SEQ_LEN
            )
model = ModelClass.from_pretrained(bert_type)

# %%
# modelpath="/kaggle/input/explainable-detection-of-online-sexism/dapt/kaggle/working/dapt"
# tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False, do_lower_case=True)
# model = AutoModelForMaskedLM.from_pretrained(modelpath)

# %%
def tokenize_function(row):
    return tokenizer(
        row['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_special_tokens_mask=True)
  
column_names = train_dataset.column_names

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=column_names,
)

valid_dataset = valid_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=column_names,
)

# %%
os.environ["WANDB_DISABLED"] = "true"

# %%
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


steps_per_epoch = int(len(train_dataset) / TRAIN_BATCH_SIZE)

training_args = TrainingArguments(
    output_dir='./deberta-pre',
    logging_dir='./LMlogs',             
    num_train_epochs=1,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    warmup_steps=LR_WARMUP_STEPS,
    save_steps=steps_per_epoch,
    save_total_limit=3,
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE, 
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='loss', 
    greater_is_better=False,
    seed=SEED_TRAIN
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)

path = './deberta-dpt-online-sexism'
trainer.train()
trainer.save_model(path) #save your custom model

# %%
#os.system(tar czf deberta-dpt-online-sexism.tar.gz ./deberta-dpt-online-sexism)

# %%
tokenizer = AutoTokenizer.from_pretrained(bert_type, use_fast = False, do_lower_case=True)
model = AutoModelForMaskedLM.from_pretrained(bert_type)

trainer = Trainer(
  model=model,
  data_collator=data_collator,
  eval_dataset=valid_dataset,
  tokenizer=tokenizer,
  )

eval_results = trainer.evaluate()

print('Evaluation results: ', eval_results)
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.3f}")
print('----------------\n')

# %%
for modelpath in glob.iglob(path):
    #modelpath=path
    print('Model: ', modelpath)
    tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False, do_lower_case=True)
    model = AutoModelForMaskedLM.from_pretrained(modelpath)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        #train_dataset=tokenized_dataset_2['train'],
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )
  
    eval_results = trainer.evaluate()

    print('Evaluation results: ', eval_results)
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.3f}")
    print('----------------\n')
