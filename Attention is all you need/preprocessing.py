# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 12:51:55 2025

@author: advit
"""
from datasets import load_dataset
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,normalizers

dataset=load_dataset("wmt14","de-en",split="train[:1%]").select(range(15000))
print(dataset[0]['translation'])

english_sentences=[example['translation']['en'] for example in dataset]
de_sentences=[example['translation']['de'] for example in dataset]

all_sentences=english_sentences+de_sentences

tokenizer=Tokenizer(models.BPE())
tokenizer.normalizer=normalizers.NFKC()
tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()

special_tokens=["<pad>","<unk>","<s>","</s>"]

trainer=trainers.BpeTrainer(vocab_size=37000,special_tokens=special_tokens)
tokenizer.train_from_iterator(all_sentences,trainer=trainer)

tokenizer.save("bpe_en_de_tokenizer.json")
print("tokenizer saved successfully")

from tokenizers import Tokenizer
from tqdm import tqdm

tokenizer=Tokenizer.from_file("bpe_en_de_tokenizer.json")

tokenizer_pairs=[]
for en,de in tqdm(zip(english_sentences,de_sentences),total=len(english_sentences)):
    src=tokenizer.encode(en).ids
    tgt=tokenizer.encode(de).ids
    tokenizer_pairs.append((src,tgt))
    
with open("tokenized_en_de.txt", "w", encoding="utf-8") as f:
    for src, tgt in tokenizer_pairs:
        src_line = " ".join(map(str, src))
        tgt_line = " ".join(map(str, tgt))
        f.write(f"{src_line}\t{tgt_line}\n")