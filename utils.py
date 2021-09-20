import itertools
import json
import linecache
import os
import pickle
import warnings
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import git
import numpy as np
import torch
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler

from transformers import BartTokenizer

import sys
import os
import torch
import re
import string

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask]) 
    
    
class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        option,
        self_ref=True,
        type_path="train",
        sample=100,
    ):
        super().__init__()
        sys.stderr.write('Reading corpora...')  
        self.type_path = type_path
        self.ignore_sense_id = True
        if type_path.startswith('test') and 'wiki' in data_dir and 'wiki_full' not in data_dir and 'japanese' not in data_dir:
            self.ignore_duplicates = True
        else:
            self.ignore_duplicates = False
        self.train_src = 'both'
        self.option = option
        self.data_dir = data_dir
        self.self_ref = self_ref
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id

        self.sample = ""

        self.data, self.data_ntoken, self.ref_data = self.tokenize((data_dir+'{}.txt'.format(type_path.split('_')[-1])), self.ignore_sense_id, self.train_src)
        if (option.startswith('t5_specific') or option.startswith('forward') or option.startswith('t5_general') ) and (type_path=='test' or type_path=='test_val'):
            self.data_word_egs = self.read_beams((data_dir+'{}.forward'.format(type_path.split('_')[-1])), self.ignore_sense_id)
            self.data = self.add_beams(self.data, self.data_word_egs)                 
        else:
            self.data_word_egs = self.read_examples((data_dir+'{}.eg'.format(type_path.split('_')[-1])), self.ignore_sense_id)
            self.data = self.add_examples(self.data, self.data_word_egs)

        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.target_word = []        
        self.encode(self.data, self.ignore_duplicates, option, data_dir)
        print(option)
       
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:

        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
        
        word = self.target_word[index]
        
        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask, "target_word": word}   

    def tokenize(self, path, ignore_sense_id, train_src):
        word_desc = []  # [(srcWord0, [trgId0, trgId1, ...]), (srcWord1, [trgId0, trgId1, ...])]
        word_desc_orig = [] # [(srcWord0, [trgWord0, trgWord1, ...]), ... ]
        self.self_refs = []
            
        ntoken = 0
        ref = {}
        if train_src not in set(['wordnet', 'gcide', 'both']):
            sys.stderr.write("train_src has to be one of {'wordnet' | 'gcide' | 'both'}\n")
            exit()
        with open(path, 'r', encoding='utf-8') as f:
            for index,line in enumerate(f):
             
                elems = line.strip().split('\t')
                if train_src in set(['wordnet', 'gcide']) and train_src != elems[2]:
                    continue

                word = elems[0]
                word_wo_id = word.split('%', 1)[0]
                word_wo_id = word_wo_id.replace('_',' ')
                if ignore_sense_id:
                    word = word_wo_id
                if word_wo_id not in ref:
                    ref[word_wo_id] = []
                # do not add references containing target word  
                if self.self_ref==False and word_wo_id in elems[3].split() and self.type_path.split('_')[-1]!='test':
                    self.self_refs.append(index)     
                    continue

                ref[word_wo_id].append(elems[3])
                description = elems[3].split()
                word_desc_orig.append((word,description))

                ntoken += (len(description) - 1)  # including <eos>, not including <bos>
        
        return word_desc_orig, ntoken, ref
    
    def read_examples(self, path, ignore_sense_id):
        assert os.path.exists(path)
        word_egs = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.self_ref==False and i in self.self_refs:
                    continue                
                word, eg = line.strip().split('\t')
                word_lemma = word.split('%', 1)[0]
                if ignore_sense_id:
                    word = word_lemma
                word_egs.append((word, eg.split(' ')))
        return word_egs

    def add_examples(self, word_char_vec_desc, word_egs):
        word_char_vec_desc_eg = []
        for i, (word, desc) in enumerate(word_char_vec_desc):
            eg_id = []
            for w in word_egs[i][1]:
                eg_id.append(w)
         
            word_char_vec_desc_eg.append((word, desc, eg_id))
        return word_char_vec_desc_eg  
 
    def read_beams(self, path, ignore_sense_id):
        assert os.path.exists(path)
        word_egs = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                eg = line.strip()
                word_egs.append(eg.split(' '))
        return word_egs

    def add_beams(self, word_char_vec_desc, word_egs):
        contexts = self.read_examples((self.data_dir+'{}.eg'.format(self.type_path.split('_')[-1])), self.ignore_sense_id)
        
        unique = []
        if self.ignore_duplicates:
            words = set()
            for i in range(len(word_char_vec_desc)):
                word = word_char_vec_desc[i][0].split('%', 1)[0]
                if word not in words:
                    unique.append(i)
                    words.add(word)
                                      
            word_char_vec_desc =  [line for index,line in enumerate(word_char_vec_desc) if index in unique]
            contexts = [line for index,line in enumerate(contexts) if index in unique]
        else:
            contexts = [line for index,line in enumerate(contexts) if index not in self.self_refs]

        beam = len(word_egs) // len(contexts)
        print(beam, len(word_egs), len(contexts))
        word_char_vec_desc_eg = []
        for i, egs in enumerate(word_egs):
            pred_def = []
            for w in egs:
                pred_def.append(w)
            word_char_vec_desc_eg.append((word_char_vec_desc[int(i/beam)][0], pred_def, contexts[int(i/beam)][1]))
        return word_char_vec_desc_eg  

  
    def encode(self, data, ignore_duplicates, option, data_dir):
        data_ = []
        data_ = data        
        sememe = ""

        leng = len(data_)
        for index, (word, definition, example) in enumerate(data_):
            if self.sample:
                if index>self.sample:
                    break  
            word_ = "word: " + word
            context = " context: "+" ".join(example).replace('<TRG>',word)
            sememe = "　"
            
            if option:
                if option == "t5_general":
                    context = " "
                elif option == "t5_specific":
                    sememe = "" 
                    word_ = ""
                    context =  " definition: "+" ".join(definition)   # input is definition 
                    definition = [" ".join(example).replace('<TRG>',word)]  # output is context  
                elif option.startswith("forward"):  
                    sememe = " "
                else:
                    sememe = ' {}: '.format(option) + " ".join(sems[index]) if (sems[index][0]!='') else ""

                
            source = word_ + sememe + context  
            target = " ".join(definition) 
            src = self.tokenizer.batch_encode_plus(
                  [source], max_length=self.max_source_length, pad_to_max_length=True, truncation=True, return_tensors="pt"
              )
            trg = self.tokenizer.batch_encode_plus(
                  [target], max_length=self.max_source_length, pad_to_max_length=True, truncation=True, return_tensors="pt"
              )            
            
            e = self.tokenizer.encode(word)[:-1]
            if len(e)<1:
                e = [2]
            self.inputs.append(src)
            self.targets.append(trg)    
            self.target_word.append(e)  # remove eos token
                 
        print(len(self.inputs))           

            
    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        
        pad_token_id = self.pad_token_id
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)        
        y = trim_batch(target_ids, pad_token_id)
        
        words = [x["target_word"] for x in batch]

        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
            "target_word": words
        }
        return batch
    
logger = getLogger(__name__)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


# def save_git_info(folder_path: str) -> None:
#     """Save git information to output_dir/git_log.json"""
#     repo_infos = get_git_info()
#     save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path):
    with open(path, "w") as f:
        json.dump(content, f, indent=4)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# def get_git_info():
#     repo = git.Repo(search_parent_directories=True)
#     repo_infos = {
#         "repo_id": str(repo),
#         "repo_sha": str(repo.head.object.hexsha),
#         "repo_branch": str(repo.active_branch),
#     }
#     return repo_infos


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "bleu"]


def calculate_rouge(output_lns: List[str], reference_lns: List[str], use_stemmer=True) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}


def freeze_params(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"
