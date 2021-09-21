# os.environ['MOVERSCORE_MODEL'] = "bandainamco-mirai/distilbert-base-japanese"
# !pip install moverscore
# !pip uninstall moverscore --yes
# !pip install https://github.com/mymusise/emnlp19-moverscore/archive/master.zip 
# !conda install pyemd --yes
# !pip install mecab-python3
# !pip install unidic-lite

# import MeCab
# tagger = MeCab.Tagger('-Owakati')

import os 
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import random
from nltk.translate import bleu_score, nist_score
# from moverscore_v2 import get_idf_dict, word_mover_score
from typing import List, Union, Iterable
import re
import argparse
from subprocess import Popen, PIPE
import glob
from collections import defaultdict
from shutil import rmtree

def tokenize(data_dir, eg_path, ignore_sense_id, one2one):
    word_desc_orig = [] # [(srcWord0, [trgWord0, trgWord1, ...],[Example]), ... ]
    refs = {}
    egs = []
    if one2one:
        refs = []
    with open(data_dir, 'r', encoding='utf-8') as f1, open(eg_path, 'r', encoding='utf-8') as f2:
        for index, line in enumerate(zip(f1,f2)):               
            elems = line[0].strip().split('\t')
            word = elems[0]
            _,eg = line[1].strip().split('\t')
            word_wo_id = word.split('%', 1)[0]
            word_wo_id = word_wo_id.replace('_',' ')
            if ignore_sense_id:
                word = word_wo_id
            if word_wo_id not in refs and not one2one:
                refs[word_wo_id] = []
         
            if not one2one:
                refs[word_wo_id].append(elems[3])
            else:
                refs.append([word,elems[3]])
            egs.append(eg.split(' '))
                
            description = elems[3].split()
            word_desc_orig.append((word, description, eg))

    return word_desc_orig, refs, egs

def read_outputfile(dataset, path, beam_sz=100):   
    test_predictions = []
    with open('{}'.format(path),'r',encoding='utf-8') as f:
        for i,line in enumerate(f.readlines()):
            sent = line.rstrip().replace('""',"")
            test_predictions.append([dataset[int(i/beam_sz)][0],sent])
    return test_predictions

def tokenize_jp(references, test_predictions):
    for word in references:
        tmp = []
        references[word] = list(dict.fromkeys(references[word]))
        for s in references[word]:
            tmp.extend([" ".join(tagger.parse(s).split())])
        references[word] = tmp

    for index, (word,pred) in enumerate(test_predictions):
        test_predictions[index][1] = " ".join(tagger.parse(pred).split())
    
    return references, test_predictions

def sentence_score(hypothesis: str, references: List[str], trace=0):

    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
    sentence_score = np.mean(scores)
#     if trace > 0:
#         print(hypothesis, references, sentence_score)  
    return sentence_score

def get_rid_of_period(l):
    pattern = re.compile("\.(?!\d)")
    return [pattern.sub('', sent) for sent in l]

def bleu_(hyp, ref, mode, beam_sz=100, one2one=False, tmp_dir=''):
    bleus = []
    num_hyp = 0
    with open(os.devnull, 'w') as devnull:    
        for i,(word,desc) in enumerate(hyp):
            word_wo_id = word.split('%', 1)[0].replace('_', ' ')
            desc = get_rid_of_period([desc])
            if one2one:
                refs = list(dict.fromkeys([ref[i][-1]])) 
            else:
                refs = list(dict.fromkeys(ref[word_wo_id])) 

            # compute sentence bleu
            if mode == 'nltk': #  3~5 point lower than sentence_bleu.cpp
                auto_reweigh = False if len(desc[0].split()) == 0 else True
     
                ref_list = [ref.split() for ref in get_rid_of_period(refs)]
                bleu = bleu_score.sentence_bleu(ref_list, desc[0].split(),smoothing_function=bleu_score.SmoothingFunction().method2,auto_reweigh=auto_reweigh)

            elif mode == 'nist':
                auto_reweigh = False if len(desc[0].split()) == 0 else True

                ref_list = [ref.split() for ref in get_rid_of_period(refs)]

                pred = desc[0].split()
                n = 5            
                pred_len = len(pred)
                if pred_len < 5:
                    n = pred_len
                try:
                    bleu = nist_score.sentence_nist(ref_list, pred, n=n)
                except:
                    bleu = 0
                    
            elif mode == 'moverscore':
                ref_list = [ref for ref in get_rid_of_period(refs)]

                try:
                    bleu = sentence_score(desc[0], ref_list)
                except:
                    bleu = 0

            else:
                ref_paths = []                   
                refs = [ref for ref in get_rid_of_period(refs) if len(ref.split())>0]
                
                for j, ref_ in enumerate(refs):
                    ref_path = tmp_dir+'ref/' + str(j)
                    with open(ref_path, 'w',encoding='utf-8') as f:
                        f.write(ref_ + '\n')
                        ref_paths.append(ref_path)

                # write a hyp to tmp file
                with open(tmp_dir+'hyp', 'w',encoding='utf-8') as f:
                    f.write(desc[0] + '\n')

                rp = Popen(['cat', tmp_dir+'hyp'], stdout=PIPE)
                bp = Popen(['./sentence-bleu'] + ref_paths, stdin=rp.stdout, stdout=PIPE, stderr=devnull)
                out, err = bp.communicate()
                bleu = float(out.strip())
            num_hyp += 1
            bleus.append(bleu)

        return bleus

def cal_bleu_for_beams(data_dir, type_path, pred_dir, output_dir, beam_sz, mode, one2one, c_range, tmp_dir):
    if c_range[1]-c_range[0]<1.0:
        mose_path = "_tmp{}".format(int(c_range[0]*10))
    else:
        mose_path = ""
        
    word_desc, references, examples = tokenize(
    '{}.txt'.format(data_dir+type_path), '{}.eg'.format(data_dir+type_path), ignore_sense_id=True, one2one=one2one)
    test_predictions = read_outputfile(word_desc, '{}.forward'.format(data_dir+type_path), beam_sz)   
    
    if 'wiki' in data_dir and 'japanese' not in data_dir:
        unique = get_duplicate_idx(word_desc)
        word_desc = [line for i,line in enumerate(word_desc) if i in unique]  
        test_predictions = read_outputfile(word_desc, '{}.forward'.format(data_dir+type_path), beam_sz)   
    if 'japanese' in data_dir:
        references, test_predictions = tokenize_jp(references, test_predictions)
        
    scores = []
    for i in tqdm(range(int(len(test_predictions)/beam_sz*c_range[0]),int(len(test_predictions)/beam_sz*c_range[1]))):
        if one2one:
            ref_list = [references[i] for j in range(beam_sz)]
        else:
            ref_list = references    
        
        scores += bleu_(test_predictions[beam_sz*i:(i+1)*beam_sz], ref_list, mode=mode, one2one=one2one, tmp_dir=tmp_dir)

    with open('{}_{}_{}{}.txt'.format(output_dir, type_path, mode, mose_path), 'w', encoding='utf-8') as f:
        for i in scores:
            f.write(str(i)+'\n')
    return scores

def main(args):
    c_range = [float(item) for item in args.c_range.split(',')]

    if args.mode=="mose":
        os.makedirs(args.tmp_dir, exist_ok=True)
        os.makedirs(args.tmp_dir+'ref', exist_ok=True)

    bleus = cal_bleu_for_beams(
        data_dir=args.data_dir, type_path=args.type_path, pred_dir=args.pred_dir, output_dir=args.output_dir, beam_sz=args.beam_sz,mode=args.mode, one2one=args.one2one, c_range=c_range,tmp_dir=args.tmp_dir)    
    
    if args.mode=="mose":
          rmtree(args.tmp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
                "--data_dir",
                type=str,
                default = 'cnn_tiny/',
                help="dataset directory",
            ) 
    parser.add_argument(
                "--pred_dir",
                type=str,
                default = 'bart_utest_output/test_predictions',
                help="",
            )
    parser.add_argument(
                "--output_dir",
                type=str,
                default = 'bart_utest_output/result.txt',    
                help="",
            )
    parser.add_argument(
                "--tmp_dir",
                type=str,
                default = 'c1/',    
                help="store tmp result when calculating mose",
            )    
    parser.add_argument(
                "--one2one",
                default= False,
                action="store_true",
                help="default evaluation is one pred to many refs",
            )  
    parser.add_argument(
                "--ignore_sense_id",
                default= False,
                action="store_true",
                help="word%oxford.2 ignore symbols after % by default",
            )
    parser.add_argument(
                "--mode",
                type=str,
                default= 'nist',
                help="nltk sentence bleu (nltk) or or nist or moverscore or mose bleu (mose)",
            )
    parser.add_argument(
                "--beam_sz",
                type=int,
                default = 100,    
                help="beam size",
            )
    parser.add_argument(
                "--c_range",
                type=str,
                default = "0,1",    
                help="calculate store from range a to b eg. 0.1,0.5",
            )   
    parser.add_argument(
                "--type_path",
                type=str,
                default = "test",    
                help="val or test",
            )   
    args = parser.parse_args()
    main(args)
