import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import string
import random
from nltk.translate import bleu_score
from calculate_scores import tokenize 
import argparse

def standardization(l, std):
    if std==False:
        return l
    l_mean = statistics.mean(l)
    l_stdev = statistics.stdev(l)
    try:
        r = [(i - l_mean) / l_stdev for i in l]
    except:
        r = [i for i in l]
    return r

def get_duplicate_data(dataset):
    indexs = []
    words = set()
    for i,data in enumerate(dataset):
        word = data[0]
        if word not in words:
            indexs.append(data)
            words.add(word)
    return indexs    

def read_scores(file_path):
    try:
        return [float(line.strip()) for line in open(file_path, encoding='utf-8').readlines()]  
    except:
        return 

def get_word_desc(data_dir, type_path):
    word_desc, _, _ = tokenize('{}{}.txt'.format(data_dir, type_path), '{}{}.eg'.format(data_dir,type_path), 
    ignore_sense_id=True, one2one=False)
    return word_desc

def read_result(beam_sz, data_dir, word_desc, type_path, scores_dict, metrics_dict, std, rm_err):   
    bad_defs_expand = []

    for key in scores_dict: # read scores from the model
        scores_dict[key] = standardization(read_scores(scores_dict[key]), std)
    
    for key in metrics_dict:  # read evaluation metrics
        metrics_dict[key] = read_scores(metrics_dict[key])

#     if 'wiki' in data_dir and 'wiki_full' not in data_dir and 'japanese' not in data_dir: # delete duplicate words
#         word_desc = get_duplicate_data(word_desc) 
        
#     if 'slang' in data_dir and rm_err:
#         bad_defs_expand = rm_error(word_desc, beam_sz)

    columns_dict = {'word':[item[0] for item in word_desc for i in range(beam_sz)]}
    columns_dict.update({'pred':[line.strip() for line in open(data_dir+'{}.forward'.format(type_path), encoding='utf-8').readlines()]})    
    columns_dict.update(scores_dict)
    columns_dict.update(metrics_dict)
    
#     for i in columns_dict:
#         print(i, len(columns_dict[i]))
    return pd.DataFrame(columns_dict).drop(bad_defs_expand).reset_index()

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_grids(fix, step, l=[], option=""):
    if l:
        return l
    grids = []
    for forw_w in range(fix[0], fix[1]): 
        x = np.arange(0, 1-forw_w*step+step, step)
        if len(option): x = [0]
        for genr_w in x:
            if option=="general":
                grids.append([forw_w*step, 0.0, 1-forw_w*step])
            elif option=="specific":
                grids.append([forw_w*step, 1-forw_w*step, 0.0])
            else:
                grids.append([forw_w*step, (1-genr_w-forw_w*step), genr_w])
    return grids

def calculate_weight_loss(beams, beam_sz, grids, score_names, score_type='bleu', drop=[]):
    max_bleu = -1
    for forw_w, spec_w, genr_w in tqdm(grids):
        bleus = []
        sents = []
        l = beams[score_names[0]]*forw_w + beams[score_names[1]]*spec_w + beams[score_names[2]]*genr_w

        for k in range(int(len(beams)/beam_sz)):
            if k in drop: continue
            min_idx = l[k*beam_sz:(k+1)*beam_sz].idxmin()
            bleus.append(beams.iloc[min_idx][score_type])
            sents.append(beams.iloc[min_idx]['pred'])
        if np.mean(bleus) > max_bleu:
            max_bleu = np.mean(bleus)
            max_combination = [forw_w, spec_w, genr_w]
#         print('{}:{:.2f}\t {}:{:.2f}\t {}:{:.2f}\t{}:{:.4f}'.format(
#             score_names[0], forw_w, score_names[1], spec_w, score_names[2], genr_w, score_type, np.mean(bleus)))
    beams100 = [pd.DataFrame(y).reset_index(drop=True) for x, y in beams.groupby(beams.index // beam_sz)]
    return max_bleu, max_combination, beams100, sents, bleus

def calculate_weight_loss_by_words(beams, beam_sz, grids, score_names, score_type='bleu', drop=[], option=""):
    max_bleu = -1
    for forw_w, spec_w, genr_w in tqdm(grids):
        scores = []
        sents = dict()
        sent_dict = []
        l = beams[score_names[0]]*forw_w + beams[score_names[1]]*spec_w + beams[score_names[2]]*genr_w

        for k in range(int(len(beams)/beam_sz)):
            if k in drop: continue
            min_idx = l[k*beam_sz:(k+1)*beam_sz].idxmin()
            if beams.iloc[k*beam_sz]['word'] not in sents:
                sents[beams.iloc[k*beam_sz]['word']] = []
            sents[beams.iloc[k*beam_sz]['word']].append(beams.iloc[min_idx][score_type])
            sent_dict.append([beams.iloc[min_idx]['word'],beams.iloc[min_idx]['pred'], beams.iloc[min_idx][score_type]])
        for i in sents:
            scores.append(np.mean(sents[i]))
    return np.mean(scores), scores, sent_dict, sents 

def get_paths_proposed(dataset_name, f_dir, g_dir, s_dir, type_path, score_type, score_dir):
    
    floss_path = f_dir+'{}_losses.txt'.format(type_path)
    wloss_path = g_dir+'{}_losses.txt'.format(type_path)
    closs_path = s_dir+'{}_losses.txt'.format(type_path)
    
    score_path = score_dir+'{}_{}_{}.txt'.format(dataset_name, type_path, score_type)
    scores_dict = {'forw_scor':floss_path, 'genr_scor':wloss_path, 'spec_scor':closs_path}
    metrics_dict = {score_type: score_path}
    
    return scores_dict, metrics_dict  

def main(args):

    score_names = ['forw_scor', 'spec_scor', 'genr_scor']
    grids = get_grids(fix=[0, 11] , step=0.1, l=[], option=args.option)

    scores_dict, metrics_dict = get_paths_proposed(args.dataset_name, args.f_dir, args.g_dir, args.s_dir, 'val',
                                                   args.score_type, score_dir=args.evaluation_score_path)
    word_desc = get_word_desc(args.data_dir, 'val')

    beams = read_result(beam_sz=args.beam_sz, data_dir=args.data_dir, word_desc=word_desc, 
                        type_path='val', scores_dict=scores_dict, metrics_dict=metrics_dict, std=True, rm_err=args.rm_err)
    
    max_score, max_combination, _, predictions, _ = calculate_weight_loss(
        beams, beam_sz=args.beam_sz, grids=grids, score_names=score_names, score_type=args.score_type, drop=[])

    scores_dict, metrics_dict = get_paths_proposed(args.dataset_name, args.f_dir, args.g_dir, args.s_dir, 'test',
                                                   args.score_type, score_dir=args.evaluation_score_path)
    word_desc = get_word_desc(args.data_dir, 'test')
    
    beams = read_result(beam_sz=args.beam_sz, data_dir=args.data_dir, word_desc=word_desc,
                        type_path='test', scores_dict=scores_dict, metrics_dict=metrics_dict, std=True, rm_err=args.rm_err)

    if 'wiki' in args.dataset_name:
        max_score, _, _, predictions  = calculate_weight_loss_by_words(
                beams, beam_sz=args.beam_sz, grids=[max_combination], score_names=score_names,
            score_type=args.score_type, drop=[], option="")
    else:
        max_score, max_combination, beams100, sents, scores = calculate_weight_loss(
        beams, beam_sz=args.beam_sz, grids=[max_combination], score_names=score_names, score_type=args.score_type, drop=[])
    
    with open('{}{}_{}_best_predictions.txt'.format(args.output_dir, args.dataset_name, args.score_type), 'w') as f:
        f.write("The {} score of dataset {} is {}\n".format(args.score_type, args.dataset_name, max_score))
        for pred in predictions:
            f.write(pred + '\n')
#     print(np.mean([line.iloc[0][args.score_type] for line in beams100]))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
                "--dataset_name",
                type=str,
                default = 'wordnet',
                help="dataset name",
            )     
    parser.add_argument(
                "--data_dir",
                type=str,
                default = 'cnn_tiny/',
                help="dataset directory",
            ) 
    parser.add_argument(
                "--pred_dir",
                type=str,
                default = '',
                help="",
            ) 
    parser.add_argument(
                "--f_dir",
                type=str,
                default = '',
                help="",
            )  
    parser.add_argument(
                "--g_dir",
                type=str,
                default = '',
                help="",
            )   
    parser.add_argument(
                "--s_dir",
                type=str,
                default = '',
                help="",
            ) 
    parser.add_argument(
                "--evaluation_score_path",
                type=str,
                default = '',
                help="",
            )  
    parser.add_argument(
                "--output_dir",
                type=str,
                default = 'bart_utest_output/result.txt',    
                help="",
            ) 
    parser.add_argument(
                "--rm_err",
                default= False,
                action="store_true",
                help="remove error items in slang dataset",
            )  
    parser.add_argument(
                "--ignore_sense_id",
                default= False,
                action="store_true",
                help="word%oxford.2 ignore symbols after % by default",
            )
    parser.add_argument(
                "--score_type",
                type=str,
                default= 'nist',
                help="nltk sentence bleu (nltk) or or nist or moverscore or mose bleu (mose)",
            )
    parser.add_argument(
                "--option",
                type=str,
                default= '',
                help="for ablation study (only use two re-ranking scores), general or specific",
            )    
    parser.add_argument(
                "--beam_sz",
                type=int,
                default = 100,    
                help="beam size",
            )
    
    args = parser.parse_args()
    main(args)