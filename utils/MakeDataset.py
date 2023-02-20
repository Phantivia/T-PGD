import torch
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
def make(task = 'sst2', tokenizer_checkpoint = 'bert-base-uncased', local_path = None):
    task_info = {
        'sst2':{
            'num_labels':2,
            'text_key':'sentence',
            'tokenize_keys': ('sentence', None)
        },
        'imdb':{
            'num_labels':2,
            'text_key':'text',
            'tokenize_keys': ('text', None)
        },
        'ag_news':{
            'num_labels':4,
            'text_key':'text',
            'tokenize_keys': ('text', None)
        },
        'mnli_hypothesis':{
            'num_labels':3,
            'text_key':'hypothesis',
            'tokenize_keys': ('premise', 'hypothesis')
        },
        'mnli_premise':{
            'num_labels':3,
            'text_key':'premise',
            'tokenize_keys': ('premise', 'hypothesis')
        },
        'yelp':{
            'num_labels':2,
            'text_key':'text',
            'tokenize_keys': ('text', None)
        },
        'sst2_train':{
            'num_labels':2,
            'text_key':'sentence',
            'tokenize_keys': ('sentence', None)
        },
    }
    

    if local_path is not None:
        dataset = load_from_disk(local_path)
    else:
        dataset = load_dataset(task)

    info = task_info[task]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, use_fast=True, mirror = 'tuna')
    
    stop_words = ['not', 'not', "'s", 's', 'doesn', "doesn't", 'no', 'was', 'wasn', "wasn't", 'without',  'could', 'couldn', "couldn't",
                'is', 'isn', "isn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'were', 'weren', "weren't", 'and', 'but', 'can', 'cannot',]
    
    filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
    if task.startswith('mnli'):
        for word in filter_words:
            stop_words.append(word)
    
    DLC_words = []
    for stop_word in stop_words:
        DLC_words.append(stop_word)
        DLC_words.append(stop_word.capitalize())
        if tokenizer_checkpoint.startswith('roberta'):
            DLC_words.append("Ġ" + stop_word)
            DLC_words.append("Ġ" + stop_word.capitalize())
    
    stop_words = DLC_words
    stop_words = " ".join(stop_words)
    stop_ids = list(set(list(tokenizer(stop_words)['input_ids'])))
    print("Stop words: ", stop_words)
    print("Stop ids: ", stop_ids)
    sentence1_key, sentence2_key = info['tokenize_keys']
    def preprocess_function(examples):
        if task.startswith('mnli'):
            encoded_premise = tokenizer(examples[sentence1_key], truncation=True)
            premise_length = len(encoded_premise['input_ids']) - 2
            encoded_hypothesis = tokenizer(examples[sentence2_key], truncation=True)
            hypothesis_length = len(encoded_hypothesis['input_ids']) - 2

            encode = tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
            encode['mask'] = [0] * len(encode['input_ids'])

            if task == 'mnli_hypothesis':
                encode['mask'][-hypothesis_length-1:-1] = [1] * hypothesis_length
            else:
                encode['mask'][1:premise_length+1] = [1] * premise_length

            encode['adv_start'] = list(encode['mask']).index(1)
            encode['adv_end'] = encode['adv_start'] + np.array(encode['mask']).sum()

        else:
            encode = tokenizer(examples[sentence1_key], truncation=True)
            encode['mask'] = [1] * len(encode['input_ids'])
            encode['mask'][0], encode['mask'][-1] = 0, 0
            encode['adv_start'] = 1
            encode['adv_end'] = -1
        # Figure out stop words we want to avoid to perturb
        for index in range(len(encode['mask'])):
            if encode['input_ids'][index] in stop_ids:
                encode['mask'][index] = 0

        perturb_positions = []
        for index in range(len(encode['mask'])):
            if encode['mask'][index] == 1:
                perturb_positions.append(index)
        encode['perturb_positions'] = perturb_positions
        return encode

    
    encoded_dataset = dataset.map(preprocess_function)
    encoded_dataset.set_format("numpy")

    return dataset, encoded_dataset, info['num_labels'], info['text_key'], stop_ids

    
