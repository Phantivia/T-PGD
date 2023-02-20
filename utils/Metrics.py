import tensorflow as tf
import tensorflow_hub as hub

from strsimpy.levenshtein import Levenshtein
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

import language_tool_python
from bert_score import BERTScorer
import transformers
import math
import os

use_path = '/home/phantivia/universal-sentence-encoder_4'

class USE:
    def __init__(self, cuda_device):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
        # Restrict TensorFlow to only use the victim model's GPU
            try:
                tf.config.experimental.set_visible_devices(gpus[cuda_device], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[cuda_device], True)
            except RuntimeError as e:
                # Visible devices must be set at program startup
                print(e)
        with tf.device('/GPU:' + str(cuda_device)):
            path = use_path
            print('Trying to fetch USE model from '+ path)
            self.embed = hub.load(path)
            print("Successfully fetched USE model.")

    def __call__(self, sentence1, sentence2):
        sentence1, sentence2 = sentence1.lower(), sentence2.lower()
        embeddings = self.embed([sentence1, sentence2])

        vector1 = tf.reshape(embeddings[0], [512, 1])
        vector2 = tf.reshape(embeddings[1], [512, 1])

        return tf.matmul(vector1, vector2, transpose_a=True).numpy()[0][0]

class BERTScore:
    
    def __init__(self, cuda_device):
        self.scorer = BERTScorer(lang="en", rescale_with_baseline=True, device = cuda_device)
    
    def __call__(self, sentence1, sentence2):
        sentence1, sentence2 = sentence1.lower(), sentence2.lower()
        P, R, F1 = self.scorer.score([sentence1], [sentence2])
        return F1.tolist()[0]


class EditDistance:
    def __init__(self, normalized = False):
        self.lev = Levenshtein() if not normalized else NormalizedLevenshtein()
    
    def __call__(self, sentence1, sentence2):
        sentence1, sentence2 = sentence1.lower(), sentence2.lower()
        return self.lev.distance(sentence1, sentence2)

class GrammarChecker:
    def __init__(self):
        self.lang_tool = language_tool_python.LanguageTool('en-US')

    def __call__(self, sentence):
        sentence = sentence.lower()
        matches = self.lang_tool.check(sentence)
        return len(matches)


class GPT2LM:
    def __init__(self, cuda=-1, model_resolution = 'gpt2'):
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_resolution)
        self.lm = transformers.GPT2LMHeadModel.from_pretrained(model_resolution)
        # self.lm = torch.load('gpt2-large.pkl')
        self.cuda = cuda
        if self.cuda >= 0 :
            self.lm.cuda(self.cuda)

    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """
        sent = sent.lower()
        ipt = self.tokenizer(sent, return_tensors="pt", verbose=False)
        
        if self.cuda >= 0:
            for k in ipt.keys():
                ipt[k] = ipt[k].cuda(self.cuda)
        
        return math.exp(self.lm(**ipt, labels=ipt.input_ids)[0])

    
