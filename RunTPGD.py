
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default=None, help='task to do')

parser.add_argument('--model_checkpoint', type=str, default=None, help='task to do')
parser.add_argument('--tokenizer_checkpoint', type=str, default=None, help='task to do')
parser.add_argument('--decode_mode', type=str, default=None, help='task to do')

parser.add_argument('--victim_model_checkpoint', type=str, default=None, help='task to do')
parser.add_argument('--victim_tokenizer_checkpoint', type=str, default=None, help='task to do')

parser.add_argument('--data_local_path', type=str, default=None, help='task to do')

parser.add_argument("--cuda_device", type=int, default=0)
parser.add_argument("--victim_device", type=int, default=-1)

parser.add_argument("--start", type=int, default=0, help="start step, for multi-thread process")
parser.add_argument("--end", type=int, default=50, help="end step, for multi-thread process")

parser.add_argument('--decode_layer', type=int, default=6)
parser.add_argument('--perturb_layer', type=int, default=0)
parser.add_argument('--num_seg_steps', type=int, default=10)
parser.add_argument('--num_adv_steps', type=int, default=100)

parser.add_argument('--adv_lr', type=float, default = 3.0)
parser.add_argument('--init_mag', type=float, default=1.0)
parser.add_argument("--decode_weight", type=float, default= -0.001)

parser.add_argument('--bs_lower_limit', type=float, default=0.0)
parser.add_argument('--bs_upper_limit', type=float, default=1.0)

parser.add_argument('--local_victim', type=str, default=None)
parser.add_argument('--local_model', type=str, default=None)
parser.add_argument('--target_metric', type=str, default='bs')

parser.add_argument('--stop_random_cover', type=bool, default=False)

parser.add_argument('--eval_lower_limit', type=float, default=0.0)


args = parser.parse_args()
print("ARGS: ", args)

cuda_device, victim_device = args.cuda_device, args.victim_device
import os
if victim_device != -1:
    print("running process on cuda:", cuda_device, "victim model loaded on cuda:", victim_device) 
else:
    print("running process on cuda:", cuda_device)
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)+"," +str(victim_device) if victim_device != -1 else str(cuda_device)
cuda_device = 0
victim_device = 1 if victim_device != -1 else 0


import torch
import numpy as np

import copy
from tqdm import tqdm
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import Block, MakeDataset, Prober, Attacker
from datasets import Dataset
import pandas as pd
import copy 

from utils.Metrics import USE, BERTScore, EditDistance


# %%

task = args.task
tokenizer_checkpoint = args.tokenizer_checkpoint
decode_mode = args.decode_mode
model_checkpoint = args.model_checkpoint
data_local_path = args.data_local_path

dataset, encoded_dataset, num_labels, text_key, stop_ids = MakeDataset.make(task = task, tokenizer_checkpoint = tokenizer_checkpoint, local_path = data_local_path)

# %%

# tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, use_fast=True, mirror = 'tuna')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, use_fast=True)
if args.local_model is None:
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
else:
    model = torch.load(args.local_model)

temp_state = copy.deepcopy(model.state_dict())

if decode_mode == 'Bert':
    for i in model.bert.encoder.layer:
        p_layer = Block.MyBertSelfAttention(model.config)
        i.attention.self = p_layer
elif decode_mode == 'Roberta':
    for i in model.roberta.encoder.layer:
        p_layer = Block.MyBertSelfAttention(model.config)
        i.attention.self = p_layer
elif decode_mode == 'Albert':
        p_layer = Block.MyAlbertTransformer(model.config)
        model.albert.encoder = p_layer

model.load_state_dict(temp_state)
struct = model.cuda(cuda_device)


# %%

mlm_model = AutoModelForMaskedLM.from_pretrained(tokenizer_checkpoint)
if decode_mode == 'Bert':
    probe_layer = mlm_model.cls
elif decode_mode == 'Roberta':
    probe_layer = mlm_model.lm_head
elif decode_mode == 'Albert':
    probe_layer = mlm_model.predictions
probe_layer.cuda(cuda_device)

if args.local_victim is None:
    victim_model = AutoModelForSequenceClassification.from_pretrained(args.victim_model_checkpoint)
else:
    victim_model = torch.load(args.local_victim)

victim_tokenizer = AutoTokenizer.from_pretrained(args.victim_tokenizer_checkpoint)
    
prober = Prober.Prober(tokenizer, probe_layer, victim_model, victim_tokenizer = victim_tokenizer, victim_cuda_device=victim_device, model_cuda_device=cuda_device, decode_mode = decode_mode)

attacker = Attacker.PandRAttacker(model, dataset, encoded_dataset, text_key, prober, model_cuda_device=cuda_device, stop_ids = stop_ids, target_metric = args.target_metric)

lengths = []
for mask in encoded_dataset['mask']:
    lengths.append(mask.sum())
average_input_length = np.array(lengths).mean()
print('average_input_length = ', average_input_length)

# Fix Label mismatch problem for MNLI dataset
if task.startswith('mnli'):
    
    label_fix_dict = {
            'bert-base-uncased':{0:1, 1:2, 2:0},
            'roberta-base':{0:2, 1:1, 2:0},
            'albert-base-v2':{0:0, 1:1, 2:2},
        }
    data_list = []
    for dat in encoded_dataset:
        data_list.append(copy.deepcopy(dat))
        d = data_list[-1]
        d['label'] = label_fix_dict[args.victim_tokenizer_checkpoint][d['label']]
        for key in ['input_ids', 'attention_mask']:
            d[key] = list(d[key])
    
    df = pd.DataFrame(data_list)
    ds = Dataset.from_pandas(df)
    test_dataset = ds
    test_dataset.set_format("numpy")
    attacker.encoded_dataset = test_dataset


results = []
bs_random_cover = False if args.stop_random_cover else True
for index in tqdm(range(args.start, args.end)):
    result = attacker.attack_step(index = index, bs_lower_limit = args.bs_lower_limit, bs_upper_limit = args.bs_upper_limit, task_loss_weight = 1.0, decode_loss_weight = args.decode_weight,  init_mag = args.init_mag,
                        adv_lr = args.adv_lr ,num_seg_steps = args.num_seg_steps, num_adv_steps = args.num_adv_steps, perturb_layer_index = args.perturb_layer, decode_hidden_index = args.decode_layer, average_input_length = average_input_length,
                        use_random_cover = True, show_info = False, task = task, 
                        scheduler = None, SEED = 114514)
    results.append(result)
    
num_attack_failed = 0
num_predict_failed = 0
num_predict_succeed = 0

use = USE(cuda_device)
# bs = BERTScore(cuda_device)
def bs(s1, s2):
    return 0.4999
edit_distance = EditDistance()
edit_distance_norm = EditDistance(normalized=True)

uses = []
bss = []
edit_distances = []
edit_distances_norm = []
num_word_changes = []

queries = []
bpsteps = []
segsteps = []
uses = []
for r in results:
    if r is None:
        num_predict_failed += 1
    if r is not None:
        num_predict_succeed += 1
        if r['max_bs'] <= args.eval_lower_limit:
            num_attack_failed += 1
        else:
            if r['best_result'] is not None:
                
                queries.append(r['query'])
                bpsteps.append(r['best_result'].bpstep)
                segsteps.append(r['best_result'].step_in_seg)

                uses.append(use(r['best_result'].ori_sentence, r['best_result'].adv_sentence))
                bss.append(bs(r['best_result'].ori_sentence, r['best_result'].adv_sentence))
                edit_distances.append(edit_distance(r['best_result'].ori_sentence, r['best_result'].adv_sentence))
                edit_distances_norm.append(edit_distance_norm(r['best_result'].ori_sentence, r['best_result'].adv_sentence))
                num_word_changes.append(len(r['best_result'].adv_sentence) - len(r['best_result'].ori_sentence))

                if task.startswith('mnli'):
                    print("CONTEXT: ", r['_ori_sentence'])
                print("ORI:", r['best_result'].ori_sentence)
                print("ADV:", r['best_result'].adv_sentence)

origin_acc = num_predict_succeed / (num_predict_succeed + num_predict_failed)
acc_under_attack = num_attack_failed / (num_predict_succeed + num_predict_failed)

ASR = (origin_acc - acc_under_attack) / origin_acc

uses = np.sort(np.array(uses))
bss = np.sort(np.array(bss))
edit_distances = np.sort(np.array(edit_distances))
edit_distances_norm = np.sort(np.array(edit_distances_norm))
num_word_changes = np.array(num_word_changes)

len98, len95, len90 = int(0.02 * len(uses)), int(0.05 * len(uses)), int(0.1 * len(uses))
print(len98, len95, len90)

print("USES", uses)
print("BSS", bss)

print("ARGS:", args)

print("Steps in seg: ",segsteps)
print(f"Origin Acc={origin_acc}, Acc under attack={acc_under_attack}")
print(f'ASR={ASR},BS={bss.mean()}, USE={uses.mean()}, 98%USE={uses[len98:].mean()}, 95%USE={uses[len95:].mean()}, 90%USE={uses[len90:].mean()}, LAST 10% USE={uses[:len90].mean()}')
print(f'Edit Distance={edit_distances.mean()}, Edit Distance(Normalized)={edit_distances_norm.mean()}, NUM_WORD_CHANGES={num_word_changes.mean()}')
print(f'Query={np.array(queries).mean()}, Bpsteps={np.array(bpsteps).mean()}, NumberEvalAttacks={len(uses)}, AverageStepInSeg={np.array(segsteps).mean()}')


