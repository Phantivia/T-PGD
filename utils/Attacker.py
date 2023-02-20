import torch
from tqdm import tqdm
from dataclasses import dataclass
import re
import copy

from .Metrics import USE, BERTScore

import os
import random 


class FlipDetector:
    
    def __init__(self):
        path = os.path.dirname(__file__) + '/antonyms_vocab.dict'
        print("Trying to fetch antonyms vocabulary from: " + path)
        self.antonyms_vocab = torch.load(path)
        print("Successfully fetched antonyms_vocab.dict.")

    def remove_punctuation(self, text):
        punctuation = '!,;:?"\'/'
        text = re.sub(r'[{}]+'.format(punctuation),'',text)
        return text.strip()
    
    def get_atonyms(self, text):
        word_list = self.remove_punctuation(text).lower().split(" ")
        antonyms_dict = {}
        for idx in range(len(word_list)):
            if word_list[idx] in self.antonyms_vocab.keys():
                antonyms_dict[word_list[idx]] = idx
        return antonyms_dict, word_list

    def __call__(self, sentence1, sentence2, trigger_mag = 2):
        (d1,word_list1),( d2,word_list2) = self.get_atonyms(sentence1), self.get_atonyms(sentence2)
        for k1, i1 in d1.items():
            for k2, i2 in d2.items():
                if (k1 in self.antonyms_vocab[k2]) and (abs(i1 - i2) <= trigger_mag):
                    context_start = i1 - 3 if (i1 - 3) <= 0 else 0
                    context_end = i1 + 3 if (i1 + 3) < len(word_list1) else len(word_list1) - 1

                    word_list1[i1] = k1 + f'[{k2}]' # Origin_word[Flipped_word]
                    context = " ".join(word_list1[context_start: context_end])
                    return True, {i1:k1, i2:k2, 'context':context}
        return False, None

@dataclass
class AttackResult:
    succeed:bool = False
    flipped:bool = False
    flipped_content:dict = None
    query:int = 0
    bpstep:int = 0
    step_in_seg:int = 0

    ori_error:int = 0
    adv_error:int = 0
    use:float = 0.0
    bs:float = 0.0
    PPL:float = 0.0
    edit_distance:int = 0.0
    edit_distance_normalized:float = 0.0

    ori_sentence:str = None
    adv_sentence:str = None


class PandRAttacker:

    def __init__(self, model, dataset, encoded_dataset, text_key, prober, model_cuda_device, stop_ids, target_metric = 'bs'):

        self.model = model
        self.model_cuda_device = model_cuda_device

        self.dataset = dataset
        self.encoded_dataset = encoded_dataset
        self.encoded_dataset.set_format("numpy")
        self.text_key = text_key

        self.prober = prober
        self.decode_mode = prober.decode_mode

        self.bs = BERTScore(self.prober.victim_cuda_device) if target_metric == 'bs' else USE(self.prober.victim_cuda_device)
        self.flip_detector = FlipDetector()

        self.stop_ids = stop_ids


    def model_forward(self, input_ids, attention_mask, labels, decode_hidden_index = -1):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels = labels, output_hidden_states = True)
        loss, logits, hiddens = output['loss'], output['logits'], output['hidden_states']
        return loss, logits, hiddens[decode_hidden_index]


    def random_cover(self, ids, decode_mode, seed = 114514, pos = None):
            if len(pos) == 0: return ids
            random.seed(seed)
            rindex = random.randint(0, len(pos) - 1)
            rindex = pos[rindex]
            ids[:, rindex] = 104 if decode_mode == 'Bert' else 50264 if decode_mode == 'Roberta' else 4 if decode_mode == 'Albert' else 0
            return ids


    def attack_step(self, index = 0, bs_lower_limit = 0.0, bs_upper_limit = 0.50, task_loss_weight = 1.0, decode_loss_weight = 1.0,  init_mag = 1.0,
                adv_lr = 3e-2, num_seg_steps = 5, num_adv_steps = 3, perturb_layer_index = 0, decode_hidden_index = -1, average_input_length = 20,
                use_random_cover = True, show_info = False, task = None, attack_info = False, 
                scheduler = None, SEED = 114514):


        result = AttackResult()
        outputs ={
            'succeed':False,
            'max_bs':0.0,
            'rec_acc':0.0,
            'query':0,
            'results':[],
            'best_result':None,
            'ori_label':-1,
        }

        def run_test(ids, num_bp_steps = 0, num_step_in_seg = -1):
            for id in ids:
                victim_loss = torch.Tensor([0]).squeeze()
                if not id.tolist() in id_base:
                    id_base.append(id.tolist())
                    result.adv_sentence = self.prober.ids_to_sentence(id, adv_start = adv_start, adv_end = adv_end)
                    result.flipped, result.flipped_content = self.flip_detector(result.ori_sentence, result.adv_sentence)


                    if not result.flipped :
                        result.bs = self.bs(result.ori_sentence, result.adv_sentence)
                        
                        if result.bs > outputs['max_bs']:
                            result.succeed, victim_loss = self.prober.attack(result.adv_sentence, label, _ori_sentence = _ori_sentence, task = task, print_info = attack_info )
                            result.query += 1
                            outputs['query'] = result.query

                            if result.succeed:
                                result.bpstep = num_bp_steps
                                result.step_in_seg = num_step_in_seg
                                outputs['max_bs'] = result.bs
                                outputs['best_result'] = copy.deepcopy(result)
                                
                                outputs['results'].append(copy.deepcopy(result))

                                if show_info:
                                    print(f'HIT! ori = {result.ori_sentence}, adv = {result.adv_sentence} , FLIPPED = {result.flipped_content}, bs = {result.bs} ')
                                
                                if result.bs > bs_upper_limit:
                                    outputs['succeed'] = True
                                    return 'success', victim_loss

                        elif result.bs < bs_lower_limit:
                            return 'reset', victim_loss

                return 'continue', victim_loss

        result.ori_sentence = self.dataset[index][self.text_key]
        ori_input_ids = torch.tensor(self.encoded_dataset['input_ids'][index]).unsqueeze(0).cuda(self.model_cuda_device)
        attention_mask = torch.tensor(self.encoded_dataset['attention_mask'][index]).unsqueeze(0).cuda(self.model_cuda_device)
        label = torch.tensor(self.encoded_dataset['label'][index]).unsqueeze(0).cuda(self.model_cuda_device)

        perturb_positions = self.encoded_dataset['perturb_positions'][index]
        perturb_mask = torch.tensor(self.encoded_dataset['mask'][index]).unsqueeze(0).cuda(self.model_cuda_device)
        decode_mask = torch.tensor(self.encoded_dataset['mask'][index]).unsqueeze(0).cuda(self.model_cuda_device)

        #For MNLI task we do adversarial on specific slice of input_ids
        adv_start, adv_end = self.encoded_dataset['adv_start'][index], self.encoded_dataset['adv_end'][index]

        #_ori_sentence is used to save hypothesis while doing adversarial on premise / save premise while doing adversarial on hypothesis
        if task == 'mnli_hypothesis':
            _ori_sentence = self.dataset[index]['premise']
        elif task == 'mnli_premise':
            _ori_sentence = self.dataset[index]['hypothesis']
        else:
            _ori_sentence = None
        outputs['_ori_sentence'] = _ori_sentence
        
        # Test if model correctly classifies the origin input
        attack_with_origin_sentence_succeed, victim_loss = self.prober.attack(result.ori_sentence, label, _ori_sentence = _ori_sentence, task = task, print_info = attack_info)
        if attack_with_origin_sentence_succeed:
            return None

        input_length = len(ori_input_ids[0])
        input_ids = self.random_cover(ori_input_ids.clone(), self.decode_mode, seed=SEED, pos = perturb_positions) if use_random_cover else ori_input_ids.clone()

        id_base = [input_ids.tolist(), ]
        
        # Specify layer to perturb
        if self.decode_mode == 'Bert':
            p_layer = self.model.bert.encoder.layer[perturb_layer_index].attention.self
        elif self.decode_mode == 'Roberta':
            p_layer = self.model.roberta.encoder.layer[perturb_layer_index].attention.self
        elif self.decode_mode == 'Albert':
            p_layer = self.model.albert.encoder
            p_layer.set_pos(perturb_layer_index)
        p_layer.perturb = None

        # Calculate original reconstruction accuracy (for analysis)
        torch.manual_seed(SEED)
        origin_loss, origin_logits, origin_hidden = self.model_forward(input_ids=input_ids, attention_mask=attention_mask, labels = label, decode_hidden_index = decode_hidden_index)
        loss, reconstruct_ids = self.prober.decode(origin_hidden, origin_ids = ori_input_ids, decode_mask = decode_mask, perturb_positions = perturb_positions, stop_ids = self.stop_ids)
        outputs['rec_acc'] = ((input_ids == reconstruct_ids).sum() / input_ids.numel()).tolist()

        test_result = run_test(reconstruct_ids, num_bp_steps= -1)
        if test_result == 'success': return outputs

        # Initialize perturbation
        torch.manual_seed(SEED)
        p_layer.p_init(input_length, init_mag = init_mag, cuda_device=self.model_cuda_device, perturb_mask = perturb_mask)

        torch.manual_seed(SEED)
        loss, logits, hidden = self.model_forward(input_ids=input_ids, attention_mask=attention_mask, labels = label, decode_hidden_index=decode_hidden_index)
        decode_loss, init_ids = self.prober.decode(hidden, origin_ids = input_ids, decode_mask = decode_mask, perturb_positions = perturb_positions, stop_ids = self.stop_ids)

        test_result = run_test(init_ids, num_bp_steps= 0)
        if test_result == 'success': return outputs

        
        for seg_step in range(num_seg_steps):
            adv_accu_loop = tqdm(range(num_adv_steps)) if show_info else range(num_adv_steps)
            seg_max_model_loss, seg_max_victim_loss = 0.0, 0.0
            for adv_step in adv_accu_loop:
                
                lr = scheduler(adv_lr, seg_step, i) if scheduler is not None else adv_lr

                p_layer.p_accu(loss * task_loss_weight + decode_loss * decode_loss_weight, lr, input_length = input_length, average_input_length = average_input_length, perturb_mask = perturb_mask)

                torch.manual_seed(SEED)
                loss, logits, hidden =  self.model_forward(input_ids=input_ids, attention_mask=attention_mask, labels = label, decode_hidden_index = decode_hidden_index)
                decode_loss, p_ids = self.prober.decode(hidden, origin_ids = input_ids, decode_mask = decode_mask, perturb_positions = perturb_positions, stop_ids = self.stop_ids)

                test_result, victim_loss = run_test(p_ids, num_bp_steps= seg_step * num_adv_steps + adv_step + 1 ,num_step_in_seg = adv_step)

                seg_max_model_loss = loss.tolist() if loss.tolist() > seg_max_model_loss else seg_max_model_loss
                seg_max_victim_loss = victim_loss.tolist() if victim_loss.tolist() > seg_max_victim_loss else seg_max_victim_loss


                if test_result == 'success': return outputs
                elif test_result == 'reset': break

            if show_info:
                print(f'##RESET!## MAX_MODEL_DLOSS = {seg_max_model_loss}, VICTIM_LOSS = {seg_max_victim_loss}, QUERY: {result.query}')                    
            
            SEED += 1
            input_ids = self.random_cover(ori_input_ids.clone(), self.decode_mode, seed=SEED, pos = perturb_positions) if use_random_cover else ori_input_ids.clone()

            torch.manual_seed(SEED)
            p_layer.p_init(input_length, init_mag = init_mag, cuda_device=self.model_cuda_device, perturb_mask = perturb_mask)

            torch.manual_seed(SEED)
            loss, logits, hidden = self.model_forward(input_ids=input_ids, attention_mask=attention_mask, labels = label, decode_hidden_index=decode_hidden_index)

        torch.cuda.empty_cache()
        return outputs