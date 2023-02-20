import torch
from torch.nn import CrossEntropyLoss
class Prober():
    
    def __init__(self, tokenizer,  probe_layer, victim_model, victim_cuda_device = 0, model_cuda_device = 0, decode_mode = 'Bert', victim_tokenizer = None):
        self.celoss = CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.probe_layer = probe_layer
        self.victim = victim_model
        self.victim_cuda_device = victim_cuda_device
        self.model_cuda_device = model_cuda_device
        self.decode_mode = decode_mode

        self.victim_tokenizer = victim_tokenizer if victim_tokenizer is not None else tokenizer

        
        if torch.cuda.is_available():
            self.probe_layer.cuda(model_cuda_device)

        if self.victim_cuda_device is not None:
            self.victim.cuda(victim_cuda_device)


    def decode(self, hidden, origin_ids = None, decode_mask = None, perturb_positions = None, stop_ids = None):

        probe = self.probe_layer(hidden)
        
        _probe = probe.detach()
        
        if self.decode_mode == 'Bert':
            _probe[:, :,0:1000] = 0.0
        elif self.decode_mode == 'Roberta':
            _probe[:, :,0:4] = 0.0
            _probe[:, :,50264] = 0.0
        elif self.decode_mode == 'Albert':
            _probe[:, :, 0:5] = 0.0
        
        _probe[:, :, stop_ids] = 0.0

        reconstruction_ids = torch.topk(_probe, 1, -1)[1].squeeze(-1)

        #Process mask

        ksam_edoced = (-decode_mask + 1).int() #reversed mask [Example: 0, 0, 1 → 1, 0, 0]
        reconstruction_ids = reconstruction_ids * decode_mask + origin_ids * ksam_edoced
        reconstruction_ids = reconstruction_ids.long()

        probe = probe[:, perturb_positions, :]
        origin_ids = origin_ids[:, perturb_positions].long()
        decode_loss = self.celoss(probe.reshape(-1, self.tokenizer.vocab_size), origin_ids.reshape(-1))

        return decode_loss, reconstruction_ids
    
    def ids_to_sentence(self, ids, adv_start =None, adv_end =None):

        ids = ids[adv_start:adv_end]
        if self.decode_mode == 'Albert':
            ids = ids.unsqueeze(0)
        reconstruction_tokens = self.tokenizer.batch_decode(ids)
        reconstruction_sentence = self.tokenizer.convert_tokens_to_string(reconstruction_tokens)
        return reconstruction_sentence
        
    
    def attack(self, reconstruction_sentence, answer_label, print_info = False, _ori_sentence = None, task = None):
        if _ori_sentence is not None:
            if task == 'mnli_hypothesis':    #Attack on hypothesis
                inputs = self.victim_tokenizer(_ori_sentence, reconstruction_sentence, return_tensors = 'pt', truncation=True,)
            elif task == 'mnli_premise':  #Attack on premise
                inputs = self.victim_tokenizer(reconstruction_sentence, _ori_sentence, return_tensors = 'pt', truncation=True,)
        else:
            inputs = self.victim_tokenizer(reconstruction_sentence, return_tensors = 'pt', truncation=True,)
        inputs['labels'] = answer_label
        

        for k in inputs.keys():
            if self.victim_cuda_device is not None:
                inputs[k] = inputs[k].cuda(self.victim_cuda_device)

        output = self.victim(**inputs)
        loss = output[0]
        logits = output[1]

        adv_answer = logits.argmax(dim = 1).cuda(self.model_cuda_device)
        answer_label = answer_label.cuda(self.model_cuda_device)

        if print_info:
            print(reconstruction_sentence)
            print('ORI_LABEL', answer_label,'ADV_LABEL:', adv_answer, "ADV_LOGITS: ", logits, "LOSS: ", loss )
        return (False, loss) if adv_answer.equal(answer_label) else (True, loss)