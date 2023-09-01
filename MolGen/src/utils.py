
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch import nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
# from score_modules.SA_Score import sascorer
import selfies as sf
from rdkit.Chem import QED
import os
import subprocess
from tqdm import tqdm
import time
import math
import json
import pandas as pd
import pdb

def set_optim(opt, model_list, freeze_part=[], accumulation_step=None):
    named_parameters = []
    for model in model_list:
        model_para_train, freeze_layer = [], []
        model_para = list(model.named_parameters())
        for n, p in model_para:
            if not any(nd in n for nd in freeze_part):
                model_para_train.append((n, p))
            else:
                p.requires_grad = False
                freeze_layer.append((n, p))
                
        named_parameters.extend(model_para_train) 
    
    parameters = [
        {'params': [p for n, p in named_parameters], "lr": opt.lr, 'weight_decay': opt.weight_decay}
    ]
    
    if opt.optim == 'adamw':
        optimizer = optim.AdamW(parameters, lr=opt.lr, eps=opt.adam_epsilon)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(parameters, lr=opt.lr)
    
    # 梯度累计
    if accumulation_step is None:
        accumulation_step = opt.accumulation_steps
    # schedule 设定
    if opt.scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(opt.warmup_steps/accumulation_step), num_training_steps=int(opt.total_steps/accumulation_step))
    elif opt.scheduler == 'cos':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(opt.warmup_steps/accumulation_step), num_training_steps=int(opt.total_steps/accumulation_step))
    
    return optimizer, scheduler


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return 1.0


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio) * step / float(max(1, self.warmup_steps)) + self.min_ratio

        return max(0.0,
                   1.0 + (self.min_ratio - 1) * (step - self.warmup_steps) / float(max(1.0, self.scheduler_steps - self.warmup_steps)),
                )


class Loss_log():
    def __init__(self):
        self.loss = [999999.]
        self.acc = [0.]
        self.flag = 0
        self.token_right_num = []
        self.token_all_num = []
        self.use_top_k_acc = 0

    def acc_init(self, topn=[1]):
        self.loss = []
        self.token_right_num = []
        self.token_all_num = []
        self.topn = topn
        self.use_top_k_acc = 1
        self.top_k_word_right = {}
        for n in topn:
            self.top_k_word_right[n] = []
    
    def get_token_acc(self):
        if len(self.token_all_num) == 0:
            return 0.
        elif self.use_top_k_acc == 1:
            res = []
            for n in self.topn:
                res.append(round((sum(self.top_k_word_right[n]) / sum(self.token_all_num)) * 100 , 3))
            return res
        else:
            return [sum(self.token_right_num)/sum(self.token_all_num)]
    
    def update_token(self, token_num, token_right):
        self.token_all_num.append(token_num)
        if isinstance(token_right, list):
            for i, n in enumerate(self.topn):
                self.top_k_word_right[n].append(token_right[i])
        self.token_right_num.append(token_right)
        
    def update(self, case):
        self.loss.append(case)

    def update_acc(self, case):
        self.acc.append(case)

    def get_loss(self):
        return self.loss[-1]

    def get_acc(self):
        return self.acc[-1]

    def get_min_loss(self):
        return min(self.loss)

    def get_loss(self):
        if len(self.loss) == 0:
            return 500.
        return np.mean(self.loss)
    
    def early_stop(self):
        if self.loss[-1] > min(self.loss):
            self.flag += 1
        else:
            self.flag = 0

        if self.flag > 1000:
            return True
        else:
            return False

    def torch_accuracy(output, target, topk=(1,)):
        '''
        param output, target: should be torch Variable
        '''

        topn = max(topk)
        batch_size = output.size(0)

        _, pred = output.topk(topn, 1, True, True) 
        pred = pred.t() 

        is_correct = pred.eq(target.view(1, -1).expand_as(pred))

        ans = []
        ans_num = []
        for i in topk:
            is_correct_i = is_correct[:i].contiguous().view(-1).float().sum(0, keepdim=True)
            ans_num.append(int(is_correct_i.item()))
            ans.append(is_correct_i.mul_(100.0 / batch_size))

        return ans, ans_num

def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss

class label_smoothing_loss(nn.Module):
    def __init__(self, ignore_index, epsilon=0.1):
        super(label_smoothing_loss, self).__init__()
        self.ignore_idx = ignore_index
        self.epsilon = epsilon

    def forward(self, input, target):
        input = input.transpose(1, 2) # [batch_size, seq_len, word_num]
        input = torch.log_softmax(input, dim=2)
        k = input.size(2)
        target_prob = torch.ones_like(input).type_as(input) * self.epsilon * 1 / k
        mask = torch.arange(k).unsqueeze(0).unsqueeze(0).expand(target.size(0), target.size(1), -1).type_as(target)
        mask = torch.eq(mask, target.unsqueeze(-1).expand(-1, -1, k))
        target_prob.masked_fill_(mask, 1 - self.epsilon + (self.epsilon * 1 / k))
        loss = - torch.mul(target_prob, input)
        loss = loss.sum(2)
        # mask ignore_idx
        mask = (target != self.ignore_idx).type_as(input)
        loss = (torch.mul(loss, mask).sum() / mask.sum()).mean()
        return loss

def qed(smile):
    if smile:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            return QED.qed(mol)
        else: return -100
    else: return -100

def get_largest_ring_size(mol):
    cycle_list = mol.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length

def plogp(smile):
    if smile:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            log_p = Descriptors.MolLogP(mol)
            sas_score = sascorer.calculateScore(mol)
            largest_ring_size = get_largest_ring_size(mol)
            cycle_score = max(largest_ring_size - 6, 0)
            if log_p and sas_score and largest_ring_size:
                p_logp = log_p - sas_score - cycle_score
                return p_logp
            else: 
                return -100
        else:
            return -100
    else:
        return -100

def sim(input_smile, output_smile):
    if input_smile and output_smile:
        input_mol = Chem.MolFromSmiles(input_smile)
        output_mol = Chem.MolFromSmiles(output_smile)
        if input_mol and output_mol:
            input_fp = AllChem.GetMorganFingerprint(input_mol, 2)
            output_fp = AllChem.GetMorganFingerprint(output_mol, 2)
            sim = DataStructs.TanimotoSimilarity(input_fp, output_fp)
            return sim
        else: return 0
    else: return 0 

def sf_decode(selfies):
    try:
        decode = sf.decoder(selfies)
        return decode
    except sf.DecoderError:
        return ''
    
def sf_encode(smile):
    try:
        encode = sf.encoder(smile)
        return encode
    except sf.EncoderError:
        return '' 

def smiles_to_affinity(smiles, autodock, protein_file, num_devices, path):
    if not os.path.exists(f'{path}/ligands'):
        os.mkdir(f'{path}/ligands')
    if not os.path.exists(f'{path}/outs'):
        os.mkdir(f'{path}/outs')
    # subprocess.run('rm core.*', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run(f'rm {path}/outs/*.xml', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run(f'rm {path}/outs/*.dlg', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run(f'rm -rf {path}/ligands/*', shell=True, stderr=subprocess.DEVNULL)
    for device in range(num_devices):
        os.mkdir(f'{path}/ligands/{device}')
    device = 0
    for i, hot in enumerate(tqdm(smiles, desc='preparing ligands')):
        subprocess.Popen(f'obabel -:"{smiles[i]}" -O {path}/ligands/{device}/ligand{i}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d', shell=True, stderr=subprocess.DEVNULL)
        # subprocess.Popen(f'obabel -:"{smiles[i]}" -O ligands/{device}/ligand{i}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d', shell=True)
        device += 1
        if device == num_devices:
            device = 0
    while True:
        total = 0
        for device in range(num_devices):
            total += len(os.listdir(f'{path}/ligands/{device}'))
        if total == len(smiles):
            break
    
    time.sleep(1)
    print('running autodock..')
    if len(smiles) == 1:
        subprocess.run(f'{autodock} -M {protein_file} -s 0 -L {path}/ligands/0/ligand0.pdbqt -N {path}/outs/ligand0', shell=True, stdout=subprocess.DEVNULL)
    else:
        ps = []
        for device in range(num_devices):
            ps.append(subprocess.Popen(f'{autodock} -M {protein_file} -s 0 -B {path}/ligands/{device}/ligand*.pdbqt -N ../../outs/ -D {device + 1}', shell=True))
        stop = False
        while not stop: 
            for p in ps:
                stop = True
                if p.poll() is None:
                    time.sleep(1)
                    stop = False
    affins = [0 for _ in range(len(smiles))]
    for file in tqdm(os.listdir(f'{path}/outs'), desc='extracting binding values'):
        if file.endswith('.dlg') and '0.000   0.000   0.000  0.00  0.00' not in open(f'{path}/outs/{file}').read():
            affins[int(file.split('ligand')[1].split('.')[0])] = float(subprocess.check_output(f"grep 'RANKING' {path}/outs/{file} | tr -s ' ' | cut -f 5 -d ' ' | head -n 1", shell=True).decode('utf-8').strip())
    result = [min(affin, 0) for affin in affins]
    dc = [delta_g_to_kd(i) for i in result]
    return dc

def delta_g_to_kd(x):
    return math.exp(x / (0.00198720425864083 * 298.15))

