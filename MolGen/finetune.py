import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import deepspeed
import torch.nn.functional as F
from config import cfg
from torchlight import initialize_exp, set_seed, get_dump_path
from src.data import DataCollatorForFinetuneTasks, CandidateDataset
from src.utils import Loss_log, RankingLoss, label_smoothing_loss
import warnings
warnings.filterwarnings('ignore')
from model import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers.trainer_pt_utils import get_parameter_names
from src.distributed_utils import init_distributed_mode, is_main_process
import torch.distributed as dist

class Runner():
    def __init__(self, args, writer=None, logger=None, rank=0):
        self.rank = rank
        self.mask_ratio = args.mask_ratio
        self.poisson_lambda = args.poisson_lambda
        self.pad_to_multiple_of = args.pad_to_multiple_of
        self.args = args
        self.writer = writer
        self.logger = logger
        self.logger_path = get_dump_path(self.args)
        # model choice
        self.model = BartForConditionalGeneration(BartConfig())
        
        # data loading
        self.finetune_path = args.finetune_path
        self.data_init()

        set_seed(args.random_seed)
        self.dataloader_init(self.finetune_set)  
        self.optim_init(self.args)

    def data_init(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        unwanted_words = [i for i in self.tokenizer.encoder.keys()]
        
        important_tokens = ['<s>','<pad>','</s>','<unk>']
        unwanted_words = list(set(unwanted_words).difference(set(important_tokens)))
        for word in unwanted_words:
            del self.tokenizer.encoder[word]
        selfies_tokens = np.load('../moldata/vocab_list/zinc.npy').tolist()
        self.tokenizer.add_tokens(selfies_tokens, special_tokens=False)
        self.tokenizer.add_tokens('<mask>', special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.load_state_dict(torch.load(self.args.checkpoint_path, map_location='cpu'),strict=False)
        
        self.config = self.model.config
        self.match_n_layer = self.config.decoder_layers
        self.match_n_head = self.config.decoder_attention_heads
        self.n_embd = self.config.d_model
        assert self.n_embd % self.match_n_head == 0
        self.match_n_embd = self.n_embd // self.match_n_head # huggingface BART's dim of kv need to be calculated
        
        # Prefix related.
        self.preseqlen = self.args.prefix_sequence_length
        self.mid_dim = self.args.mid_dim
        if self.rank == 0:
            self.logger.info("prefix-tuning sequence length is {}.".format(self.preseqlen))
        self.input_tokens = torch.arange(self.preseqlen).long().cuda()
        
        self.wte = nn.Embedding(self.preseqlen, self.n_embd).cuda()
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        ).cuda()

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd).cuda()
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        ).cuda()

        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd).cuda()
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        ).cuda()
        
        if self.rank == 0:
            self.logger.info("Loading finetune dataset...")
            
        self.finetune_set = CandidateDataset(self.tokenizer, self.finetune_path, max_len=self.args.max_length)
        if self.rank == 0:
            self.logger.info("Finish loading!")

        self.finetune_sampler = torch.utils.data.distributed.DistributedSampler(self.finetune_set)
    
    def dataloader_init(self, finetune_set=None):
        bs = self.args.batch_size
        self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
        finetune_collator = DataCollatorForFinetuneTasks(
            self.tokenizer,
            self.mask_ratio,
            self.poisson_lambda,
        )   
        self.train_dataloader = self._dataloader_dist(finetune_set, self.finetune_sampler, bs, finetune_collator)


    def optim_init(self, opt, total_step=None):
        step_per_epoch = len(self.train_dataloader)
        opt.total_steps = int(step_per_epoch * opt.epoch) if total_step is None else int(total_step)
        
        if self.rank == 0 and total_step is None:
            self.logger.info(f"total_steps: {opt.total_steps}")
            self.logger.info(f"weight_decay: {opt.weight_decay}")

    def _dataloader_dist(self, train_set, train_sampler, batch_size, collator):
        torch.multiprocessing.set_start_method('spawn', force=True)
        train_dataloader = DataLoader(
            train_set,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=self.args.workers,
            persistent_workers=True, 
            drop_last=True,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def _dataloader(self, train_set, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            num_workers=self.args.workers,
            persistent_workers=True,
            shuffle=(self.args.only_test == 0),
            drop_last=(self.args.only_test == 0),
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    
    def run(self):
        self.loss_log = Loss_log()
        self.curr_loss = 0.
        self.lr = self.args.lr
        self.curr_loss_dic = defaultdict(float)
        self.loss_weight = [1, 1]
        self.step = 0
        
        if self.args.smooth > 0:
            self.mle_fn = label_smoothing_loss(ignore_index=1, epsilon=self.args.smooth)
        else:
            self.mle_fn = torch.nn.CrossEntropyLoss(ignore_index=1)
        
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        self.model_engine, optimizer, _, self.scheduler = deepspeed.initialize(self.args, model=self.model,
                                                        model_parameters=optimizer_grouped_parameters)
        
        with tqdm(total=self.args.epoch) as _tqdm: 
            for i in range(self.args.epoch):
                self.train(_tqdm)
                _tqdm.update(1)

        self.model_engine.save_checkpoint(save_dir=os.path.join(self.logger_path, 'model'), client_state={'checkpoint_step': self.step})

    def get_prompt(self, bsz=None):
        old_bsz = bsz
        bsz = bsz * 30
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb


        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        past_key_values_dec = self.control_trans_dec(
            temp_control_dec
        )  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (
            self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        )
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb


        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, preseqlen
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_dec.device)
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
            }
            result.append(temp)

        return result


    def loss_output(self, batch):
        # contrastive loss
        src_id = batch["src_input_ids"].cuda()
        candidate_id = batch["candidate_ids"].cuda()
        batch_size = src_id.size(0)
        input_mask = src_id != self.tokenizer.pad_token_id
        cand_mask = candidate_id != self.tokenizer.pad_token_id
        cand_mask[:, :, 0] = 1

        self.model_engine.scoring_mode()
        past_prompt = self.get_prompt(batch_size)
        output = self.model_engine(
            input_ids=src_id, 
            attention_mask=input_mask,
            decoder_input_ids=candidate_id, 
            decoder_attention_mask=cand_mask,
            output_hidden_states=True,
            past_prompt=past_prompt,
            )
        output = output[0] # [bz x cand_num, seq_len, word_dim]
        output = output.view(batch_size, -1, output.size(1), output.size(2)) # [bz, cand_num, seq_len, word_dim]
        probs = output[:, 0]
        output = output[:, :, :-1]  # truncate last token
        candidate_id = candidate_id[:, :, 1:]  # shift right
        cand_mask = candidate_id != self.tokenizer.pad_token_id
        candidate_id = candidate_id.unsqueeze(-1)
        if self.args.normalize:
            if self.args.score_mode == "log":
                _output = F.log_softmax(output, dim=3)
            else:
                _output = F.softmax(output, dim=3)
            scores = torch.gather(_output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
        else:
            scores = torch.gather(output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
        cand_mask = cand_mask.float()
        scores = torch.mul(scores, cand_mask).sum(-1) / ((cand_mask.sum(-1)) ** self.args.length_penalty) # [bz, cand_num]
        output = {'score': scores, "probs": probs}   
        similarity = scores  
        ranking_loss = RankingLoss(similarity, None, self.args.cand_margin, self.args.gold_margin, self.args.gold_weight, True)
        probs = output["probs"]  # [bz, seq_len, word_num]
        probs = output["probs"][:, :-1]  # truncate last token
        gold = src_id[:, 1:]  # shift right
        mle_loss = self.mle_fn(probs.transpose(1, 2), gold)
        loss = self.args.rank_weight * ranking_loss + self.args.mle_weight * mle_loss
        return loss, ranking_loss, mle_loss

    # one time train
    def train(self, _tqdm):
        self.model_engine.train()
        curr_loss = 0.
        self.loss_log.acc_init()
        for batch in self.train_dataloader:
            # Forward pass
            loss, _, _ = self.loss_output(batch)
            # Backward pass
            self.model_engine.backward(loss)
            # Optimizer Step
            self.model_engine.step()

            self.step += 1
            
            # -------- statistics --------
            if is_main_process():
                self.output_statistic(loss, output=None)
                self.lr = self.get_learning_rate()
                _tqdm.set_description(f'Train | step [{self.step}/{self.args.total_steps}] LR [{self.lr:.5f}] Loss {self.loss_log.get_loss():.5f} ')
                if self.step % self.args.eval_step == 0 and self.step > 0:
                    self.loss_log.update(self.curr_loss)
                    self.update_loss_log()
                    self.writer.add_scalars("lr", {"lr": self.lr}, self.step)

            if self.step % 1000 == 0:
                self.model_engine.save_checkpoint(save_dir=os.path.join(self.logger_path, 'model'), client_state={'checkpoint_step': self.step})

    def get_learning_rate(self):
        # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
        # not run for the first few dozen steps while loss scale is too large, and thus during
        # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
        try:
            last_lr = self.scheduler.get_last_lr()[-1]
        except:
            last_lr = 0
        return last_lr

    def output_statistic(self, loss, output):
        self.curr_loss += loss.item()
        if output is None:
            return
        for key in output['loss_dic'].keys():
            self.curr_loss_dic[key] += output['loss_dic'][key]
        
        if 'loss_weight' in output and output['loss_weight'] is not None:
            self.loss_weight = output['loss_weight']
            
    def update_loss_log(self):
        vis_dict = {"train_loss": self.curr_loss}
        vis_dict.update(self.curr_loss_dic)
        self.writer.add_scalars("loss", vis_dict, self.step)
        
        # init log loss
        self.curr_loss = 0.
        for key in self.curr_loss_dic:
            self.curr_loss_dic[key] = 0.
        

if __name__ == '__main__':
    cfg = cfg()
    cfg.get_args()
    cfgs = cfg.update_train_configs()
    
    set_seed(cfgs.random_seed)
    init_distributed_mode(args=cfgs)
    rank = cfgs.rank
    
    writer, logger = None, None
    if rank == 0:
        logger = initialize_exp(cfgs)
        logger_path = get_dump_path(cfgs)
        cfgs.time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        comment = f'bath_size={cfgs.batch_size} exp_id={cfgs.exp_id}'
        if not cfgs.no_tensorboard and not cfgs.only_test:
            writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', cfgs.time_stamp), comment=comment)

    cfgs.device = torch.device(cfgs.device)
    
    # -----  Begin ----------
    torch.cuda.set_device(cfgs.gpu)
    runner = Runner(cfgs, writer, logger, rank)
    if cfgs.only_test:
        runner.test()
    else:
        runner.run()

    # -----  End ----------
    if not cfgs.no_tensorboard and not cfgs.only_test and rank == 0:
        writer.close()
        logger.info("done!")
        
    if cfgs.dist and not cfgs.only_test:
        dist.barrier()
        dist.destroy_process_group()
        
        
