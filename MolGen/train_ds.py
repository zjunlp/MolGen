import os
import os.path as osp
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import deepspeed
from config import cfg
from torchlight import initialize_exp, set_seed, get_dump_path
from src.data import DataCollatorForDenoisingTasks, SelfiesDataset
from src.utils import Loss_log
import warnings
warnings.filterwarnings('ignore')
from model import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers.trainer_pt_utils import get_parameter_names
from src.distributed_utils import init_distributed_mode, is_main_process
import torch.distributed as dist

class Runner:
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
        self.pretrain_path = args.pretrain_path
        self.data_init()
        
        set_seed(args.random_seed)
        self.dataloader_init(self.pretrain_set) 
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
        
        if self.rank == 0:
            self.logger.info("Loading pretrain dataset...")
        self.pretrain_set = SelfiesDataset(self.tokenizer, self.pretrain_path)
        if self.rank == 0:
            self.logger.info("Finish loading!")

        self.pretrain_sampler = torch.utils.data.distributed.DistributedSampler(self.pretrain_set)

    def dataloader_init(self, pretrain_set=None):
        bs = self.args.batch_size

        self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
        pretrain_collator = DataCollatorForDenoisingTasks(
            self.tokenizer,
            self.mask_ratio,
            self.poisson_lambda,
            self.pad_to_multiple_of,
        )   
        self.train_dataloader = self._dataloader_dist(pretrain_set, self.pretrain_sampler, bs, pretrain_collator)
    

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
            pin_memory=False,
            num_workers=self.args.workers,
            persistent_workers=True, 
            drop_last=True,
            batch_size=batch_size,
            collate_fn=collator,
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


    def loss_output(self, batch):
        input_ids = batch["input_ids"].cuda()
        decoder_input_ids = batch["decoder_input_ids"].cuda()
        attention_mask = input_ids != self.tokenizer.pad_token_id
        labels = batch["labels"].cuda()
        _output = self.model_engine(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids, 
                            labels=labels)
        loss = _output.loss
        return loss


    # one time train
    def train(self, _tqdm):
        self.model_engine.train()
        curr_loss = 0.
        self.loss_log.acc_init()
        for batch in self.train_dataloader:
            # Forward pass
            loss = self.loss_output(batch)
            # Backward pass
            self.model_engine.backward(loss)
            # Optimizer Step
            self.model_engine.step()

            self.step += 1
            
            # -------- stat --------
            if is_main_process():
                self.output_statistic(loss, output=None)
                self.lr = self.get_learning_rate()
                _tqdm.set_description(f'Train | step [{self.step}/{self.args.total_steps}] LR [{self.lr:.5f}] Loss {self.loss_log.get_loss():.5f} ')
                if self.step % self.args.eval_step == 0 and self.step > 0:
                    self.loss_log.update(self.curr_loss)
                    self.update_loss_log()
                    self.writer.add_scalars("lr", {"lr": self.lr}, self.step)

            if self.step % 10000 == 0:
                self.model_engine.save_checkpoint(save_dir=os.path.join(self.logger_path, 'model'), client_state={'checkpoint_step': self.step})

    def get_learning_rate(self):
        # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
        # not run for the first few dozen steps while loss scale is too large, and thus during
        # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
        try:
            last_lr = self.scheduler.get_last_lr()[-1]
        except:
            last_lr = 0
        # except AssertionError as e:
        #     if "need to call step" in str(e):
        #         logger.warning("tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
        #         last_lr = 0
        #     else:
        #         raise
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
        
        
