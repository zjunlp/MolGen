import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import numpy as np
import deepspeed
from config import cfg
from torchlight import initialize_exp, set_seed, get_dump_path
from src.utils import set_optim, Loss_log
import warnings
warnings.filterwarnings('ignore')
from model import BartTokenizer, BartForConditionalGeneration, BartConfig
import pandas as pd
import selfies as sf
from pandarallel import pandarallel
import numpy as np
import selfies as sf
import warnings
warnings.filterwarnings('ignore')
from src.distributed_utils import init_distributed_mode
import torch.distributed as dist
from src.utils import qed, plogp, sim, sf_decode, sf_encode
import moses

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
        self.finetune_path = args.finetune_path
        self.data_init()
        set_seed(args.random_seed)

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
        
        if self.rank == 0:
            self.logger.info("Loading downstream dataset...")
            
        self.input_data = pd.read_csv(self.args.input_path)

        pandarallel.initialize(shm_size_mb=60720, nb_workers=20,progress_bar=True)
        if 'selfies' not in self.input_data.columns.tolist():
            if rank == 0:
                self.logger.info('convert smiles to selfies ...')
            self.input_data['selfies'] = self.input_data['smiles'].parallel_apply(sf_encode)    
            self.input_data.to_csv(self.args.input_path, index=None)
            
        if self.rank == 0:
            self.logger.info("Finish loading!")
        
        input_selfies = self.input_data['selfies'].tolist()
        self.input_dataloader = DataLoader(input_selfies, batch_size=self.args.batch_size,shuffle=False)

    def generate_molecules(self):
        self.model_engine, _, _, self.scheduler = deepspeed.initialize(self.args, model=self.model,
                                                        model_parameters=self.model.parameters())
        self.model_engine.eval()
        if rank == 0:
            self.logger.info(f'start generating with {self.args.generate_mode} ...')
        candidates = []
        candidate_smiles = []
        with tqdm(total=len(self.input_dataloader)) as _tqdm:
            for i, batch in enumerate(self.input_dataloader):
                _tqdm.set_description(f'Generate | step [{i}/{len(self.input_dataloader)}]')
                batch_encode = self.tokenizer.batch_encode_plus(batch, max_length=self.args.max_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                if self.args.generate_mode == 'beam':
                    molecules = self.model_engine.generate(
                        input_ids=batch_encode["input_ids"].cuda(),
                        attention_mask=batch_encode["attention_mask"].cuda(),
                        num_return_sequences=self.args.beam, 
                        num_beam_groups=self.args.beam, 
                        diversity_penalty=self.args.diversity_penalty, 
                        num_beams=self.args.beam,
                        max_length=self.args.max_len,
                        min_length=self.args.min_len,
                        length_penalty=self.args.length_penalty,
                        early_stopping=True,
                        past_prompt=None
                    )
                elif self.args.generate_mode == 'topk':
                    molecules = self.model_engine.generate(
                        input_ids=batch_encode["input_ids"].cuda(),
                        attention_mask=batch_encode["attention_mask"].cuda(),
                        do_sample=True,
                        max_length=self.args.max_len,
                        min_length=self.args.min_len,
                        top_k=self.args.top_k,
                        top_p=self.args.top_p,
                        temperature = self.args.temperature,
                        num_return_sequences=self.args.return_num,
                        past_prompt=None
                    )
                cand = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ","") for g in molecules]
                cand_smiles = [sf.decoder(selfies) for selfies in cand]
                candidates.extend(cand)
                candidate_smiles.extend(cand_smiles)
                _tqdm.update(1)
        input_smiles = self.input_data['smiles'].tolist()
        input = [i for i in input_smiles for r in range(self.args.return_num)]
        if self.args.property == 'plogp':
            input_prop = self.input_data[self.args.property].tolist()
            input_props = [i for i in input_prop for r in range(self.args.return_num)]
            pairs = {"start_smiles": input, f"input_{self.args.property}": input_props, "candidates": candidates, "candidate_smiles": candidate_smiles}
        else:
            pairs = {"start_smiles": input, "candidates": candidates, "candidate_smiles": candidate_smiles}
        data = pd.DataFrame(pairs)
        if rank == 0:
            self.logger.info(f'saving dataframe to {self.args.generate_path} ...')
        data.to_csv(self.args.generate_path, index=None) 
        if rank == 0:
            self.logger.info('finish!') 

    def statistics(self):
        data = pd.read_csv(self.args.generate_path)
        data.dropna(axis=0, how='any', inplace=True)
        pandarallel.initialize(shm_size_mb=60720, nb_workers=40,progress_bar=True)
        
        if self.args.property == 'plogp':
            data['output_plogp'] = data['candidate_smiles'].parallel_apply(plogp)
        elif self.args.property == 'qed':
            data['output_qed'] = data['candidate_smiles'].parallel_apply(qed)
            data['input_qed'] = data['start_smiles'].parallel_apply(qed)

        data.to_csv(self.args.generate_path)

        data['sim'] = data.parallel_apply(lambda x: sim(x['start_smiles'],x['candidate_smiles']),axis=1)
        if self.args.property == 'plogp':
            data['improve'] = data[f'output_{self.args.property}'] - data[f'input_{self.args.property}']
            data['improve'][data[f'output_{self.args.property}']==-100]=0
            for similarity in [0.0, 0.2, 0.4, 0.6]:
                data['improve'][data['sim']<similarity]=0
                df = data['improve'].groupby(data['start_smiles'])
                improve = df.max()
                imp = np.array(improve)
                if rank==0:
                    self.logger.info(f'when sim>={similarity}, improvement:')
                    im = imp[imp!=0]
                    self.logger.info(im.mean())
            df = data[f'output_{self.args.property}'].groupby(data['start_smiles'])
            max_plogp = df.max()
            if rank==0:
                self.logger.info('top 3 max plogp:')
            max_plogps = sorted(max_plogp, reverse=True)[0:3]
            if rank==0:
                self.logger.info(max_plogps)
                
            max_plogp_smiles = []
            for max_plogp in max_plogps:
                smiles = data['candidate_smiles'][data[f'output_{self.args.property}']==max_plogp].tolist()
                max_plogp_smiles.extend(smiles)
            if rank==0:
                self.logger.info('top 3 max plogp smiles:')
                self.logger.info(max_plogp_smiles)
            
        elif self.args.property == 'qed':
            df = data[f'output_{self.args.property}'].groupby(data['start_smiles'])
            max_qed = df.max()
            if rank==0:
                self.logger.info('top 3 max qed:')
            max_qeds = sorted(max_qed, reverse=True)[0:3]
            if rank==0:
                self.logger.info(max_qeds)
                
            max_qed_smiles = []
            for max_qed in max_qeds:
                smiles = data['candidate_smiles'][data[f'output_{self.args.property}']==max_qed].tolist()
                max_qed_smiles.extend(smiles)
            if rank==0:
                self.logger.info('top 3 max qed smiles:')
                self.logger.info(max_qed_smiles)

    def generate_candidate_selfies(self):
        self.model_engine, _, _, self.scheduler = deepspeed.initialize(self.args, model=self.model,
                                                        model_parameters=self.model.parameters())
        self.model_engine.eval()
        if rank == 0:
            self.logger.info(f'start generating with {self.args.generate_mode} ...')
        candidates = []
        with tqdm(total=len(self.input_dataloader)) as _tqdm:
            for i, batch in enumerate(self.input_dataloader):
                _tqdm.set_description(f'Generate | step [{i}/{len(self.input_dataloader)}]')
                batch_encode = self.tokenizer.batch_encode_plus(batch, max_length=self.args.max_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                if self.args.generate_mode == 'beam':
                    molecules = self.model_engine.generate(
                        input_ids=batch_encode["input_ids"].cuda(),
                        attention_mask=batch_encode["attention_mask"].cuda(),
                        num_return_sequences=self.args.beam, 
                        num_beam_groups=self.args.beam, 
                        diversity_penalty=self.args.diversity_penalty, 
                        num_beams=self.args.beam,
                        max_length=self.args.max_len,
                        min_length=self.args.min_len,
                        length_penalty=self.args.length_penalty,
                        early_stopping=True,
                        past_prompt=None
                    )
                elif self.args.generate_mode == 'topk':
                    molecules = self.model_engine.generate(
                        input_ids=batch_encode["input_ids"].cuda(),
                        attention_mask=batch_encode["attention_mask"].cuda(),
                        do_sample=True,
                        max_length=self.args.max_len,
                        min_length=self.args.min_len,
                        top_k=self.args.top_k,
                        top_p=self.args.top_p,
                        num_return_sequences=self.args.return_num,
                        past_prompt=None
                    )
                cand = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ","") for g in molecules]
                candidates.extend(cand)
                _tqdm.update(1)
        candidate_selfies = {"candidates": candidates}
        data = pd.DataFrame(candidate_selfies)
        if rank==0:
            self.logger.info(f'saving dataframe to {self.args.output_path} ...')
        data.to_csv(self.args.output_path, index=None)    
        if rank==0:
            self.logger.info('finish!')

    def preprocess(self):    
        source_data = pd.read_csv(self.args.input_path)
        candidate_data = pd.read_csv(self.args.output_path)

        if rank == 0:
            self.logger.info('start calculate properties ...')
        pandarallel.initialize(shm_size_mb=60720, nb_workers=40,progress_bar=True)
        candidate_data['candidate_smiles'] = candidate_data['candidates'].parallel_apply(sf_decode)
        
        if self.args.property == 'plogp':
            candidate_data['plogp'] = candidate_data['candidate_smiles'].parallel_apply(plogp)
        elif self.args.property == 'qed':
            candidate_data['qed'] = candidate_data['candidate_smiles'].parallel_apply(qed)
        
        property_list = sorted(set(candidate_data[self.args.property]))
        min_prop = property_list[1]
        cands = candidate_data['candidates'].tolist()
        candidates = [cands[i:i+self.args.return_num] for i in range(0,len(cands), self.args.return_num)]
        props = candidate_data[self.args.property].tolist()
        props = [min_prop-2 if prop == -100 else prop for prop in props]
        properties = [props[i:i+self.args.return_num] for i in range(0,len(props), self.args.return_num)]

        all_candidates = []
        for i in range(len(candidates)):
            pairs = []
            for j in range(self.args.return_num):
                cand = candidates[i][j] 
                prop = properties[i][j]
                pair = (cand, prop)
                pairs.append(pair)
            all_candidates.append(pairs)
            
        output = {
            "input": source_data['selfies'].tolist(),
            "candidates": all_candidates,
        }
        output_data = pd.DataFrame(output)
        if rank==0:
            self.logger.info(f'saving data to {self.args.finetune_path} ...')
        output_data.to_csv(self.args.finetune_path, index=None)




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
        debug, info = logger.debug, logger.info
        if not cfgs.no_tensorboard and not cfgs.only_test:
            writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', cfgs.time_stamp), comment=comment)

    cfgs.device = torch.device(cfgs.device)
    
    # -----  Begin ----------
    torch.cuda.set_device(cfgs.gpu)
    runner = Runner(cfgs, writer, logger, rank)
    if cfgs.process == 'generate':
        runner.generate_molecules()
        runner.statistics()
    elif cfgs.process == 'preprocess':
        runner.generate_candidate_selfies()
        runner.preprocess()

        
    # -----  End ----------
    if not cfgs.no_tensorboard and not cfgs.only_test and rank == 0:
        writer.close()
        logger.info("done!")
    
    if cfgs.dist and not cfgs.only_test:
        dist.barrier()
        dist.destroy_process_group()

