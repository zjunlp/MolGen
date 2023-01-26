import torch
import numpy as np
import pandas as pd
from numpy.random import permutation, poisson
from typing import Dict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import math
from dataclasses import dataclass


class CandidateDataset(torch.utils.data.Dataset):
    def __init__(self,
                tokenizer, finetune_path, max_len=100, is_sorted=True
                ):
        self.tokenizer = tokenizer
        self.finetune_path = finetune_path
        self.data = pd.read_csv(self.finetune_path)
        self.maxlen = max_len
        self.sorted = is_sorted
        self.inputs = self.data['input'].tolist()
        self.candidate_inputs = self.data['candidates'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input = self.inputs[index]
        src = self.tokenizer.batch_encode_plus([input], max_length=self.maxlen, return_tensors="pt", padding="max_length", truncation=True)
        src_input_ids = src["input_ids"]
        src_input_ids = src_input_ids.squeeze(0)
            
        candidate = self.candidate_inputs[index]
        candidate = eval(candidate)
        
        if self.sorted:
            candidate = sorted(candidate, key=lambda x:x[1], reverse=True)
        candidates = [x[0] for x in candidate]
        cand = self.tokenizer.batch_encode_plus(candidates, max_length=self.maxlen, return_tensors="pt", padding="max_length", truncation=True)
        candidate_ids = cand["input_ids"]
        result = {
            "src_input_ids": src_input_ids, 
            "candidate_ids": candidate_ids,
            }
        return result
    

@dataclass
class DataCollatorForFinetuneTasks:

    tokenizer: PreTrainedTokenizerBase
    mask_ratio: float = 0.3
    poisson_lambda: float = 3.0

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError

    def __call__(self, batch) -> Dict[str, np.ndarray]:
        """Batching, adding whole word mask and permutate sentences
        Args:
            examples (dict): list of examples each examples contains input_ids field
        """
        # Handle dict or lists with proper padding and conversion to tensor.
        def pad(X, max_len=-1):
            if max_len < 0:
                max_len = max(x.size(0) for x in X)
            result = torch.ones(len(X), max_len, dtype=X[0].dtype) * self.tokenizer.pad_token_id
            for (i, x) in enumerate(X):
                result[i, :x.size(0)] = x
            return result
        
        max_len = batch[0]["src_input_ids"].size(0)
        src_input_ids = pad([x["src_input_ids"] for x in batch], max_len)
        candidate_ids = [x["candidate_ids"] for x in batch]
        candidate_ids = [pad(x, max_len) for x in candidate_ids]
        candidate_ids = torch.stack(candidate_ids)

        decoder_input_ids = self.shift_tokens_right(src_input_ids)
        do_permutate = False
        if self.mask_ratio:
            input_ids, labels = self.add_whole_word_mask(np.array(src_input_ids), do_permutate)
        input_ids = torch.LongTensor(input_ids)
        decoder_input_ids = torch.LongTensor(decoder_input_ids)
        labels = torch.LongTensor(labels)
        result = {
            "src_input_ids": src_input_ids,
            "candidate_ids": candidate_ids,
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            }
        return result     
    
    def shift_tokens_right(self, inputs):
        """Shift decoder input ids right: https://github.com/huggingface/transformers/issues/7961.
        Examples:
            <s>My dog is cute.</s><s>It loves to play in the park.</s><pad><pad>
            shift to -> </s><s>My dog is cute.</s><s>It loves to play in the park.<pad><pad>
        """

        shifted_inputs = np.roll(inputs, 1, axis=-1)

        # replace first token with eos token
        shifted_inputs[:, 0] = self.tokenizer.eos_token_id

        # when there's padding, the last eos tokens will not be rotate to first positon
        # we'll need to replace it with a padding token

        # replace eos tokens at the end of sequences with pad tokens
        end_with_eos = np.where(shifted_inputs[:, -1] == self.tokenizer.eos_token_id)
        shifted_inputs[end_with_eos, -1] = self.tokenizer.pad_token_id

        # find positions where where's the token is eos and its follwing token is a padding token
        last_eos_indices = np.where(
            (shifted_inputs[:, :-1] == self.tokenizer.eos_token_id)
            * (shifted_inputs[:, 1:] == self.tokenizer.pad_token_id)
        )

        # replace eos tokens with pad token
        shifted_inputs[last_eos_indices] = self.tokenizer.pad_token_id
        return shifted_inputs   
        
    def add_whole_word_mask(self, inputs, do_permutate):
        labels = inputs.copy()

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = np.array(special_tokens_mask, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token = ~(labels == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.astype(float).sum() * self.mask_ratio))
        if num_to_mask == 0:
            return inputs, labels

        # generate a sufficient number of span lengths
        lengths = poisson(lam=self.poisson_lambda, size=(num_to_mask,))
        while np.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = np.concatenate([lengths, poisson(lam=self.poisson_lambda, size=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = np.argmin(np.abs(np.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[: idx + 1]

        # select span start indices
        token_indices = np.argwhere(is_token == 1)
        span_starts = permutation(token_indices.shape[0])[: lengths.shape[0]]

        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        mask = np.full_like(labels, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = labels.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask)] = False
        inputs[np.where(mask)] = self.tokenizer.mask_token_id

        if not do_permutate:
            labels[np.where(mask)] = -100
        else:
            labels[np.where(special_tokens_mask)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_inputs = np.full_like(labels, fill_value=self.tokenizer.pad_token_id)

        # splits = list(map(lambda x: x.reshape(-1),  np.split(inputs_copy, indices_or_sections=2, axis=0))
        for i, example in enumerate(np.split(inputs, indices_or_sections=new_inputs.shape[0], axis=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0 : new_example.shape[0]] = new_example

        # batching now fixed
        return new_inputs, labels
    
    