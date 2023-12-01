import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    def __init__(
            self,
            tok_dir,
            tokenizer,
            seq_len=8,
            jitter=False, # jitter the start index of each sequence for data variation
            device='cpu',
            store_as_cpu=True,
            split='train'
        ):
        self.tok_dir = Path(tok_dir)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.jitter = jitter
        self.device = device
        self.store_as_cpu = store_as_cpu

        self.tok_paths = list(self.tok_dir.glob("**/*.json"))
        if split == 'train':
            self.tok_paths = self.tok_paths[:int(len(self.tok_paths)*0.8)]
        else:
            self.tok_paths = self.tok_paths[int(len(self.tok_paths)*0.8):]
        self.tokens = []
        for path in tqdm(self.tok_paths):
            assert path.is_file(), f"{path} is not a file"
            with open(path) as json_file:
                token_json = json.load(json_file)
            assert len(token_json['ids']) == 1, f"Not support more than one sequence/track for {path}"
            tokens = add_bos_eos(token_json['ids'][0], tokenizer['BOS_None'], tokenizer['EOS_None'])
            self.tokens.extend(tokens)
        
        if not self.store_as_cpu:
            self.tokens = torch.tensor(self.tokens, dtype=torch.int64).to(self.device)
        else:
            # store as numpy
            self.tokens = np.array(self.tokens, dtype = np.int64)
        
        
        self.num_seq = self.tokens.shape[0] // self.seq_len
        # Throw away the last few tokens if they don't fit into a sequence

    def __len__(self):
        return self.num_seq

    def __getitem__(self, idx):
        start_idx = idx*self.seq_len
        if self.jitter:
            jitter_range = self.seq_len//2
            jitter_start = torch.randint(
                max(start_idx-jitter_range, 0), 
                min(start_idx+jitter_range, len(self.tokens)-self.seq_len-1),
                (1,)).item()
            start_idx = jitter_start

        x = self.tokens[start_idx:start_idx+self.seq_len]
        y = self.tokens[start_idx+1:start_idx+self.seq_len+1]
        if self.store_as_cpu:
            # numpy => torch => device
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            if self.device == 'cuda':
                x = x.pin_memory().to_device('cuda', non_blocking=True)
                y = y.pin_memory().to_device('cuda', non_blocking=True)
            else:
                x = x.to(self.device)
                y = y.to(self.device)
        return x, y

def add_bos_eos(tokens, bos_token, eos_token):
    tokens = [bos_token] + tokens + [eos_token]
    return tokens