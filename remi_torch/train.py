import time
from tqdm import tqdm
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from miditok import REMI, TokenizerConfig

from remi_torch.data.dataset import TokenizedDataset
from remi_torch.model.nanoGPT import GPT, GPTConfig

np.random.seed(42)
torch.random.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.cuda.is_available() else 'cpu'


weight_decay=1e-1
learning_rate=6e-4
(beta1, beta2)=(0.9,0.95)
device_type = 'cuda' if 'cuda' in device else device

tokenizer = REMI(params=Path("./remi_tok.json"))
config = GPTConfig(
    block_size=16,
    vocab_size=len(tokenizer.vocab),
    n_layer=1,
    n_head=4,
    n_embd=32,
    dropout=0.2,
    bias=False,
)
model = GPT(config).to(device)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

batch_size = 16

dataset = TokenizedDataset(Path("../data/REMI_pop/train_tokens_noBPE"), tokenizer, jitter=True, device=device, split='train')
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


log_interval = 1000
max_iters = 10000
iter_num = 0
t0 = time.time()
for X, Y in train_dataloader:
    logits, loss = model(X, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() # loss as float. TODO note CPU-GPU sync! profile, make sure not too slow
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break