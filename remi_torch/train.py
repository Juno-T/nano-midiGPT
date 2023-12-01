from calendar import c
import time
from typing import Literal
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
device_type = 'cuda' if 'cuda' in device else device

def init_train(init_from: Literal['scratch', 'resume'], ckpt_path: Path = None):
    if init_from == 'scratch':    
        tokenizer = REMI(params=Path("./remi_tok.json"))

        model_args = dict(
            block_size=16,
            vocab_size=len(tokenizer.vocab),
            n_layer=1,
            n_head=4,
            n_embd=32,
            dropout=0.2,
            bias=False,
        )
        optimizer_args = dict(
            weight_decay=1e-1,
            learning_rate=6e-4,
            beta_1_2=(0.9,0.95),
        )
        train_args = {
            'batch_size': 16,
            'num_workers': 0,
            'shuffle': True,
            'model_args': model_args,
            'optimizer_args': optimizer_args,
        }
        model_state_dict = None
        optimizer_state_dict = None
    elif init_from == 'resume':
        ckpt_path = Path(ckpt_path)
        assert ckpt_path is not None and ckpt_path.is_file(), f"{ckpt_path} is not a file"
        ckpt = torch.load(str(ckpt_path))
        model_state_dict = ckpt['model']
        optimizer_state_dict = ckpt['optimizer']
        model_args = ckpt['model_args']
        optimizer_args = ckpt['optimizer_args']
        train_args = ckpt['train_args']
        tokenizer = REMI(params=(Path(ckpt_path).parent / ckpt['remi_rel_path']))
    else:
        raise ValueError(f"init_from {init_from} not recognized")
    return model_args, model_state_dict, optimizer_args, optimizer_state_dict, train_args, tokenizer

def train(
    ckpt_path,
    log_interval = 1,
    max_iters = 1,
    resume_path = None,
):
    ckpt_path = Path(ckpt_path)
    ckpt_dir = ckpt_path.parent
    init_from = 'scratch' if resume_path is None else 'resume'
    model_args, model_state_dict, optimizer_args, optimizer_state_dict, train_args, tokenizer = init_train(init_from, resume_path)

    model = GPT(GPTConfig(**model_args)).to(device)
    optimizer = model.configure_optimizers(
        optimizer_args['weight_decay'],
        optimizer_args['learning_rate'],
        optimizer_args['beta_1_2'],
        device_type
    )
    if init_from == 'resume':
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        # free up memory
        del model_state_dict
        del optimizer_state_dict

    dataset = TokenizedDataset(Path("../data/REMI_pop/train_tokens_noBPE"), tokenizer, jitter=True, device=device, split='train')
    train_dataloader = DataLoader(dataset, batch_size=train_args['batch_size'], shuffle=train_args['shuffle'], num_workers=train_args['num_workers'])

    train_args['iter_num'] = train_args.get('iter_num', 0)
    iter_num = train_args['iter_num']
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


        # if losses['val'] < best_val_loss or always_save_checkpoint:
            # best_val_loss = losses['val']
        if iter_num > 0:
            train_args['iter_num'] = iter_num
            # save remi
            remi_save_path = ckpt_dir / "remi_tokenizer.json"
            remi_save_path_rel = remi_save_path.relative_to(ckpt_dir)
            tokenizer.save_params(remi_save_path)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'optimizer_args': optimizer_args,
                'train_args': train_args,
                'remi_rel_path': remi_save_path_rel,
                # 'best_val_loss': best_val_loss,
                # 'config': config,
            }
            print(f"saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, str(ckpt_path))

        # termination conditions
        if iter_num > max_iters:
            break

if __name__ == "__main__":
    train(
        log_interval = 1,
        max_iters = 1,
        ckpt_path = Path("./checkpoint/test/ckpt2.pt"),
        resume_path = Path("./checkpoint/test/ckpt.pt"),
    )