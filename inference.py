from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLaMA:
    
    def __init__(self, model:Transformer, tokenizer:SentencePieceProcessor, model_args:ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
        
    @staticmethod
    def build(checkpoint_dir:str, tokenizer_path:str, load_model:bool, max_seq_size:int, max_batch_size:int, device:str):
        prev_time = time.time()
        
        if load_model:
            checkpoints = sorted(Path(checkpoint_dir).glob('*.pth'))
            assert len(checkpoints) > 0, f"no checkpoint file found in {checkpoint_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location='cuda')
            print(f'Loaded checkpoint in {time.time() - prev_time:.2f}s')
            prev_time = time.time()
            
        with open(Path("params.json") , "r") as f:
            params = json.loads(f.read())
            
        model_args:ModelArgs = ModelArgs(
            max_seq_len=max_seq_size,
            max_batch_size=max_batch_size,
            device=device,
            **params)
            
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        model = Transformer(model_args).to(device)
        
      # if load_model:
        del checkpoint['rope.freqs']
        model.load_state_dict(checkpoint, strict=True)
        print(f'Loaded state dict in {time.time() - prev_time:.2f}s')
          
        return LLaMA(model, tokenizer, model_args)
    def text_completion(self, prompts:list[str], max_gen_len:Optional[int]=None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
            
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch_size must be less than or equal to {self.args.max_batch_size}"
        
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id
        cur_iterator = tqdm(range(1, total_len), desc='Generating tokens')
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1, cur_pos], cur_pos)
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= (~prompt_tokens_mask[:, cur_pos] & (next_token == self.tokenizer.eos_id))
            if all(eos_reached):
                break
                
        out_tokens = []
        out_text = []
        for prempt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)