from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch.distributions import Multinomial
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'running on {device}')

# -----------------Model Struction---------------------
@dataclass
class ModelArgs:
    # Model struct params
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: Optional[int] = None # for Grouped-query attention 
    vocab_size: int = None  # set in data preparation stage
    norm_eps: float = 1e-5

    #Training params
    batch_size:int = 32
    seq_size:int = 16
    total_iters:int = 2500
    iters_interval:int = 100
    eval_iters:int = 100
    learning_rate:float = 1e-4

    #Inference params
    inference:bool = None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, theta: float = 10000.0):
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return torch.repeat_interleave(x, 2, dim=2)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.k_cache = torch.empty(0) # only for one promts at a time, batch_size==1
        self.v_cache = torch.empty(0)

    def forward(
        self,
        x: torch.Tensor,
        freqs_complex: torch.Tensor,
        start_pos:int,
        ):
        batch_size, seq_len, _ = x.shape  # (B, Seq_Q, Dim)

        # (B, Seq_Q, Dim) -> (B, Seq_Q, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, Seq_Q, Dim) -> (B, Seq_Q, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, Seq_Q, Dim) -> (B, Seq_Q, H_KV * Head_Dim)
        xv = self.wv(x)

        # (B, Seq_Q, H_Q * Head_Dim) -> (B, Seq_Q, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, Seq_Q, H_KV * Head_Dim) -> (B, Seq_Q, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, Seq_Q, H_KV * Head_Dim) -> (B, Seq_Q, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (B, Seq_Q, H_Q, Head_Dim) --> (B, Seq_Q, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex)
        # (B, Seq_Q, H_KV, Head_Dim) --> (B, Seq_Q, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex)

        # For inference
        if self.args.inference:
            xk = torch.cat([xk, self.k_cache], dim=1) if self.k_cache.numel() else xk
            xv = torch.cat([xv, self.v_cache], dim=1) if self.v_cache.numel() else xv

        # (B, Seq_KV, H_KV, Head_Dim) --> (B, Seq_KV, H_Q, Head_Dim)
        keys = repeat_kv(xk, self.n_rep)
        # (B, Seq_KV, H_KV, Head_Dim) --> (B, Seq_KV, H_Q, Head_Dim)
        values = repeat_kv(xv, self.n_rep)

        scores = torch.einsum('imkl, inkl -> ikmn', xq, keys) / math.sqrt(self.head_dim)
        # (B, H_Q, Seq_Q, Seq_KV) -> (B, H_Q, Seq_Q, Seq_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, Seq_Q, Seq_KV) @ (B, H_Q, Seq_KV, Head_Dim) -> (B, H_Q, Seq_Q, Head_Dim)
        output = torch.einsum('itkl,imtn->itkn',scores, values)
        # (B, H_Q, Seq_Q, Head_Dim) -> (B, Seq_Q, H_Q, Head_Dim) -> (B, Seq_Q, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, Seq_Q, Dim) -> (B, Seq_Q, Dim)

class FeedForward(nn.Module):
    def __init__(
      self,
      args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.dim

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, start_pos:int):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(
          self.attention_norm(x), freqs_complex, start_pos
        )
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(self.n_layers)])

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = None

    def forward(self, tokens: torch.Tensor, start_pos:int=None):
        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the start_pos positions
        freqs_complex = self.freqs_complex[start_pos:start_pos + 1, :] if args.inference else self.freqs_complex

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, freqs_complex, start_pos)
        h = self.norm(h)
        output = self.output(h).float()

        return output

    def trainer(self):
        args.inference = False
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.seq_size)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        total_iter = tqdm(range(self.args.total_iters))

        for i in total_iter:
            x, y = get_batch('train')
            y_pre = self(x)
            B, S, C = y_pre.shape
            loss = F.cross_entropy(y_pre.view(B*S, C), y.view(B*S))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % self.args.iters_interval == 0 or i == self.args.total_iters - 1:
                losses = estimate_loss()
                print(f"The traning loss:{losses['train']:.2f}, val_loss: {losses['val']:.2f}")

    @torch.no_grad()
    def generater(self, context:str, max_token:int=128):
        args.inference = True
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, max_token)
        context_toekns = encode(context)
        context_len = len(context_toekns)
        out_token = None
        generated_toekns = []

        for token_pos in range(max_token):
            if token_pos < context_len:
                next_token = torch.tensor(context_toekns[token_pos]).view(1,1)
            else:
                next_token = out_token
            out_logit = self(next_token.to(device), token_pos)
            out_token = Multinomial(logits=out_logit).sample().argmax(dim=-1)
            generated_toekns.append(out_token.item())
        text = decode(context_toekns + generated_toekns[context_len:])
        print(text)
        


# -----------------Data Preparation---------------------
args = ModelArgs()
with open('shakspere.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
args.vocab_size = vocab_size
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - args.seq_size, (args.batch_size,))
    x = torch.stack([data[i:i+args.seq_size] for i in ix])
    y = torch.stack([data[i+1:i+args.seq_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            Y_pre = model(X)
            B, S, C = Y_pre.shape
            loss = F.cross_entropy(Y_pre.view(B*S, C), Y.view(B*S))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = Transformer(args).to(device)
model.trainer()
model.generater('You', 128)