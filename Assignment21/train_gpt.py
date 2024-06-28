from statistics import mean
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
from dataclasses import dataclass
import time

class CasualSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd%config.n_head==0
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT=1
        self.n_embd=config.n_embd
        self.head=config.n_head
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    def forward(self,x):
        B,T,C=x.size()
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        k=k.view(B,T,self.head,C//self.head).transpose(1,2)#(B,nh,T,C)
        q=q.view(B,T,self.head,C//self.head).transpose(1,2)#(B,nh,T,C)
        v=v.view(B,T,self.head,C//self.head).transpose(1,2)
        # attn=(q@k.transpose(-2,-1))*(1/math.sqrt(k.size(-1)))
        # attn=attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # attn=F.softmax(attn,dim=-1)
        #y=attn@v# B,nh,T,T * B,nh,T,C =B,nh,T,C
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.c_proj(y)
        return y
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT=1
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attention=CasualSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)
    def forward(self,x):
        x = x+self.attention(self.ln_1(x))
        x = x+self.mlp(self.ln_2(x))
        return x
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f= nn.LayerNorm(config.n_embd) ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size)
        self.transformer.wte.weight=self.lm_head.weight
        self.apply(self.__init_weights)
    
    def __init_weights(self,module):
        if(isinstance(module,nn.Linear)):
            std=0.02
            if(hasattr(module,"NANGPT_SCALE_INIT")):
                std*=(2*self.config.n_layer)** -0.5
            torch.nn.init.normal_(module.weight,mean=0,std=std)
        elif(isinstance(module,nn.Embedding)):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)


        
    def forward(self,x,target=None):
        B,T=x.size()
        assert T<=self.config.block_size
        tok_embd=self.transformer.wte(x)
        pos=torch.arange(0,T,dtype=torch.long,device=x.device)
        pos_embd=self.transformer.wpe(pos)
        x=tok_embd+pos_embd
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        loss=None
        if target is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1))
        return logits,loss
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
class DataLoaderLite():
    def __init__(self,B,T):
        self.B=B
        self.T=T
        self.current_position=0
        with open("input.txt","r") as f:
            text=f.read()
        enc=tiktoken.get_encoding("gpt2")
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')
    def get_next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_position:self.current_position+ B*T +1]
        x=buf[:-1].view(B,T)
        y=buf[1:].view(B,T)
        self.current_position+=B*T
        if(self.current_position+B*T+1>=len(self.tokens)):
            self.current_position=0
        return x,y
torch.manual_seed(42)

dataloader=DataLoaderLite(B=8,T=256)
torch.set_float32_matmul_precision('high')
max_lr = 6e-4 
min_lr = max_lr * 0.1
warmup_steps = 50
max_steps = 5000

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

device=torch.device("mps")
model = GPT(GPTConfig())
model=model.to(device)
# model=torch.compile(model)
#optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
for step in range(5000):
    x,y=dataloader.get_next_batch()
    x,y=x.to(device),y.to(device)
    t1=time.time()
    optimizer.zero_grad()
    logit,loss=model(x,y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.mps.synchronize()
    t2=time.time()
    dt=(t2-t1)*1000
    tokens_per_sec=dataloader.B*dataloader.T/(t2-t1)
    print(f'Step {step} | loss: {loss.item()} | norm: {norm} | lr: {lr} |dt: {dt}ms | token: {tokens_per_sec}/sec')
torch.save(model.state_dict(), "./gpt.pt")
torch.save(model.state_dict(), "./gpt.pth")


# enc=tiktoken.get_encoding('gpt2')
# tokens=enc.encode("Hello Iam Ironman")
# tokens=torch.tensor(tokens,dtype=torch.long)
# tokens=tokens.unsqueeze(0).repeat(5,1)
# x = tokens.to(device)
# while x.size(1) < 30:
#     # forward the model to get the logits
#     with torch.no_grad():
#         logits = model(x)[0] # (B, T, vocab_size)
#         # take the logits at the last position
#         logits = logits[:, -1, :] # (B, vocab_size)
#         # get the probabilities
#         probs = F.softmax(logits, dim=-1)
#         # do top-k sampling of 50 (huggingface pipeline default)
#         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         # select a token from the top-k probabilities
#         # note: multinomial does not demand the input to sum to 1
#         ix = torch.multinomial(topk_probs, 1) # (B, 1)
#         # gather the corresponding indices
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#         # append to the sequence
#         x = torch.cat((x, xcol), dim=1)

# for i in range(5):
#     tokens = x[i, :30].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)

