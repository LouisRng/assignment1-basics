import torch
import numpy as np
import os
import numpy.typing as npt
from datetime import datetime
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import timeit 
from cs336_basics.dataloader import get_batch, save_checkpoint, load_checkpoint
from cs336_basics.transformer import TransformerLM, cross_entropy, softmax
from cs336_basics.optimizer import AdamW, gradient_clipping, lr_schedule 
from cs336_basics.tokenizer import BPETokenizer

# args 
def parse_args(): 
    parser = argparse.ArgumentParser()
    # 数据集路径
    parser.add_argument("--train_data", type=str, required=False)
    parser.add_argument("--val_data", type=str, required=False)
    
    # 模型超参：d_model, num_layers, num_heads, d_ff, vocab_size, context_length, rope_theta
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=768)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # 优化器超参：lr, betas, weight_decay, eps
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9,0.999])
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-8)
    
    # 训练超参：batch_size, num_steps, grad_clip
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # LR schedule：warmup_steps, cosine_decay_end (or final_lr)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--cos_steps", type=int, default=2500)
    parser.add_argument("--final_lr", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=1e-2)
    
    # IO：train_data, val_data, ckpt_dir, log_interval, val_interval, ckpt_interval
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    
    # 其他：seed, device, resume_from
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--n_eval_batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--generate", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_token_len", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--merges_path", type=str, default=None)

    # benchmarking
    parser.add_argument("--bm_mode", type=bool, default=False)
    parser.add_argument("--forward", type=bool, default=False)
    parser.add_argument("--backward", type=bool, default=False)
    parser.add_argument("--optim", type=bool, default=False)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num_bm", type=int, default=10) 
    
    return parser.parse_args()

def set_seed(seed: int):
    torch.manual_seed(seed)

# file path
def load_data(path: str | os.PathLike) -> npt.NDArray:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    return data

@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, device, n_eval_batches):
    model.eval()
    total_loss = 0
    for _ in range(n_eval_batches):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x) 
        loss = cross_entropy(logits, y)
        total_loss += loss.item() 
    model.train()
    return total_loss / n_eval_batches
     
@torch.no_grad()
def decoding(
    model, 
    tokenizer, 
    prompt: str, 
    max_token_len: int, 
    temperature: float, 
    p: float,
    device,
    context_length: int
):
    model.eval()

    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # 输入需要有 batch 维度

    # 尝试找 EOS token id
    try:
        eot_id = tokenizer.reversed_vocab[b"<|endoftext|>"]
    except KeyError:
        eot_id = None 

    generated_ids = []
    
    for _ in range(max_token_len):
        # 截断到 context_length，生产中是用 KV Cache
        if ids.shape[1] > context_length:
            input_ids = ids[:, -context_length:]
        else:
            input_ids = ids # (1, s, v)
            
        logits = model(input_ids)[0, -1, :]     # (v,)
        
        probs = softmax(logits, temperature=temperature, dim=-1)
        
        # top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True) 
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # 找到第一个 cumulative >= p 的位置，保留到这里
        # nucleus 包括这个位置（"smallest set such that sum >= p"）
        # 求第一个满足的下标，nonzero 返回非 0 的坐标
        # item() 会返回 Number 基类，不能直接用
        cutoff = (cumulative >= p).nonzero()[0].item()          
        nucleus_probs = sorted_probs[:cutoff + 1]
        nucleus_indices = sorted_indices[:cutoff + 1]
        
        nucleus_probs= nucleus_probs / nucleus_probs.sum()
        sampled_pos = torch.multinomial(nucleus_probs, num_samples=1)
        next_id = nucleus_indices[sampled_pos].item()

        # endoftext 切断
        if eot_id is not None and next_id == eot_id:
            break
        
        # 不要把生成的 token 转换成 str 再拼回去重新 encode，效率非常低下
        # 直接把生成的 token 拼回 encode 后的 ids
        generated_ids.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1) # (1, s)，沿着 seq_len 维度拼接
        
    output_text = tokenizer.decode(generated_ids)
    print("Prompt:", prompt)
    print("Generation:", output_text)
    
    model.train()
    return output_text
    

def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else \
            "mps" if torch.backends.mps.is_available() else "cpu"
    
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)
    
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = TransformerLM(
        vocab_size=args.vocab_size, 
        context_length=args.context_length, 
        num_layers=args.num_layers, 
        d_model=args.d_model, 
        num_heads=args.num_heads, 
        d_ff=args.d_ff, 
        rope_theta=args.rope_theta
    ).to(device)
    
    optimizer = AdamW(
        params=model.parameters(), 
        betas=args.betas, 
        weight_decay=args.weight_decay, 
        lr=args.lr, 
        eps=args.eps
    )
    
    start_step = 0
    if args.resume_from:
        ckpt_path = ckpt_dir / args.resume_from
        start_step = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resume from step {start_step}")

    
    model.train()

    forward_start = 0
    forward_elapsed = 0
    backward_start = 0
    backward_elapsed = 0
    optim_start = 0
    optim_elapsed = 0
   

    for step in range(start_step, args.num_steps):
        t_step_start = time.time()
        
        lr = lr_schedule(step, args.lr_max, args.final_lr, args.warmup_steps, args.cos_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if args.bm_mode:
            print("benchmarking mode ... ")
            x = torch.from_numpy(
                np.stack(
                    [np.random.randint(0, args.vocab_size, args.context_length) for _ in range(args.batch_size)]
                )
            ).to(device)
            y = torch.from_numpy(
                np.stack(
                    [np.random.randint(0, args.vocab_size, args.context_length) for _ in range(args.batch_size)]
                )
            ).to(device)
        else:
            x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        
        if args.backward == True and args.warmup < step < args.num_bm + args.warmup:
            optim_start = timeit.default_timer()
        
        if args.backward == True and args.warmup < step < args.num_bm + args.warmup:
            backward_start = timeit.default_timer()
            
        if args.forward == True and args.warmup < step < args.num_bm + args.warmup:
            forward_start = timeit.default_timer() 
            
        logits = model(x)
        
        if args.forward == True and args.warmup < step < args.num_bm + args.warmup:
            forward_elapsed += timeit.default_timer() - forward_start
        
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        
        if args.backward == True and args.warmup < step < args.num_bm + args.warmup:
            forward_elapsed += timeit.default_timer() - backward_start
        
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
    
        optimizer.step()
        
        if args.optim == True and args.warmup < step < args.num_bm + args.warmup:
            optim_elapsed += timeit.default_timer() - optim_start

        if step % args.log_interval == 0:
            elapsed = time.time() - t_step_start
            tokens_per_sec = args.batch_size * args.context_length / elapsed
            print(f"step {step:6d}  loss {loss.item():.4f}  lr {lr:.2e}  tok/s {tokens_per_sec:,.0f}")

        if step % args.val_interval == 0 and step > 0:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device, args.n_eval_batches)
            print(f"steps {step} val loss {val_loss:.4f}")

        if step % args.ckpt_interval == 0 and step > 0:
            ckpt_path = ckpt_dir / f"{step}"
            save_checkpoint(model, optimizer, step, ckpt_path)

    if args.forward:
        print(f"Forward one step costs {forward_elapsed / args.num_bm: .4f} seconds")
    if args.backward:
        print(f"Forward and backward without counting optimizer one step costs {backward_elapsed / args.num_bm: .4f} seconds")
    if args.optim:
        print(f"Forward and backward with counting optimizer one step costs {optim_elapsed / args.num_bm: .4f} seconds")
            
    tokenizer = BPETokenizer.from_files(args.vocab_path, args.merges_path)
 
    decoding(model, tokenizer, args.generate, args.max_token_len, args.temperature, args.p, device=device, context_length=50)
            
    save_checkpoint(model, optimizer, args.num_steps, ckpt_dir / "final.pt")

if __name__ == "__main__":
    main()











    
    
