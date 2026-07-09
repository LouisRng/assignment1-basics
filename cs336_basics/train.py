import torch
import json
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
from cs336_basics.model import BasicTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax, clip_gradient
from cs336_basics.optimizer import AdamW, lr_schedule  
from cs336_basics.tokenizer import BPETokenizer

# args 
def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--eos_token_id", type=int, default=256) # 256 -> <|endoftext|>
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--from_pretrained", type=str, required=False)
    parser.add_argument("--save_model", type=bool, default=False)
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

    model = BasicTransformerLM(
        vocab_size=args.vocab_size, 
        context_length=args.context_length, 
        d_model=args.d_model, 
        num_layers=args.num_layers, 
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

    
    if args.train:
        model.train()

    forward_start = 0
    forward_elapsed = 0
    backward_start = 0
    backward_elapsed = 0
    optim_start = 0
    optim_elapsed = 0
   

    if args.train:
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
                clip_gradient(model.parameters(), args.grad_clip)
        
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

        if args.save_model:
            model_config = Path("checkpoints/model_config.json")
            model_weights = Path("checkpoints/model.pt") 
            model_weights.parent.mkdir(parents=True, exist_ok=True)
            with open(model_config, "w") as f:
                json.dump(model.config, f, indent=4)
            torch.save(model.state_dict(), model_weights)
            print(f"model config saved to {model_config.resolve()}\nmodel weights saved to {model_weights.resolve()}")

        if args.forward:
            print(f"Forward one step costs {forward_elapsed / args.num_bm: .4f} seconds")
        if args.backward:
            print(f"Forward and backward without counting optimizer one step costs {backward_elapsed / args.num_bm: .4f} seconds")
        if args.optim:
            print(f"Forward and backward with counting optimizer one step costs {optim_elapsed / args.num_bm: .4f} seconds")
            
    if args.from_pretrained:
        model.eval()
        model = model.from_pretrained(args.from_pretrained)
            
    tokenizer = BPETokenizer.from_files(args.vocab_path, args.merges_path)
 
    if args.generate:
        print(f"Prompt: {args.generate}")
        ids = tokenizer.encode(args.generate)
        ids = torch.tensor(ids, dtype=torch.long)  
        new_ids = model.generate(ids, args.max_token_len, args.temperature, top_k=None, top_p=args.p, eos_token_id=args.eos_token_id)
        print(f"new_ids.shape: {new_ids.shape}")
        generation = tokenizer.decode(new_ids.tolist())
        print(f"Generation: {generation}")
            
    save_checkpoint(model, optimizer, args.num_steps, ckpt_dir / "final.pt")

if __name__ == "__main__":
    main()











    
    
