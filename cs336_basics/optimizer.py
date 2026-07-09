from collections.abc import Callable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")            
        # defaults 定义超参
        defaults = {"lr": lr}
        
        # self.param_groups 是一个列表，包含了所有的参数组，每个参数组是一个字典，包含了该组的参数和超参
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        # 遍历所有的参数组
        for group in self.param_groups:
            lr = group["lr"] # get the learning rate
            # 遍历该组的所有参数
            for p in group["params"]:
                if p.grad is None:
                    continue

                # state 是一个 defaultdict(dict) 字典，第一次访问会自动创建一个空字典
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay): 
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")            
            
        defaults = {
            "lr": lr, 
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            lr_t = lr
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue 
                grad = p.grad.data
                state = self.state[p]
                
                # 1.第一次访问初始化
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                    # get 方法并不能存储值，只能不存在时返回值
                    # t = state.get("t", 0)
                    # m = state.get("m", torch.zeros_like(p.data))
                    # v = state.get("v", torch.zeros_like(p.data))
                    
                # 2.推进步数 
                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]
                
                # 3.用当前步数更新一阶/二阶矩 (in-place)
                m.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

                # m.mul_(beta[0]).add_(grad, alpha=1 - beta[0]) 是 in-place 链式。 
                # 等价于 m = beta[0] * m + (1 - beta[0]) * grad，但不创建新张量——这对优化器很重要，否则每步都分配 m.numel() 大小的临时显存。 
                
                # 4. bias-corrected lr
                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                
                # 5. 用更新后的 m，v 更新参数
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)

                # 6. AdamW
                p.data.mul_(1 - lr * weight_decay)
                
        return loss
        
        
                
# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1) 
#
# for t in range(10):
#     opt.zero_grad()
#     loss = (weights**2).mean()
#     print(loss.cpu().item())
#     loss.backward()
#     opt.step()

def lr_schedule(t, lr_max, lr_min, t_w, t_c):
    lr_t = 0
    # warm-up
    if t < t_w:
        lr_t = t * lr_max / t_w 
    # cosine annealing
    if t_w <= t <= t_c:
        lr_t = lr_min + (1 + math.cos((t - t_w) * math.pi / (t_c - t_w))) / 2 * (lr_max - lr_min)
    # Post-annealing
    if t > t_c:
        lr_t = lr_min

    return lr_t
    
    
