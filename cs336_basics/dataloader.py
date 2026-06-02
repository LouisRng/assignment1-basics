import numpy as np
import numpy.typing as npt
import torch
import typing
import os
from pathlib import Path

CHECKPOINT = (Path(__file__).parent / ".." / "checkpoint").resolve()

MODEL_STATE = CHECKPOINT / ""

def get_batch(
    data: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(data)
    ix = np.random.randint(0, n - context_length, size=batch_size)
    x = torch.stack([
        torch.from_numpy((data[i : i + context_length]).astype(np.int64)) 
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1 : i + 1 + context_length]).astype(np.int64)) 
        for i in ix
    ])
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    '''dump all the state from the first three parameters into the file-like object out
    args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        iteration: int
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    '''
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)
    
def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    '''load a checkpoint from src, and then recover the model and optimizer states from that checkpoint.
    args:
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
    '''
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration
