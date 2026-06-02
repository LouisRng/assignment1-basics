"""
对 train_bpe 做 cProfile 性能分析。
用法:
    python profile_tokenizer.py
分析结果:
    python -m pstats tokenizer.prof       # 交互式
    python -c "import pstats; p=pstats.Stats('tokenizer.prof'); p.sort_stats('cumulative'); p.print_stats(30)"
可视化 (需安装 snakeviz):
    pip install snakeviz
    snakeviz tokenizer.prof
"""

import time
import shutil
import json
import cProfile
import pstats
import io
import os
import tempfile
import numpy as np
from multiprocessing import Pool
from cs336_basics.tokenizer import train_bpe, BPETokenizer, find_chunk_boundaries
from pathlib import Path

num_processes = os.cpu_count() or 1

VOCAB_NAME = "owt_bpe_vocab.json"
MERGES_NAME = "owt_bpe_merge.json"

DATA_DIR = (Path(__file__).parent / ".." / "data").resolve()
TRAIN_FILE = DATA_DIR / "owt_train.txt"

OUTPUT_DIR = (Path(__file__).parent / ".." / "result").resolve()
PROFILE_PATH = OUTPUT_DIR / "owt_tokenizer.prof"

VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>"]

def main():
    print(f"Training BPE on {TRAIN_FILE}")
    print(f"Vocab size: {VOCAB_SIZE}, special tokens: {SPECIAL_TOKENS}")

    vocab, merges = train_bpe(TRAIN_FILE, VOCAB_SIZE, SPECIAL_TOKENS)

    print(f"Trained: {len(vocab)} vocab entries, {len(merges)} merges")

    # Serialize vocab as JSON (token_id -> utf-8 decoded string wtih repr for non-utf8)
    vocab_json_path = OUTPUT_DIR / VOCAB_NAME
    vocab_serializable = {}
    for token_id, token_bytes in vocab.items():
        try:
            vocab_serializable[str(token_id)] = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            vocab_serializable[str(token_id)] = repr(token_bytes)
    with open(vocab_json_path, "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)
    print(f"Vocab saved to {vocab_json_path}")

    merges_json_path = OUTPUT_DIR / MERGES_NAME
    merges_serializable = []
    for a, b in merges:
        try:
            a_str = a.decode("utf-8")
        except UnicodeDecodeError:
            a_str = repr(a)
        try:
            b_str = b.decode("utf-8")
        except UnicodeDecodeError:
            b_str = repr(b)
        merges_serializable.append([a_str, b_str])
    with open(merges_json_path, "w", encoding="utf-8") as f:
        json.dump(merges_serializable, f, ensure_ascii=False, indent=2)
    print(f"Merges saved to {merges_json_path}")

    # Print a few sample merges for inspection
    print("\nFirst 20 merges:")
    for i, (a, b) in enumerate(merges[:20]):
        print(f"  {i+1:3d}: {a!r} + {b!r} -> {a+b!r}")

# sample n_samples documents from the training file and encode them with the trained tokenizer, printing some stats about the encoding
def encode_doc(
    doc_filepath: str | os.PathLike,
    vocab_filepath: str | os.PathLike,
    merges_filepath: str | os.PathLike,
    special_tokens: list[str] | None = None,
    n_samples: int = 10,
    max_docs: int = 500,
):
    import random

    tokenizer = BPETokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    delimiter = special_tokens[0] if special_tokens else "<|endoftext|>"

    # 流式读取文件，收集够 max_docs 个文档就停止，避免把整个大文件加载进内存
    documents: list[str] = []
    buf = ""
    CHUNK = 1 << 20  # 每次读取 1MB，适当调整以平衡内存使用和性能

    with open(doc_filepath, "r", encoding="utf-8") as f:
        while len(documents) < max_docs:
            raw = f.read(CHUNK)
            if not raw:
                break
            buf += raw
            parts = buf.split(delimiter)
            # 除最后一段外，其余都是完整文档
            for doc in parts[:-1]:
                doc = doc.strip()
                if doc:
                    documents.append(doc)
                    if len(documents) >= max_docs:
                        break
            # 最后一段可能不完整，留到下一轮继续拼接
            buf = parts[-1]

        # 处理文件末尾剩余内容
        if buf.strip() and len(documents) < max_docs:
            documents.append(buf.strip())

    if len(documents) < n_samples:
        print(f"Warning: only found {len(documents)} documents, sampling all of them")
        sampled = documents
    else:
        sampled = random.sample(documents, n_samples)

    print(f"Encoding {len(sampled)} sampled documents with the trained BPE tokenizer...")

    encoded_text: list[list[int]] = [tokenizer.encode(text) for text in sampled]

    total_bytes = sum(len(d.encode("utf-8")) for d in sampled)
    total_tokens = sum(len(e) for e in encoded_text)

    print(f"Sample encoded document (first 20 tokens): {encoded_text[0][:20]}")
    print(f"Sample encoded document length: {len(encoded_text[0])} tokens")
    print(f"Average tokens per document: {total_tokens / len(encoded_text):.2f}")
    print(f"Tokenizer compression ratio (bytes / tokens): {total_bytes / total_tokens:.2f} bytes/token")
    


def _encode_chunk_worker(args: tuple) -> int:
    """Worker: encode one file chunk and save token IDs to a temp .npy file.
    Returns the number of tokens written.
    """
    input_path, start, end, vocab_path, merges_path, special_tokens, tmp_path = args
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    tokens = tokenizer.encode(chunk)
    # NOTE: 这里将 tokens 写盘而不直接返回这个大数组，只是返回 tokens 的长度
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(tmp_path) 
    return len(tokens)


def encode_and_save_dataset(
    input_path: str | os.PathLike,
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    output_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
) -> None:
    """Encode an entire dataset file and serialize token IDs as a uint16 NumPy array.

    Uses multiprocessing to parallelize across CPU cores. Each worker encodes
    its chunk and writes to a temporary .npy file; the main process concatenates
    them and saves the final array, avoiding large inter-process data transfer.
    """
    n_proc = os.cpu_count() or 4
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, n_proc, split_token)

    tmp_dir = tempfile.mkdtemp(prefix="bpe_encode_")
    # NOTE: worker 把结果写到磁盘 .npy，只返回 token 数量这个 int。这是为了避免大数组通过 IPC 回传
    # multiprocessing.Pool 的返回值要 pickle 序列化再走管道，对于几 GB 的 token 数组开销巨大，写盘反而更快。
    chunk_args = [
        (
            input_path, start, end,
            vocab_path, merges_path, special_tokens,
            os.path.join(tmp_dir, f"chunk_{i}.bin"),
        )
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]

    print(f"Encoding {input_path.name if hasattr(input_path, 'name') else input_path} "
          f"with {len(chunk_args)} workers...")
    t0 = time.time()

    with Pool(len(chunk_args)) as pool:
        counts = pool.map(_encode_chunk_worker, chunk_args)
        pool.close()
        pool.join()

    total_tokens = sum(counts)
    print(f"  Encoding done in {time.time() - t0:.1f}s — {total_tokens:,} tokens")

    # Concatenate chunk arrays in order and save
    print("  Concatenating chunks...")
    with open(output_path, "wb") as out_f:
        for args in chunk_args:
            with open(args[6], "rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=16 * 1024 * 1024) # 16MB buffer 经验值
    
    # Clean up temp files
    for args in chunk_args:
        os.remove(args[6])
    os.rmdir(tmp_dir)

    size_gb = os.path.getsize(output_path) / 1e9
    print(f"  Saved {total_tokens:,} tokens ({size_gb:.2f} GB) → {output_path}")


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    #
    # main()
    #
    # profiler.disable()
    # profiler.dump_stats(PROFILE_PATH)
    #
    # # 直接打印 Top 30 耗时函数（按累计时间排序）
    # stream = io.StringIO()
    # stats = pstats.Stats(profiler, stream=stream)
    # stats.strip_dirs()
    # stats.sort_stats("cumulative")
    # stats.print_stats(30)
    # print(stream.getvalue())
    
    # print(f"\nprofile results saved to {PROFILE_PATH}")

    special_tokens = ["<|endoftext|>"]

    # TinyStories tokenizer paths
    tiny_vocab  = OUTPUT_DIR / "tinystories_bpe_vocab.json"
    tiny_merges = OUTPUT_DIR / "tinystories_bpe_merge.json"

    # OWT tokenizer paths
    owt_vocab  = OUTPUT_DIR / "owt_bpe_vocab.json"
    owt_merges = OUTPUT_DIR / "owt_bpe_merge.json"

    datasets = [
        # (input_text,                                     vocab,       merges,       output_npy)
        # (DATA_DIR / "TinyStoriesV2-GPT4-train.txt", tiny_vocab, tiny_merges, OUTPUT_DIR / "TinyStoriesV2-GPT4-train_tokens.bin"),
        # (DATA_DIR / "TinyStoriesV2-GPT4-valid.txt", tiny_vocab, tiny_merges, OUTPUT_DIR / "TinyStoriesV2-GPT4-valid_tokens.bin"),
        (DATA_DIR / "owt_train.txt",                owt_vocab,  owt_merges,  OUTPUT_DIR / "owt_train_tokens.bin"),
        (DATA_DIR / "owt_valid.txt",                owt_vocab,  owt_merges,  OUTPUT_DIR / "owt_valid_tokens.bin"),
    ]

    for input_path, vocab_path, merges_path, output_path in datasets:
        encode_and_save_dataset(input_path, vocab_path, merges_path, output_path, special_tokens)
    
     



