
# from __future__ import annotations
import ast
import json
import os
import regex as re
from typing import BinaryIO
from collections import defaultdict
from multiprocessing import Pool
from typing import Iterable, Self
import heapq
from functools import total_ordering


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
 
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    
    chunk_boundaries: list[int] = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)  
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))
                          

def _pretokenize_chunk(args: tuple) -> dict[tuple, int]:
    """Pre-tokenize one chunk of the corpus, returning word-frequency counts.
    Returns:
        dict[tuple, int]: 
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes) 
    """
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
    delimiter = "|".join(re.escape(tok) for tok in special_tokens) 
    splited_chunk = re.split(delimiter, chunk)
    
    word_freqs = defaultdict(int)
    pat = re.compile(PAT)
    
    for part in splited_chunk: 
        for match in pat.finditer(part):
            word_bytes = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            word_freqs[word_bytes] += 1
        
    return word_freqs
    
@total_ordering
class _RevPair:
    """Wrapper that reverses bytes-pair comparision, so min-heap picks 
    the lexicographically largest pair on ties
    (matching the semantics of max(pair_counts, key=lambda p: (pair_counts[p], p))).
    """
    #NOTE: 默认情况下，Python 每个实例都会带 __dict__ 来存储属性，但它很占内存（每个实例都是一个 hash table）
    # __slots__ 告诉 Python 只有这一个属性
    __slots__ = ("pair",) 
    
    def __init__(self, pair) -> None:
        self.pair = pair

    def __eq__(self, other: object) -> bool:
        #NOTE: 容忍度不同，a == b 绝不应该抛异常
        return isinstance(other, _RevPair) and self.pair == other.pair

    def __lt__(self, other: Self) -> bool:
        return self.pair > other.pair
    

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int, 
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Returns:
        vocab: mapping from token id -> bytes
        merges: list of (bytes, bytes) merge pairs in creation order
    """
    num_processes = os.cpu_count() or 4
    print(f"Training BPE with vocab size {vocab_size} using {num_processes} processes...")

    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    # Find chunk boundaries at special token boundaries
    with open(input_path, "rb") as f:
        chunks = find_chunk_boundaries(f, num_processes, split_special_token=split_token)

    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(chunks[:-1], chunks[1:])
    ]

    word_freqs: dict[tuple, int] = {}
    
    # 修复：map 完成后主动 close + join，with 退出时 terminate() 变成 no-op
    with Pool(num_processes) as p:
        results = p.map(_pretokenize_chunk, chunk_args)
        p.close()   # 通知 worker 不再接收新任务
        p.join()    # 等待 worker 进程和所有守护线程正常退出   
        
    for chunk_result in results:
        for word, freq in chunk_result.items():
            word_freqs[word] = word_freqs.get(word, 0) + freq
    
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    for tokens in special_tokens:
        vocab[next_id] = tokens.encode("utf-8")
        next_id += 1

    num_merges = vocab_size - len(vocab) 
    if num_merges <= 0:
        return vocab, []
    
    # Build initial pair counts and reverse index (pair -> set of words)
    # pretokenize 后的 word_freqs 是 dict[tuple[bytes], int]，其中 tuple[bytes] 是一个 word 的 byte-level token 序列
    # pair_counts 统计每个 byte pair 在整个语料库中出现的频率（加权了 word_freqs）
    # pair_word_index 反向索引每个 byte pair 出现在哪些 word 中（以便后续 merge 后更新）
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_word_index: dict[tuple[bytes, bytes], set] = defaultdict(set)

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_word_index[pair].add(word)

    merges: list[tuple[bytes, bytes]] = []

    # best_pair 可以通过 max(pair_counts, key=...) 来获取，但每次 merge 后需要更新，所以改用 heap 来高效获取频率最高的 pair
    #NOTE: max(Iterable, key=fn) traverse Iterable, compair fn(elem) and return the original elem.
    # python 对于元组首先比较第一个元素，相同比较第二个(lexicographically largest)
    # best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

    # 使用 heap 来高效获取频率最高的 pair，注意 heapq 是 min-heap，所以存储负数频率
    #NOTE: 用 _RevPair 来包装 pair，使得在频率相同的情况下，lexicographically larger 的 pair 会被优先弹出
    heap = [(-count, _RevPair(pair)) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    
    for _ in range(num_merges):  
        if not pair_counts:
            break        

        while heap:
            neg_count, rev_key = heap[0]
            if pair_counts.get(rev_key.pair, 0) == -neg_count:
                break
            heapq.heappop(heap)

        if not heap:
            break

        neg_count, rev_key = heapq.heappop(heap)
        best_pair = rev_key.pair

        if -neg_count == 0:
            break

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1

        # Apply the merge to every word that contains best_pair
        for word in list(pair_word_index[best_pair]):
            freq = word_freqs.get(word, 0)
            if freq == 0:
                continue
            
            # Build merged words
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1 
                    and word[i] == best_pair[0] 
                    and word[i + 1] == best_pair[1]
                ):
                    new_word.append(new_token)
                    i += 2 
                else:
                    new_word.append(word[i])
                    i += 1
                    
            new_word_t = tuple(new_word)

            if new_word_t == word:
                continue
        
            # decrement the counts for all pairs in old word 
            for j in range(len(word) - 1):
                pair = (word[j], word[j + 1])
                pair_counts[pair] -= freq
                pair_word_index[pair].discard(word)
                # push 更新后的 count
                if pair_counts[pair] > 0:
                    heapq.heappush(heap, (-pair_counts[pair], _RevPair(pair)))

            # increment the counts for all pairs in new word 
            for j in range(len(new_word_t) - 1):
                pair = (new_word_t[j], new_word_t[j + 1])
                pair_counts[pair] += freq
                pair_word_index[pair].add(new_word_t)
                # push 更新后的 count
                heapq.heappush(heap, (-pair_counts[pair], _RevPair(pair)))

            # Update word_freqs 
            del word_freqs[word]
            word_freqs[new_word_t] = word_freqs.get(new_word_t, 0) + freq
            
        # Remove the merged pair (its count is now accounted for inside loop)
        pair_counts.pop(best_pair, None)
        pair_word_index.pop(best_pair, None)

    return vocab, merges


class BPETokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        '''Construct a tokenizer from a given vocabulary, list of merges, 
        and (optionally) a list of special tokens.
        ''' 
        # 防御性拷贝，每个实例拥有自己独立的数据
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = list(special_tokens) if special_tokens else []
        
        existing_bytes = set(self.vocab.values())
        for tokens in self.special_tokens:
            token_bytes = tokens.encode("utf-8")
            if token_bytes not in existing_bytes:
                self.vocab[len(self.vocab)] = token_bytes
                existing_bytes.add(token_bytes)
        # Inverse vocab: encode 操作需要用到         
        self.reversed_vocab = {v: k for k, v in vocab.items()} # dict[bytes, int]

        # Merge rank for O(1) lookup: merge pair -> rank (index in merge list)
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # Set of special tokens for O(1) membership test
        # membership test: 判断一个元素是否在一个集合中
        self.special_tokens_set: set[str] = set(self.special_tokens)

        # pre-compiled regex for splitting on special tokens (longest first)
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            # re.split 的构造方式会导致 special_tokens 自身被丢弃，要用捕获组
            pattern = "(" + "|".join(re.escape(tok) for tok in sorted_special) + ")"
            self._special_pattern = re.compile(pattern)
        else:
            self._special_pattern = None 
        
        # Pre-compiled GPT-2 pre-tokenization regex
        self._pat = re.compile(PAT)

     
    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike, 
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        '''Class method that constructs and return a Tokenizer from a serialized vocabulary 
        and list of merges (in the same format that your BPE training code output) and 
        (optionally) a list of special tokens. 
        '''
        def restore(s: str) -> bytes:
            if s.startswith("b'") or s.startswith('b"'):
                return ast.literal_eval(s)
            else:
                return s.encode("utf-8")
        
        vocab = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            # dict[str, str] -> dict[int, bytes]
            vocab_loaded = json.load(f)
        for id, token_str in vocab_loaded.items():
            if token_str.startswith("b'") or token_str.startswith('b"'):
                vocab[int(id)] = ast.literal_eval(token_str)
            else:
                vocab[int(id)] = token_str.encode("utf-8") 
            
        with open(merges_filepath, "r", encoding="utf-8") as f:
            # list[tuple[str, str]] -> list[tuple[bytes, bytes]]
            merges_loaded = json.load(f) 
        merges = [(restore(a), restore(b)) for a, b in merges_loaded]

        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        """Apply BPE merges greedily (lowest-rank merge first) to a token list.""" 
        while len(tokens) > 1:
            best_rank: int | None = None
            best_idx: int | None = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i

            if best_idx is None:
                break

            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]

        return tokens

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode a plain text chunk (no special tokens) into token IDs."""
        ids = []
        for match in self._pat.finditer(text):
            pre_token_bytes = match.group().encode("utf-8")
            tokens = [bytes([b]) for b in pre_token_bytes]
            tokens = self._apply_merges(tokens)
            # 生成器表达式（懒加载，不生成中间列表）
            ids.extend(self.reversed_vocab[t] for t in tokens)
        return ids
   
    def encode(self, text: str) -> list[int]:
        '''Encode an input text into a sequence of token IDs.'''
        if not text: 
            return []

        if self._special_pattern is None: 
            return self._encode_chunk(text)

        ids = []
        parts = self._special_pattern.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens_set:
                ids.append(self.reversed_vocab[part.encode("utf-8")])
            else:
                ids.extend(self._encode_chunk(part))
        return ids
                
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        '''Given an iterable of strings (e.g., a Python file handle), return a 
        generator that lazily yields token IDs. This is required for memory-efficient 
        tokenization of large files that we cannot directly load into memory.
        ''' 
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            # Find the last non-whitespace character so we don't split inside a whitespace pre-token 
            # that might continue in the next chunk. 
            last_nonws = len(buffer) - 1
            while last_nonws >= 0 and buffer[last_nonws].isspace():
                last_nonws -= 1

            if last_nonws >= 0:
                to_encode = buffer[: last_nonws + 1]
                buffer = buffer[last_nonws + 1 :]
                yield from self.encode(to_encode)
            
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        '''Decode a sequence of token IDs into text.'''
        byte_data = b"".join(self.vocab[id] for id in ids)
        return byte_data.decode("utf-8", errors="replace")
                
        
