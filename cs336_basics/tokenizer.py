import os
import regex as re
from typing import BinaryIO
from collections import defaultdict
from multiprocessing import Pool


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

    # Find chunk boundaries at special token boundaries
    with open(input_path, "rb") as f:
        chunks = find_chunk_boundaries(f, num_processes, split_special_token=b"<|endoftext|>")

    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(chunks[:-1], chunks[1:])
    ]

    word_freqs: dict[tuple, int] = {}
    
    with Pool(num_processes) as p:
        results = p.map(_pretokenize_chunk, chunk_args)
    
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
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_word_index: dict[tuple[bytes, bytes], set] = defaultdict(set)

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_word_index[pair].add(word)

    merges: list[tuple[bytes, bytes]] = []
    
    for _ in range(num_merges): 
        if not pair_counts:
            break
        
        # Pick highest-frequency pair; break ties lexicographically
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

        if pair_counts[best_pair] == 0:
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

            # increment the counts for all pairs in new word 
            for j in range(len(new_word_t) - 1):
                pair = (new_word_t[j], new_word_t[j + 1])
                pair_counts[pair] += freq
                pair_word_index[pair].add(new_word_t)

            # Update word_freqs 
            del word_freqs[word]
            word_freqs[new_word_t] = word_freqs.get(new_word_t, 0) + freq
            
        # Remove the merged pair (its count is now accounted for inside loop)
        pair_counts.pop(best_pair, None)
        pair_word_index.pop(best_pair, None)

    return vocab, merges
