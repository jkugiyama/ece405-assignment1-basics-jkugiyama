"""
Flexible tokenization script for text data.
Supports streaming processing for large files and custom tokenizer/vocab handling.

Usage:
    python tokenize_flexible.py --input-path data/owt_train.txt --output-path data/tokens.npy
    python tokenize_flexible.py --input-path data/owt_train.txt --output-path data/tokens.npy --max-chars 1000000
    python tokenize_flexible.py --input-path data/owt_train.txt --output-path data/tokens.npy --lines 100
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from cs336_basics.get_tokenizer import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_gpt2_tokenizer() -> Tokenizer:
    """Load the GPT2 reference tokenizer from fixtures."""
    fixtures_dir = Path("tests/fixtures")
    
    # Load vocab from gpt2_vocab.json (maps string -> token_id)
    with open(fixtures_dir / "gpt2_vocab.json", "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    
    # Invert to get token_id -> bytes
    vocab = {token_id: token_str.encode("utf-8") 
             for token_str, token_id in vocab_dict.items()}
    
    # Load merges from gpt2_merges.txt
    merges = []
    with open(fixtures_dir / "gpt2_merges.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) == 2:
                    merges.append((
                        parts[0].encode("utf-8"),
                        parts[1].encode("utf-8")
                    ))
    
    logger.info(f"Loaded GPT2 tokenizer: {len(vocab)} vocab items, {len(merges)} merges")
    return Tokenizer(vocab, merges, special_tokens=None)


def read_lines(file_path: str, num_lines: Optional[int] = None):
    """Generator to read lines from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if num_lines and i >= num_lines:
                break
            yield line


def read_chunks(file_path: str, chunk_size: int = 1000000):
    """Generator to read file in chunks of characters."""
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def tokenize_file_streaming(
    input_path: str,
    output_path: str,
    tokenizer: Optional[Tokenizer] = None,
    max_chars: Optional[int] = None,
    num_lines: Optional[int] = None,
    chunk_size: int = 1000000,
) -> int:
    """
    Tokenize a file with streaming/chunked processing.
    
    Args:
        input_path: Path to input text file
        output_path: Path to output .npy file
        tokenizer: Tokenizer instance. If None, loads GPT2 reference.
        max_chars: Maximum character count to process. If None, process entire file.
        num_lines: Maximum number of lines to process. If None, process entire file.
        chunk_size: Size of chunks to process at a time (in characters)
    
    Returns:
        Total number of tokens encoded
    """
    
    if tokenizer is None:
        tokenizer = load_gpt2_tokenizer()
    
    logger.info(f"Reading from {input_path}...")
    
    # Determine what data to read
    if num_lines is not None:
        logger.info(f"Processing first {num_lines} lines...")
        text_data = []
        for line in read_lines(input_path, num_lines):
            text_data.append(line)
        text = "".join(text_data)
    elif max_chars is not None:
        logger.info(f"Processing first {max_chars:,} characters...")
        text_data = []
        chars_read = 0
        for chunk in read_chunks(input_path, chunk_size):
            remaining = max_chars - chars_read
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                text_data.append(chunk[:remaining])
                chars_read += remaining
            else:
                text_data.append(chunk)
                chars_read += len(chunk)
        text = "".join(text_data)
    else:
        logger.info("Processing entire file...")
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    logger.info(f"Read {len(text):,} characters")
    logger.info(f"Tokenizing...")
    
    tokens = tokenizer.encode(text)
    
    logger.info(f"Encoded {len(tokens):,} tokens")
    logger.info(f"Saving to {output_path}...")
    
    tokens_array = np.array(tokens, dtype=np.int32)
    np.save(output_path, tokens_array)
    
    logger.info(f"✓ Done! Saved {len(tokens):,} tokens ({len(tokens) * 4 / 1024 / 1024:.1f} MB)")
    
    return len(tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Flexible tokenization script for text data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tokenize entire file
  python tokenize_flexible.py --input-path data/owt_train.txt --output-path data/tokens.npy
  
  # Tokenize first 1M characters (fast test)
  python tokenize_flexible.py --input-path data/owt_train.txt --output-path data/tokens.npy --max-chars 1000000
  
  # Tokenize first 100 lines
  python tokenize_flexible.py --input-path data/owt_train.txt --output-path data/tokens.npy --lines 100
  
  # Use custom tokenizer
  python tokenize_flexible.py --input-path data/owt_train.txt --output-path data/tokens.npy --tokenizer-path custom.pkl
        """
    )
    
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to input text file"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output .npy file"
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Path to pickled tokenizer (vocab+merges). If not provided, uses GPT2 reference."
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Maximum number of characters to process (useful for testing)"
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=None,
        help="Maximum number of lines to process"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000000,
        help="Size of chunks to process at a time (default: 1M characters)"
    )
    
    args = parser.parse_args()
    
    # Load custom tokenizer if provided
    tokenizer = None
    if args.tokenizer_path:
        logger.info(f"Loading tokenizer from {args.tokenizer_path}...")
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
    
    # Validate arguments
    if args.max_chars and args.lines:
        logger.warning("Both --max-chars and --lines specified. Using --max-chars.")
        args.lines = None
    
    # Run tokenization
    tokenize_file_streaming(
        input_path=args.input_path,
        output_path=args.output_path,
        tokenizer=tokenizer,
        max_chars=args.max_chars,
        num_lines=args.lines,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
