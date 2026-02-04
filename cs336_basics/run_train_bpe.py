import argparse
import logging
import pickle

from cs336_basics import bpe

parser = argparse.ArgumentParser("run_train_bpe")
parser.add_argument("--input-path", required=True)
parser.add_argument("--vocab-size", type=int, required=True)
parser.add_argument("--special-tokens")
parser.add_argument(
    "--num-processes",
    type=int,
    default=1,
)
parser.add_argument("--output-path", required=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_train_bpe(input_path, vocab_size, special_tokens=None, num_processes=1):
    """Core BPE training function."""
    special_tokens = special_tokens or []
    vocab, merges = bpe.train_bpe(
        input_path,
        vocab_size,
        special_tokens,
        num_processes=num_processes,
    )
    return vocab, merges


def main():
    """CLI entry point."""
    args = parser.parse_args()
    logger.info(f"Running {parser.prog} with {args}")
    special_tokens = args.special_tokens.split(",") if args.special_tokens else []
    vocab, merges = run_train_bpe(
        args.input_path,
        args.vocab_size,
        special_tokens,
        num_processes=args.num_processes,
    )
    logger.info(
        f"Writing {len(vocab)} vocab and {len(merges)} merges to {args.output_path}"
    )
    with open(args.output_path, "wb") as f:
        pickle.dump(dict(args=args, vocab=vocab, merges=merges), f)
    logger.info("Done training.")


if __name__ == "__main__":
    main()
    