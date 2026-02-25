"""
python cs336_basics/decode.py \
  --checkpoint path/to/checkpoint.pt \
  --tokenizer-path path/to/tokenizer.pkl \
  --prompt "Once upon a time" \
  --max-tokens 50 \
  --temperature 0.8 \
  --top-p 0.9
"""

import torch
import numpy as np
from typing import List, Optional
from cs336_basics.get_tokenizer import Tokenizer

@torch.no_grad()
def top_p_sampling(logits: torch.Tensor, p: float, temperature: float = 1.0) -> int:
    # logits: (vocab_size,)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = (cumulative_probs > p).float().argmax().item() + 1
    filtered_indices = sorted_indices[:cutoff]
    filtered_probs = sorted_probs[:cutoff]
    filtered_probs = filtered_probs / filtered_probs.sum()  # renormalize
    sampled_idx = torch.multinomial(filtered_probs, 1).item()
    return filtered_indices[sampled_idx].item()


def decode(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    end_token: Optional[int] = None,
    device: str = "cpu",
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = input_ids.copy()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_tensor)
            logits = out[0, -1, :]  # (vocab_size,)
            next_token = top_p_sampling(logits, p=top_p, temperature=temperature)
        generated.append(next_token)
        if end_token is not None and next_token == end_token:
            break
        input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
    return tokenizer.decode(generated)


def main():
    import argparse
    import pickle
    import importlib
    parser = argparse.ArgumentParser(description="Decode/generate from a trained model.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer pickle")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--end-token", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-module", default="cs336_basics.transformer_lm", help="Module containing model class")
    parser.add_argument("--model-class", default="TransformerLm", help="Model class name")
    args = parser.parse_args()

    # Load tokenizer
    with open(args.tokenizer_path, "rb") as f:
        tok_data = pickle.load(f)
    tokenizer = Tokenizer(tok_data["vocab"], tok_data["merges"])

    # Load model
    model_module = importlib.import_module(args.model_module)
    ModelClass = getattr(model_module, args.model_class)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    model = ModelClass(
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=4*config.d_model,
        theta=10000.0,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        device=args.device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    # Find end token if not provided
    end_token = args.end_token
    if end_token is None:
        for k, v in tokenizer.token_by_bytes.items():
            if k == b"<|endoftext|>":
                end_token = v
                break

    output = decode(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        end_token=end_token,
        device=args.device,
    )
    print(output)

if __name__ == "__main__":
    main()
