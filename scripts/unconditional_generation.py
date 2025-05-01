from argparse import ArgumentParser

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import trange

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("filename", type=str, help="Save file")
    parser.add_argument("--model", type=str, default="ibm-research/GP-MoLFormer-Uniq", help="Path to pretrained model")
    parser.add_argument("--deterministic", action="store_true", help="Constant random features. Still uses multinomial (random) sampling")
    parser.add_argument("--temperature", type=float, default=1, help="Softmax temperature")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--num_batches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, deterministic_eval=args.deterministic, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    gen = [
        model.generate(
            do_sample=True,
            temperature=args.temperature,
            top_k=None,
            max_length=model.config.max_position_embeddings,
            num_return_sequences=args.batch_size
        ).cpu() for _ in trange(args.num_batches)
    ]
    smi = []
    for batch in gen:
        smi.extend(tokenizer.batch_decode(batch, skip_special_tokens=True))
    pd.Series(smi).to_csv(args.filename, header=False, index=False)
