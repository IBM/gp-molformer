from argparse import ArgumentParser

from rdkit import Chem, RDLogger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# suppress warnings
RDLogger.DisableLog('rdApp.*')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("prompt", type=str, help="Partial SMILES to be completed")
    parser.add_argument("--model", type=str, default="ibm-research/GP-MoLFormer-Uniq", help="Path to pretrained model")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of sequences to return")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"][:, :-1].to(device)

    gen = model.generate(
        input_ids,
        do_sample=True,
        top_k=None,
        max_length=model.config.max_position_embeddings,
        num_return_sequences=args.batch_size
    )
    smi = tokenizer.batch_decode(gen, skip_special_tokens=True)
    mols = [Chem.CanonSmiles(s, useChiral=0) for s in smi if Chem.MolFromSmiles(s) is not None]

    print(f"Total number of successful molecules generated is {len(mols)}")
    print(f"Total number of unique molecules generated is {len(set(mols))}")
    print(smi)
