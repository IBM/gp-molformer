```sh
>>> python scripts/conditional_generation.py --help
usage: conditional_generation.py [-h] [--model MODEL] [--batch_size BATCH_SIZE] [--seed SEED] prompt

positional arguments:
  prompt                Partial SMILES to be completed

options:
  -h, --help            show this help message and exit
  --model MODEL         Path to pretrained model
  --batch_size BATCH_SIZE
                        Number of sequences to return
  --seed SEED           Random seed
```

```sh
>>> python scripts/unconditional_generation.py --help
usage: unconditional_generation.py [-h] [--model MODEL] [--deterministic] [--temperature TEMPERATURE] [--batch_size BATCH_SIZE] [--num-batches NUM_BATCHES] [--seed SEED] filename

positional arguments:
  filename              Save file

options:
  -h, --help            show this help message and exit
  --model MODEL         Path to pretrained model
  --deterministic       Constant random features. Still uses multinomial (random) sampling
  --temperature TEMPERATURE
                        Softmax temperature
  --batch_size BATCH_SIZE
  --num-batches NUM_BATCHES
  --seed SEED           Random seed
```

```sh
>>> python scripts/pairtune_training.py --help
usage: pairtune_training.py [-h] [--model MODEL] [-k K] [--batch_size BATCH_SIZE] [--lr LR] [--num_epochs NUM_EPOCHS] [--eval_epochs EVAL_EPOCHS] [--lamb] [--seed SEED] {qed,drd2,logp06}

positional arguments:
  {qed,drd2,logp06}     Pair-tuning task (see data/pairtune/*)

options:
  -h, --help            show this help message and exit
  --model MODEL         Path to pretrained model
  -k K                  Number of generations per validation molecule
  --batch_size BATCH_SIZE
                        Total batch size (divided amongst all GPUs)
  --lr LR               learning rate
  --num_epochs NUM_EPOCHS
                        Number of training epochs
  --eval_epochs EVAL_EPOCHS
                        Number of epochs between evals
  --lamb                use lamb optimizer or not
  --seed SEED           Random seed
```
