# Assignment 3: Transformer is All You Need

Implementation of a Tiny Transformer language model from scratch in PyTorch, trained on the Tiny Shakespeare corpus for next-token prediction.

## Setup
```bash
pip install torch tokenizers
```

## Usage
Run all cells top to bottom in `AML_A3.ipynb`. Dataset is downloaded automatically — no manual setup needed.

## Model Architecture
- 2 Transformer blocks, d_model=128, 4 heads, ffn_dim=256
- Sinusoidal positional encoding
- Causal multi-head self-attention (from scratch)
- RMSNorm + residual connections
- Weight-tied output projection
- 327,552 parameters

## Experiments
| Experiment | Configs | Best PPL |
|---|---|---|
| Learning rate | 1e-4, 3e-4, 1e-3 | 53.93 (lr=1e-4) |
| Context length | 32, 64, 128 | 53.47 (ctx=32) |
| Model size | 1L-64d, 2L-128d, 3L-128d | 54.33 (2L-128d) |
| PE Ablation | With vs. Without PE | 54.33 → 80.45 (+48.1%) |

## Results
- **Final Validation PPL: 54.33** (main model: 2L-128d, ctx=64, lr=3e-4)
- PPL = exp(validation cross-entropy loss)
- Random baseline PPL = 500 (uniform over vocabulary)

## Key Findings
- Positional encoding is critical — removing it degrades PPL by 48.1%
- Attention is the runtime bottleneck: 75.9% of forward pass, O(T²) in sequence length
- Backward pass is 2.5× slower than forward (4.62ms vs 1.86ms)
- Model size sweet spot at 2 layers — 3 layers overfits after epoch 3
