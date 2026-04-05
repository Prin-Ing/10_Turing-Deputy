# Turing Deputy

[한국어](./../README.md) | English | [日本語](./README.jp.md)

> A TypeScript project for implementing the core pieces of a Transformer-based language model from scratch.

## Overview

This repository is a learning-oriented project that builds language-model components directly in TypeScript without relying on external model APIs.

The current codebase includes:

- tensor-based numerical operations
- a byte-level BPE tokenizer
- token embedding and positional encoding
- multi-head self-attention
- a feed-forward network with GELU
- transformer blocks
- a top-level `TuringDeputy` model that produces logits
- draft training utilities such as a data loader and an AdamW-style optimizer

## Current Model Flow

```text
ids
  -> EmbeddingLayer
  -> TransformerBlock x N
  -> LayerNorm
  -> Linear (dModel -> vocabSize)
  -> logits
```

The current `forward()` implementation in `TuringDeputy` follows this flow directly.

## Project Structure

```text
Turing-Deputy/
├── model/
│   ├── attention.ts        # scaled dot-product attention, multi-head attention
│   ├── embedding.ts        # token embedding, positional encoding, embedding layer
│   ├── feedForward.ts      # 2-layer FFN + GELU
│   ├── transformerBlock.ts # attention + FFN block
│   └── turingDeputy.ts     # top-level language model
├── src/
│   ├── core/
│   │   └── tensor.ts       # tensor data structure and core ops
│   └── train/
│       ├── dataloader.ts   # next-token prediction batches
│       ├── optimizer.ts    # AdamW-style optimizer
│       ├── tokenizer.ts    # BPE tokenizer
│       └── train.ts        # training loop placeholder
├── data/
│   └── train.txt           # training text data
├── checkpoints/
│   └── index.ts            # checkpoint placeholder
├── package.json
└── README.md
```

## Implementation Status

- `src/core/tensor.ts` contains reshape, transpose, matmul, softmax, layernorm, `relu()`, `gelu()`, and `backprop()`.
- `model/embedding.ts` provides `EmbeddingLayer`, which combines token embeddings with sinusoidal positional encoding.
- `model/attention.ts` implements scaled dot-product attention and `MultiHeadAttention`.
- `model/feedForward.ts` implements a two-layer feed-forward network with GELU.
- `model/transformerBlock.ts` implements a pre-layernorm attention/FFN block.
- `model/turingDeputy.ts` connects embedding, transformer blocks, final layer normalization, and linear projection.
- `src/train/tokenizer.ts` contains a byte-level BPE tokenizer with save/load support.
- `src/train/dataloader.ts` prepares token batches for next-token prediction.
- `src/train/optimizer.ts` contains AdamW-style parameter update logic.

## Running the Project

There is not yet a fully wired training entry point or inference CLI in the repository.

Install dependencies:

```bash
npm install
```

At the moment, the repository is mainly organized around the model components and their documentation. The next practical step is to connect `train.ts` and checkpoint handling into a runnable training flow.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

## License

MIT
