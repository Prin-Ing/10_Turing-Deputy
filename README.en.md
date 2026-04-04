# 🧠 Turing Deputy

[한국어](./README.md) | English | [日本語](./README.jp.md)

> A project to build a Transformer-based LLM from scratch in TypeScript

---

## Overview

This project aims to implement an AI system like ChatGPT or Claude in pure TypeScript without relying on external APIs.  
It covers the core Transformer components from scratch and eventually connects them to a web-based chat UI.

---

## Architecture

```text
Input Text
        │
        ▼
[Tokenizer] - BPE Tokenizer
        │
        ▼
[Embedding] - Token Embedding + Positional Encoding
        │
        ▼
[Transformer Block] x N
        ├── Multi-Head Self-Attention
        ├── Add & Layer Norm
        ├── Feed-Forward Network
        └── Add & Layer Norm
        │
        ▼
[Linear + Softmax] - Next Token Prediction
        │
        ▼
Output Text
```

---

## Project Structure

```text
Turing-Deputy/
├── src/
│   ├── core/
│   │   ├── tensor.ts          # Tensor data structure + matrix ops
│   │   └── autograd.ts        # Automatic differentiation
│   │
│   ├── train/
│   │   └── tokenizer.ts       # BPE tokenizer
│   │
│   └── ...
│
├── model/
│   └── embedding.ts           # Token embedding + positional encoding
│
├── data/
│   └── train.txt              # Training corpus
│
├── checkpoints/               # Saved model weights
├── package.json
└── README.md
```

---

## Setup and Run

```bash
git clone https://github.com/yourname/Turing-Deputy
cd Turing-Deputy
npm install

npx ts-node src/train/run.ts --data data/train.txt --epochs 10
```

Once the server is ready, the web UI should be available at `http://localhost:3000`.

---

## Tech Stack

- TypeScript (Node.js)
- Custom Tensor / Tokenizer / Transformer components
- Web UI
- Express + SSE
- GPT-style Decoder-only Transformer

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

---

## License

MIT
