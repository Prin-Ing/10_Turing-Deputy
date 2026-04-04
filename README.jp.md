# 🧠 Turing Deputy

[한국어](./README.md) | [English](./README.en.md) | 日本語

> Transformer ベースの LLM を TypeScript でスクラッチ実装するプロジェクト

---

## 概要

外部 API に依存せず、ChatGPT や Claude のような AI を pure TypeScript で直接実装するプロジェクトです。  
Transformer アーキテクチャの主要コンポーネントを最初から作成し、最終的には Web ベースのチャット UI までつなぐことを目指します。

---

## アーキテクチャ

```text
入力テキスト
        │
        ▼
[Tokenizer] - BPE トークナイザー
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
[Linear + Softmax] - 次トークン予測
        │
        ▼
出力テキスト
```

---

## プロジェクト構成

```text
Turing-Deputy/
├── src/
│   ├── core/
│   │   ├── tensor.ts          # Tensor データ構造 + 行列演算
│   │   └── autograd.ts        # 自動微分
│   │
│   ├── train/
│   │   └── tokenizer.ts       # BPE トークナイザー
│   │
│   └── ...
│
├── model/
│   └── embedding.ts           # トークン埋め込み + 位置エンコーディング
│
├── data/
│   └── train.txt              # 学習データ
│
├── checkpoints/               # 保存されたモデル重み
├── package.json
└── README.md
```

---

## インストールと実行

```bash
git clone https://github.com/yourname/Turing-Deputy
cd Turing-Deputy
npm install

npx ts-node src/train/run.ts --data data/train.txt --epochs 10
```

サーバーの準備ができたら、`http://localhost:3000` で Web UI にアクセスできます。

---

## 技術スタック

- TypeScript (Node.js)
- 自前実装の Tensor / Tokenizer / Transformer コンポーネント
- Web UI
- Express + SSE
- GPT-style Decoder-only Transformer

---

## 参考資料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

---

## ライセンス

MIT
