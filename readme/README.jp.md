# Turing Deputy

[한국어](./../README.md) | [English](./README.en.md) | 日本語

> Transformer ベースの言語モデルの主要コンポーネントを TypeScript でスクラッチ実装するプロジェクトです。

## 概要

このリポジトリは、外部のモデル API に依存せず、言語モデルの基本構成要素を TypeScript で直接実装していく学習用プロジェクトです。

現在のコードベースには以下が含まれています。

- Tensor ベースの数値演算
- byte-level BPE トークナイザー
- token embedding と positional encoding
- multi-head self-attention
- GELU を使う feed-forward network
- transformer block
- logits を出力する上位モデル `TuringDeputy`
- DataLoader と AdamW スタイル optimizer の初期実装

## 現在のモデルフロー

```text
ids
  -> EmbeddingLayer
  -> TransformerBlock x N
  -> LayerNorm
  -> Linear (dModel -> vocabSize)
  -> logits
```

現在の `TuringDeputy` の `forward()` はこの流れに沿って実装されています。

## プロジェクト構成

```text
Turing-Deputy/
├── model/
│   ├── attention.ts        # scaled dot-product attention, multi-head attention
│   ├── embedding.ts        # token embedding, positional encoding, embedding layer
│   ├── feedForward.ts      # 2-layer FFN + GELU
│   ├── transformerBlock.ts # attention + FFN block
│   └── turingDeputy.ts     # 上位 language model
├── src/
│   ├── core/
│   │   └── tensor.ts       # tensor データ構造と基本演算
│   └── train/
│       ├── dataloader.ts   # next-token prediction 用バッチ生成
│       ├── optimizer.ts    # AdamW スタイル optimizer
│       ├── tokenizer.ts    # BPE tokenizer
│       └── train.ts        # 学習ループは今後実装予定
├── data/
│   └── train.txt           # 学習用テキストデータ
├── checkpoints/
│   └── index.ts            # checkpoint 処理は今後実装予定
├── package.json
└── README.md
```

## 実装状況

- `src/core/tensor.ts` には reshape, transpose, matmul, softmax, layernorm, `relu()`, `gelu()`, `backprop()` が実装されています。
- `model/embedding.ts` は token embedding と sinusoidal positional encoding をまとめた `EmbeddingLayer` を提供します。
- `model/attention.ts` は scaled dot-product attention と `MultiHeadAttention` を実装しています。
- `model/feedForward.ts` は GELU を使う 2 層 feed-forward network を実装しています。
- `model/transformerBlock.ts` は pre-layernorm 形式の attention/FFN block を実装しています。
- `model/turingDeputy.ts` は embedding, transformer blocks, final layer normalization, linear projection を接続しています。
- `src/train/tokenizer.ts` は byte-level BPE tokenizer と save/load ロジックを含みます。
- `src/train/dataloader.ts` は next-token prediction 用のトークンバッチを生成します。
- `src/train/optimizer.ts` は AdamW スタイルのパラメータ更新ロジックを持ちます。

## 実行

現時点では、完全に接続された学習エントリーポイントや推論 CLI はまだありません。

依存関係のインストール:

```bash
npm install
```

現在は主にモデル構成要素とその文書化を進めている段階です。次の実作業は `train.ts` と checkpoint 処理を実行可能な学習フローとしてつなぐことです。

## 参考資料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

## ライセンス

MIT
