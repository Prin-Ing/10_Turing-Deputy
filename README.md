# Turing Deputy

한국어 | [English](./readme/README.en.md) | [日本語](./readme/README.jp.md)

> TypeScript로 Transformer 기반 언어 모델의 핵심 구성 요소를 직접 구현하는 프로젝트입니다.

## 소개

이 저장소는 외부 모델 API에 의존하지 않고, 언어 모델의 기본 구성 요소를 TypeScript로 직접 작성해보는 학습용 프로젝트입니다.

현재 코드베이스에는 다음 요소가 포함되어 있습니다.

- `Tensor` 기반 수치 연산
- BPE 토크나이저
- 토큰 임베딩과 positional encoding
- multi-head self-attention
- feed-forward network와 GELU
- transformer block
- 최종 logits를 생성하는 `TuringDeputy` 모델
- 데이터 로더와 AdamW 스타일 옵티마이저 초안

## 현재 모델 흐름

```text
ids
  -> EmbeddingLayer
  -> TransformerBlock x N
  -> LayerNorm
  -> Linear (dModel -> vocabSize)
  -> logits
```

현재 `TuringDeputy`의 `forward()`는 위 흐름을 그대로 따릅니다.

## 프로젝트 구조

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
│   │   └── tensor.ts       # tensor 자료구조와 기본 연산
│   └── train/
│       ├── dataloader.ts   # next-token prediction 배치 생성
│       ├── optimizer.ts    # AdamW 스타일 optimizer
│       ├── tokenizer.ts    # BPE tokenizer
│       └── train.ts        # 학습 루프 작업 예정
├── data/
│   └── train.txt           # 학습용 텍스트 데이터
├── checkpoints/
│   └── index.ts            # 체크포인트 작업 예정
├── package.json
└── README.md
```

## 구현 상태

- `src/core/tensor.ts`에는 reshape, transpose, matmul, softmax, layernorm, `relu()`, `gelu()`, `backprop()` 등이 구현되어 있습니다.
- `model/embedding.ts`는 token embedding과 sinusoidal positional encoding을 합친 `EmbeddingLayer`를 제공합니다.
- `model/attention.ts`는 scaled dot-product attention과 `MultiHeadAttention`을 구현합니다.
- `model/feedForward.ts`는 GELU를 사용하는 2-layer feed-forward network를 구현합니다.
- `model/transformerBlock.ts`는 pre-layernorm 방식의 attention/FFN block을 구현합니다.
- `model/turingDeputy.ts`는 embedding, transformer blocks, final layernorm, linear projection을 연결합니다.
- `src/train/tokenizer.ts`는 byte-level BPE tokenizer와 저장/로드 로직을 포함합니다.
- `src/train/dataloader.ts`는 텍스트를 토큰 id로 읽고 학습 배치를 생성합니다.
- `src/train/optimizer.ts`는 AdamW 스타일 파라미터 업데이트 로직을 담고 있습니다.

## 실행

현재 저장소에는 완성된 학습 진입점이나 추론 CLI는 아직 연결되어 있지 않습니다.

기본 의존성 설치:

```bash
npm install
```

현재는 각 모듈 구현과 문서화를 중심으로 정리된 상태이며, 다음 단계는 `train.ts`와 체크포인트 로직을 실제 실행 가능하게 연결하는 것입니다.

## 참고 자료

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

## 라이선스

MIT
