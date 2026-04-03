# 🧠 Turing Deputy

> Transformer 기반 LLM을 JavaScript로 밑바닥부터 구현하는 프로젝트.  
> ChatGPT, Claude와 같은 AI를 외부 API 없이, 순수 JS로 직접 만든다.

---

## 목표

ChatGPT, Claude와 같은 AI를 외부 API 없이, 순수 JavaScript로 직접 구현합니다.
Transformer 아키텍처의 모든 구성 요소를 처음부터 작성하며, CLI를 통해 대화할 수 있습니다.

---

## 아키텍처

```
입력 텍스트
    │
    ▼
[Tokenizer] — BPE 기반 토크나이저
    │
    ▼
[Embedding] — Token Embedding + Positional Encoding
    │
    ▼
[Transformer Block] × N
    ├── Multi-Head Self-Attention
    ├── Add & Layer Norm
    ├── Feed-Forward Network
    └── Add & Layer Norm
    │
    ▼
[Linear + Softmax] — 다음 토큰 예측
    │
    ▼
출력 텍스트
```

---

## 프로젝트 구조

```
Turing-Deputy/
├── src/
│   ├── core/
│   │   ├── tensor.js          # Tensor 자료구조 + 행렬 연산
│   │   └── autograd.js        # 자동 미분 (역전파)
│   │
│   ├── model/
│   │   ├── embedding.js       # Token + Positional Embedding
│   │   ├── attention.js       # Multi-Head Self-Attention
│   │   ├── ffn.js             # Feed-Forward Network
│   │   ├── transformer.js     # Transformer Block
│   │   └── gpt.js             # GPT 모델 (전체 조립)
│   │
│   ├── train/
│   │   ├── tokenizer.js       # BPE Tokenizer
│   │   ├── dataloader.js      # 배치 데이터 로더
│   │   └── optimizer.js       # AdamW Optimizer
│   │
│   └── cli/
│       └── index.js           # CLI 진입점 (대화 루프)
│
├── data/
│   └── train.txt              # 학습 데이터
│
├── checkpoints/               # 학습된 모델 가중치
├── package.json
└── README.md
```

---

## 개발 로드맵

### Phase 1 — 수학 기반 (`src/core/`)
- [ ] `tensor.js` — 다차원 배열, 행렬 곱, 브로드캐스팅
- [ ] `autograd.js` — 연산 그래프, 역전파 자동 미분

### Phase 2 — 모델 블록 (`src/model/`)
- [ ] `embedding.js` — Token Embedding, Sinusoidal Positional Encoding
- [ ] `attention.js` — Scaled Dot-Product Attention, Multi-Head Attention
- [ ] `ffn.js` — Position-wise Feed-Forward Network (ReLU/GELU)
- [ ] `transformer.js` — Transformer Block (Attention + FFN + LayerNorm)
- [ ] `gpt.js` — GPT 모델 전체 조립

### Phase 3 — 학습 파이프라인 (`src/train/`)
- [ ] `tokenizer.js` — BPE Tokenizer (학습 + 인코딩/디코딩)
- [ ] `dataloader.js` — 텍스트 → 배치 변환
- [ ] `optimizer.js` — AdamW (weight decay, gradient clipping)

### Phase 4 — 추론 & CLI (`src/cli/`)
- [ ] Greedy / Top-k / Top-p (nucleus) Sampling
- [ ] 대화 히스토리 관리
- [ ] CLI 인터페이스

---

## 설치 및 실행

```bash
# 설치
git clone https://github.com/yourname/Turing-Deputy
cd Turing-Deputy
npm install

# 학습
node src/train/run.js --data data/train.txt --epochs 10

# 대화
node src/cli/index.js --checkpoint checkpoints/model.json
```

---

## 기술 스택

| 항목 | 내용 |
|------|------|
| 언어 | JavaScript (Node.js) |
| 외부 라이브러리 | 없음 (순수 JS) |
| 인터페이스 | CLI |
| 모델 아키텍처 | GPT-style Decoder-only Transformer |
| Optimizer | AdamW |
| Tokenizer | BPE (Byte-Pair Encoding) |

---

## 참고 자료

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 원논문
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 논문
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy의 최소 GPT 구현

---

## 라이선스

MIT