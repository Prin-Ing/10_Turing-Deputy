# 🧠 Turing Deputy

> Transformer 기반 LLM을 TypeScript로 밑바닥부터 구현하는 프로젝트.  
> ChatGPT, Claude와 같은 AI를 외부 API 없이, 순수 TS로 직접 만든다.

---

## 목표

ChatGPT, Claude와 같은 AI를 외부 API 없이, 순수 TypeScript로 직접 구현합니다.
Transformer 아키텍처의 모든 구성 요소를 처음부터 작성하며, 웹 기반 채팅 UI를 통해 대화할 수 있습니다.


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
│   │   ├── tensor.ts          # Tensor 자료구조 + 행렬 연산
│   │   └── autograd.ts        # 자동 미분 (역전파)
│   │
│   ├── model/
│   │   ├── embedding.ts       # Token + Positional Embedding
│   │   ├── attention.ts       # Multi-Head Self-Attention
│   │   ├── ffn.ts             # Feed-Forward Network
│   │   ├── transformer.ts     # Transformer Block
│   │   └── gpt.ts             # GPT 모델 (전체 조립)
│   │
│   ├── train/
│   │   ├── tokenizer.ts       # BPE Tokenizer
│   │   ├── dataloader.ts      # 배치 데이터 로더
│   │   └── optimizer.ts       # AdamW Optimizer
│   │
│   └── server/
│       ├── index.ts           # Express 서버 진입점
│       └── api.ts             # /api/chat, /api/generate 엔드포인트
│
├── web/
│   ├── index.html             # 채팅 UI (직접 디자인)
│   ├── style.css              # 스타일
│   └── app.ts                 # 프론트엔드 로직 (fetch, 스트리밍)
│
├── data/
│   └── train.txt              # 학습 데이터
│
├── checkpoints/               # 학습된 모델 가중치
├── tsconfig.json
├── package.json
└── README.md
```

---

## 개발 로드맵

### Phase 1 — 수학 기반 (`src/core/`)
- [ ] `tensor.ts` — 다차원 배열, 행렬 곱, 브로드캐스팅
- [ ] `autograd.ts` — 연산 그래프, 역전파 자동 미분

### Phase 2 — 모델 블록 (`src/model/`)
- [ ] `embedding.ts` — Token Embedding, Sinusoidal Positional Encoding
- [ ] `attention.ts` — Scaled Dot-Product Attention, Multi-Head Attention
- [ ] `ffn.ts` — Position-wise Feed-Forward Network (GELU)
- [ ] `transformer.ts` — Transformer Block (Attention + FFN + LayerNorm)
- [ ] `gpt.ts` — GPT 모델 전체 조립

### Phase 3 — 학습 파이프라인 (`src/train/`)
- [ ] `tokenizer.ts` — BPE Tokenizer (학습 + 인코딩/디코딩)
- [ ] `dataloader.ts` — 텍스트 → 배치 변환
- [ ] `optimizer.ts` — AdamW (weight decay, gradient clipping)

### Phase 4 — 추론 & 웹 서버 (`src/server/`)
- [ ] Greedy / Top-k / Top-p (nucleus) Sampling
- [ ] Express 서버 + REST API (`/api/chat`, `/api/generate`)
- [ ] SSE(Server-Sent Events) 스트리밍 응답
- [ ] 대화 히스토리 관리

### Phase 5 — 웹 UI (`web/`)
- [ ] 채팅 인터페이스 (직접 디자인)
- [ ] 스트리밍 응답 렌더링
- [ ] 대화 세션 관리

---

## 설치 및 실행

```bash
# 설치
git clone https://github.com/yourname/Turing-Deputy
cd Turing-Deputy
npm install

# 빌드
npm run build

# 학습
npx ts-node src/train/run.ts --data data/train.txt --epochs 10

# 서버 실행
npx ts-node src/server/index.ts --checkpoint checkpoints/model.json
# → http://localhost:3000 에서 웹 UI 접속
```

---

## 기술 스택

| 항목 | 내용 |
|------|------|
| 언어 | TypeScript (Node.js) |
| 외부 라이브러리 | 없음 (순수 TS,JS) |
| 인터페이스 | Web (직접 디자인) |
| 백엔드 | Express + SSE 스트리밍 |
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