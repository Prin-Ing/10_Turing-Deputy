# 🧠 Turing Deputy

한국어 | [English](./README.en.md) | [日本語](./README.jp.md)

> Transformer 기반 LLM을 TypeScript로 밑바닥부터 구현하는 프로젝트

---

## 소개

외부 API 없이 ChatGPT, Claude와 같은 AI를 순수 TypeScript로 직접 구현하는 프로젝트입니다.  
Transformer 아키텍처의 핵심 구성 요소를 처음부터 작성하고, 최종적으로는 웹 기반 채팅 UI까지 연결하는 것을 목표로 합니다.

---

## 아키텍처

```text
입력 텍스트
        │
        ▼
[Tokenizer] - BPE 토크나이저
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
[Linear + Softmax] - 다음 토큰 예측
        │
        ▼
출력 텍스트
```

---

## 프로젝트 구조

```text
Turing-Deputy/
├── src/
│   ├── core/
│   │   ├── tensor.ts          # Tensor 자료구조 + 행렬 연산
│   │   └── autograd.ts        # 자동 미분
│   │
│   ├── train/
│   │   └── tokenizer.ts       # BPE 토크나이저
│   │
│   └── ...
│
├── model/
│   └── embedding.ts           # 토큰 임베딩 + 위치 인코딩
│
├── data/
│   └── train.txt              # 학습 데이터
│
├── checkpoints/               # 저장된 모델 가중치
├── package.json
└── README.md
```

---

## 설치 및 실행

```bash
git clone https://github.com/yourname/Turing-Deputy
cd Turing-Deputy
npm install

npx ts-node src/train/run.ts --data data/train.txt --epochs 10
```

서버가 준비되면 `http://localhost:3000`에서 웹 UI에 접속할 수 있습니다.

---

## 기술 스택

- TypeScript (Node.js)
- 직접 구현한 Tensor / Tokenizer / Transformer 구성 요소
- Web UI
- Express + SSE
- GPT-style Decoder-only Transformer

---

## 참고 자료

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

---

## 라이선스

MIT
