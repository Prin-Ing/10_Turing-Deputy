import { BPETokenizer } from './tokenizer'
import { DataLoader } from './dataloader'
import { TuringDeputy } from '../../model/turingDeputy'
import { Optimizer } from './optimizer'

/**
 * 토크나이저, 모델, 데이터 로더, 옵티마이저를 연결해
 * next-token prediction 학습 루프를 실행하는 간단한 스크립트입니다.
 */

// 하이퍼파라미터
const vocabSize = 1000
const dModel = 384
const numHeads = 6
const dFF = 1536
const numLayers = 6
const contextLength = 256
const batchSize = 4
const lr = 3e-4
const maxSteps = 1000

// 1. 초기화
const tokenizer = new BPETokenizer()
tokenizer.load('checkpoints/tokenizer.json')

const model = new TuringDeputy(vocabSize, dModel, numHeads, dFF, numLayers)
const loader = new DataLoader(tokenizer, contextLength, batchSize)
loader.load('train.txt')

const optimizer = new Optimizer(model.getParams(), lr)

// 2. 학습 루프
for (let step = 0; step < maxSteps; step++) {
  const { x, y } = loader.nextBatch()

  // 배치 평균 loss 계산
  let totalLoss = 0
  for (let b = 0; b < batchSize; b++) {
    const logits = model.forward(x[b])
    const loss = logits.crossEntropy(y[b])
    loss.backprop()
    totalLoss += loss.data[0]
  }

  optimizer.step()

  // grad 초기화
  for (const p of model.getParams()) {
    p.grad = null
    
  }

  if (step % 10 === 0) {
    console.log(`step ${step}, loss: ${(totalLoss / batchSize).toFixed(4)}`)
  }



}
