import { Tensor } from "../src/core/tensor"

function scaledDotProductAttention(Q: Tensor, K: Tensor, V: Tensor, mask?: Tensor): Tensor {
  const dK = Q.shape[Q.shape.length - 1]

  // 1. Q · K^T → [seqLen, seqLen]
  const scores = Q.matmul(K.transpose())

  // 2. √dK 로 나누기
  const scale = new Tensor(
    new Float32Array(scores.data.length).fill(1 / Math.sqrt(dK)),
    scores.shape
  )
  const scaled = scores.mul(scale)

  // 3. mask 적용
  if (mask) {
    for (let i = 0; i < scaled.data.length; i++) {
      if (mask.data[i] === 0) {
        scaled.data[i] = -1e9
      }
    }
  }

  // 4. 마지막 축 기준 softmax
  const attnWeights = scaled.softmax(-1)

  // 5. · V
  return attnWeights.matmul(V)
}

class MultiHeadAttention {
  numHeads: number
  dModel: number
  dHead: number
  Wq: Tensor  // [dModel, dModel]
  Wk: Tensor
  Wv: Tensor
  Wo: Tensor

  constructor(dModel: number, numHeads: number) {
    this.numHeads = numHeads
    this.dModel = dModel
    this.dHead = dModel / numHeads
    this.Wq = Tensor.randn([dModel, dModel])
    this.Wk = Tensor.randn([dModel, dModel])
    this.Wv = Tensor.randn([dModel, dModel])
    this.Wo = Tensor.randn([dModel, dModel])
  }

  forward(x: Tensor, mask?: Tensor): Tensor {
    const seqLen = x.shape[0]

    // 1. Q, K, V
    const Q = x.matmul(this.Wq)
    const K = x.matmul(this.Wk)
    const V = x.matmul(this.Wv)

    // 2. 쪼개기 [seqLen, dModel] → [numHeads, seqLen, dHead]
    const Qh = Q.reshape([seqLen, this.numHeads, this.dHead]).transpose([1, 0, 2])
    const Kh = K.reshape([seqLen, this.numHeads, this.dHead]).transpose([1, 0, 2])
    const Vh = V.reshape([seqLen, this.numHeads, this.dHead]).transpose([1, 0, 2])

    // 3. Attention
    const attn = scaledDotProductAttention(Qh, Kh, Vh, mask)

    // 4. 합치기 [numHeads, seqLen, dHead] → [seqLen, dModel]
    const out = attn.transpose([1, 0, 2]).reshape([seqLen, this.dModel])

    // 5. Wo
    return out.matmul(this.Wo)
  }
}