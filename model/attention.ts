import { Tensor } from "../src/core/tensor"

/**
 * Scaled dot-product attention을 계산합니다.
 *
 * 입력 텐서는 마지막 두 축이 각각 `[seqLen, dHead]` 구조라고 가정하며,
 * 앞쪽 축이 있다면 배치 차원 또는 multi-head 차원으로 그대로 유지됩니다.
 * 내부적으로 `QK^T / sqrt(dK)`를 계산한 뒤 optional mask를 적용하고,
 * 마지막 축에 softmax를 취한 attention weight로 `V`를 가중합합니다.
 *
 * @param Q query 텐서
 * @param K key 텐서
 * @param V value 텐서
 * @param mask attention을 차단할 위치를 표시하는 마스크 텐서. `0`인 위치는 매우 작은 값으로 치환됩니다.
 * @returns attention 결과 텐서
 */
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

/**
 * Transformer의 multi-head self-attention 레이어입니다.
 *
 * 입력을 query, key, value 공간으로 각각 선형 투영한 뒤,
 * head 단위로 분리하여 scaled dot-product attention을 수행하고
 * 다시 하나의 표현으로 합쳐 최종 출력 투영을 적용합니다.
 */
class MultiHeadAttention {
  /**
   * attention head 개수입니다.
   */
  numHeads: number

  /**
   * 모델 전체 hidden 차원 수입니다.
   */
  dModel: number

  /**
   * 각 head가 담당하는 hidden 차원 수입니다.
   */
  dHead: number

  /**
   * query projection weight입니다.
   *
   * shape: `[dModel, dModel]`
   */
  Wq: Tensor  // [dModel, dModel]

  /**
   * key projection weight입니다.
   *
   * shape: `[dModel, dModel]`
   */
  Wk: Tensor

  /**
   * value projection weight입니다.
   *
   * shape: `[dModel, dModel]`
   */
  Wv: Tensor

  /**
   * attention 출력을 다시 모델 차원으로 투영하는 output weight입니다.
   *
   * shape: `[dModel, dModel]`
   */
  Wo: Tensor

  /**
   * 무작위 초기화된 multi-head attention 레이어를 생성합니다.
   *
   * @param dModel 모델 hidden 차원 수
   * @param numHeads attention head 개수
   */
  constructor(dModel: number, numHeads: number) {
    this.numHeads = numHeads
    this.dModel = dModel
    this.dHead = dModel / numHeads
    this.Wq = Tensor.randn([dModel, dModel])
    this.Wk = Tensor.randn([dModel, dModel])
    this.Wv = Tensor.randn([dModel, dModel])
    this.Wo = Tensor.randn([dModel, dModel])
  }

  /**
   * 입력 시퀀스에 대해 multi-head self-attention을 계산합니다.
   *
   * 처리 순서:
   * 1. 입력을 `Q`, `K`, `V`로 선형 투영
   * 2. `[seqLen, dModel]`을 `[numHeads, seqLen, dHead]`로 재배열
   * 3. head별 attention 계산
   * 4. head 출력을 다시 합쳐 `[seqLen, dModel]`로 복원
   * 5. 최종 output projection 적용
   *
   * @param x 입력 표현 텐서. 일반적으로 shape는 `[seqLen, dModel]`입니다.
   * @param mask attention 마스크 텐서
   * @returns self-attention이 적용된 출력 텐서
   */
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
