import { Tensor } from "../src/core/tensor"
import { EmbeddingLayer } from "./embedding"
import { TransformerBlock } from "./transformerBlock"

/**
 * Token ids를 임베딩한 뒤 Transformer blocks, final LayerNorm,
 * 그리고 vocab projection을 거쳐 logits를 생성하는 모델입니다.
 */
export class TuringDeputy {
  /**
   * 토큰 id를 입력 임베딩으로 변환하는 레이어입니다.
   */
  embedding: EmbeddingLayer

  /**
   * 순차적으로 적용할 Transformer block 목록입니다.
   */
  blocks: TransformerBlock[]

  /**
   * 최종 layer normalization의 scale 파라미터입니다.
   */
  normWeight: Tensor

  /**
   * 최종 layer normalization의 bias 파라미터입니다.
   */
  normBias: Tensor

  /**
   * 모델 출력을 vocabulary logits로 투영하는 weight입니다.
   *
   * shape: `[dModel, vocabSize]`
   */
  linear: Tensor

  /**
   * 모델의 주요 레이어와 파라미터를 초기화합니다.
   *
   * @param vocabSize vocabulary 크기
   * @param dModel 모델 hidden 차원 수
   * @param numHeads attention head 개수
   * @param dFF feed-forward hidden 차원 수
   * @param numLayers Transformer block 개수
   */
  constructor(vocabSize: number, dModel: number, numHeads: number, dFF: number, numLayers: number) {
    this.embedding = new EmbeddingLayer(vocabSize, dModel)
    this.blocks = Array.from({ length: numLayers }, () =>
      new TransformerBlock(dModel, numHeads, dFF)
    )
    this.normWeight = Tensor.ones([dModel])
    this.normBias = Tensor.zeros([dModel])
    this.linear = Tensor.randn([dModel, vocabSize])
  }

  /**
   * 입력 토큰 id 시퀀스를 logits로 변환합니다.
   *
   * 처리 순서:
   * 1. token embedding과 positional encoding을 합친 입력 표현 생성
   * 2. 모든 Transformer block 순차 적용
   * 3. 마지막 LayerNorm 적용
   * 4. vocab 크기로 선형 투영하여 logits 생성
   *
   * @param ids 입력 토큰 id 배열
   * @param mask self-attention에 사용할 optional mask 텐서
   * @returns shape가 `[seqLen, vocabSize]`인 logits 텐서
   */
  forward(ids: number[], mask?: Tensor): Tensor {
    let x = this.embedding.forward(ids)

    for (const block of this.blocks) {
      x = block.forward(x, mask)
    }

    x = x.layernorm(this.normWeight, this.normBias)
    return x.matmul(this.linear)
  }

  /**
   * 학습 시 업데이트해야 하는 모든 파라미터 텐서를 한 배열로 수집합니다.
   *
   * embedding, 각 block의 attention/FFN/LayerNorm 파라미터,
   * 그리고 마지막 layer normalization 및 출력 projection을 포함합니다.
   *
   * @returns 최적화 대상 파라미터 텐서 배열
   */
  getParams(): Tensor[] {
    const params: Tensor[] = []

    // 1. 토큰 임베딩
    params.push(this.embedding.tokenEmbedding.weight)

    // 2. 각 TransformerBlock의 파라미터
    for (const block of this.blocks) {
      // Attention
      params.push(block.attn.Wq, block.attn.Wk, block.attn.Wv, block.attn.Wo)
      // FFN
      params.push(block.ffn.Wup, block.ffn.Wdown, block.ffn.bUp, block.ffn.bDown)
      // LayerNorm
      params.push(block.norm1Weight, block.norm1Bias, block.norm2Weight, block.norm2Bias)
    }

    // 3. 마지막 LayerNorm + linear
    params.push(this.normWeight, this.normBias, this.linear)

    return params
  }

  /**
   * prompt 토큰 시퀀스에서 시작해 top-k 샘플링으로 새 토큰을 생성합니다.
   *
   * 매 step마다 전체 시퀀스에 대한 logits를 다시 계산하고, 마지막 위치의 logits에서
   * top-k 후보만 남긴 뒤 softmax 확률에 따라 다음 토큰을 샘플링합니다.
   *
   * @param promptIds 생성 시작에 사용할 prompt 토큰 id 배열
   * @param maxNewTokens 생성할 최대 토큰 수
   * @param topK 샘플링에 사용할 상위 logits 후보 개수
   * @returns prompt를 포함한 최종 토큰 id 시퀀스
   */
  generate(promptIds: number[], maxNewTokens: number, topK: number = 40): number[] {
    const ids = [...promptIds]
    for (let i = 0; i < maxNewTokens; i++) {
      const logits = this.forward(ids)
      const vocabSize = logits.shape[1]
      const lastRow = Array.from(logits.data.slice(-vocabSize))

      const indexed = lastRow
        .map((val, idx) => ({ val, idx }))
        .sort((a, b) => b.val - a.val)
        .slice(0, topK)

      const maxVal = indexed[0].val
      const exps = indexed.map(x => Math.exp(x.val - maxVal))
      const sumExp = exps.reduce((a, b) => a + b, 0)
      const probs = exps.map(x => x / sumExp)

      let rand = Math.random()
      let nextId = indexed[0].idx
      for (let j = 0; j < probs.length; j++) {
        rand -= probs[j]
        if (rand <= 0) { nextId = indexed[j].idx; break }
      }

      ids.push(nextId)
      if (nextId === this.embedding.tokenEmbedding.weight.shape[0] - 1) break // eosId
    }
    return ids
  }
}
