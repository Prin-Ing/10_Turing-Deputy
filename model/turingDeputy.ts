import { Tensor } from "../src/core/tensor"
import { EmbeddingLayer } from "./embedding"
import { TransformerBlock } from "./transformerBlock"

/**
 * Token ids를 임베딩한 뒤 Transformer blocks, final LayerNorm,
 * 그리고 vocab projection을 거쳐 logits를 생성하는 모델입니다.
 */
export class TuringDeputy {
  embedding: EmbeddingLayer
  blocks: TransformerBlock[]
  normWeight: Tensor
  normBias: Tensor
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
}
