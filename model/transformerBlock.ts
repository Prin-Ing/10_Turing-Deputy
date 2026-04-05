import { Tensor } from "../src/core/tensor";
import { FeedForward } from "./feedForward"
import { MultiHeadAttention } from "./attention"

/**
 * Self-attention과 feed-forward network를 잔차 연결로 묶은 Transformer block입니다.
 *
 * 각 서브레이어 앞에 layer normalization을 적용하는 pre-layernorm 구조를 사용합니다.
 */
export class TransformerBlock {
  /**
   * multi-head self-attention 서브레이어입니다.
   */
  attn: MultiHeadAttention

  /**
   * position-wise feed-forward 서브레이어입니다.
   */
  ffn: FeedForward

  /**
   * attention 앞 layer normalization의 scale 파라미터입니다.
   */
  norm1Weight: Tensor

  /**
   * attention 앞 layer normalization의 bias 파라미터입니다.
   */
  norm1Bias: Tensor

  /**
   * FFN 앞 layer normalization의 scale 파라미터입니다.
   */
  norm2Weight: Tensor

  /**
   * FFN 앞 layer normalization의 bias 파라미터입니다.
   */
  norm2Bias: Tensor

  /**
   * Transformer block을 구성하는 attention, FFN, LayerNorm 파라미터를 초기화합니다.
   *
   * @param dModel 모델 hidden 차원 수
   * @param numHeads attention head 개수
   * @param dFF feed-forward hidden 차원 수
   */
  constructor(dModel: number, numHeads: number, dFF: number) {
    this.attn = new MultiHeadAttention(dModel, numHeads)
    this.ffn = new FeedForward(dModel, dFF)

    // LayerNorm 파라미터
    this.norm1Weight = Tensor.ones([dModel])
    this.norm1Bias = Tensor.zeros([dModel])
    this.norm2Weight = Tensor.ones([dModel])
    this.norm2Bias = Tensor.zeros([dModel])
  }

  /**
   * 입력 표현에 self-attention과 feed-forward sublayer를 차례대로 적용합니다.
   *
   * 각 단계는 `LayerNorm -> Sublayer -> Residual Add` 순서를 따릅니다.
   *
   * @param x 입력 텐서
   * @param mask attention 계산에 사용할 optional 마스크 텐서
   * @returns Transformer block을 지난 출력 텐서
   */
  forward(x: Tensor, mask?: Tensor): Tensor {
    // 1. Pre-LayerNorm → Attention → 잔차 연결
    const x1 = x.add(this.attn.forward(x.layernorm(this.norm1Weight, this.norm1Bias), mask))

    // 2. Pre-LayerNorm → FFN → 잔차 연결
    const x2 = x1.add(this.ffn.forward(x1.layernorm(this.norm2Weight, this.norm2Bias)))

    return x2
  }
}
