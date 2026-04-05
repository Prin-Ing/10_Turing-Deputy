import { Tensor } from "../src/core/tensor";
import { FeedForward } from "./feedForward"
import { MultiHeadAttention } from "./attention"

export class TransformerBlock {

  attn: MultiHeadAttention
  ffn: FeedForward
  norm1Weight: Tensor
  norm1Bias: Tensor
  norm2Weight: Tensor
  norm2Bias: Tensor


  constructor(dModel: number, numHeads: number, dFF: number) {
    this.attn = new MultiHeadAttention(dModel, numHeads)
    this.ffn = new FeedForward(dModel, dFF)

    // LayerNorm 파라미터
    this.norm1Weight = Tensor.ones([dModel])
    this.norm1Bias = Tensor.zeros([dModel])
    this.norm2Weight = Tensor.ones([dModel])
    this.norm2Bias = Tensor.zeros([dModel])
  }

  forward(x: Tensor, mask?: Tensor): Tensor {
    // 1. Pre-LayerNorm → Attention → 잔차 연결
    const x1 = x.add(this.attn.forward(x.layernorm(this.norm1Weight, this.norm1Bias), mask))

    // 2. Pre-LayerNorm → FFN → 잔차 연결
    const x2 = x1.add(this.ffn.forward(x1.layernorm(this.norm2Weight, this.norm2Bias)))

    return x2
  }
}
