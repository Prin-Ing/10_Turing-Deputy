import { Tensor } from "../src/core/tensor"

/**
 * Two-layer feed-forward network used to project inputs up, apply GELU,
 * and project back to the model dimension.
 */
export class FeedForward {
  Wup: Tensor    // [dModel, dFF]
  Wdown: Tensor  // [dFF, dModel]
  bUp: Tensor    // [dFF]
  bDown: Tensor  // [dModel]

  /**
   * Initializes the feed-forward projection weights and biases.
   *
   * @param dModel Input and output feature dimension.
   * @param dFF Hidden feed-forward dimension.
   */
  constructor(dModel: number, dFF: number) {
    this.Wup = Tensor.randn([dModel, dFF])
    this.Wdown = Tensor.randn([dFF, dModel])
    this.bUp = Tensor.zeros([dFF])
    this.bDown = Tensor.zeros([dModel])
  }

  /**
   * Runs the feed-forward block: linear up-projection, GELU activation,
   * and linear down-projection.
   *
   * @param x Input tensor whose last dimension matches `dModel`.
   * @returns Output tensor projected back to the model dimension.
   */
  forward(x: Tensor): Tensor {
    // 이전: for i,j 루프 + biases[0][i]
    const hidden = x.matmul(this.Wup).add(this.bUp)

    // 이전: this.activation.activate(sum)
    const activated = hidden.gelu()

    // 이전: for i,j 루프 + biases[1][i]
    const out = activated.matmul(this.Wdown).add(this.bDown)

    return out
  }
}
