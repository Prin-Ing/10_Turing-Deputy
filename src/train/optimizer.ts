import { Tensor } from "../core/tensor"

/**
 * 파라미터 텐서들에 대해 AdamW 방식으로 업데이트를 수행하는 옵티마이저입니다.
 *
 * 각 파라미터마다 1차 모멘트 `m`, 2차 모멘트 `v`를 유지하고,
 * bias correction을 적용한 뒤 weight decay와 함께 값을 갱신합니다.
 */
export class Optimizer {
  /**
   * 최적화 대상 파라미터 텐서 목록입니다.
   */
  params: Tensor[]

  /**
   * 기본 학습률입니다.
   */
  lr: number

  /**
   * AdamW의 weight decay 계수입니다.
   */
  weightDecay: number

  /**
   * 1차 모멘트 지수이동평균 계수입니다.
   */
  beta1: number

  /**
   * 2차 모멘트 지수이동평균 계수입니다.
   */
  beta2: number

  /**
   * 0으로 나누는 상황을 방지하기 위한 작은 상수입니다.
   */
  eps: number

  /**
   * 현재 optimization step 수입니다.
   */
  t: number

  /**
   * 각 파라미터의 1차 모멘트 버퍼입니다.
   */
  m: Float32Array[]

  /**
   * 각 파라미터의 2차 모멘트 버퍼입니다.
   */
  v: Float32Array[]


  /**
   * 최적화 대상 파라미터와 하이퍼파라미터를 초기화합니다.
   *
   * @param params 업데이트할 파라미터 텐서 목록
   * @param lr 학습률
   * @param weightDecay AdamW weight decay 계수
   */
  constructor(params: Tensor[], lr: number, weightDecay: number = 0.1) {
    this.params = params
    this.lr = lr
    this.weightDecay = weightDecay
    this.beta1 = 0.9
    this.beta2 = 0.999
    this.eps = 1e-8
    this.t = 0
    this.m = params.map(p => new Float32Array(p.data.length))
    this.v = params.map(p => new Float32Array(p.data.length))
  }

  /**
   * 현재 gradient를 사용해 모든 파라미터를 한 step 업데이트합니다.
   *
   * gradient가 없는 파라미터는 건너뛰며, 각 원소에 대해
   * 1. 1차/2차 모멘트 갱신
   * 2. bias correction 적용
   * 3. Adam 업데이트
   * 4. weight decay 적용
   * 순서로 값을 수정합니다.
   */
  step(): void {
    this.t++

    for (let i = 0; i < this.params.length; i++) {
      const param = this.params[i]
      const grad = param.grad
      if (!grad) continue  // gradient 없으면 스킵

      for (let j = 0; j < param.data.length; j++) {
        this.m[i][j] = this.beta1 * this.m[i][j] + (1 - this.beta1) * grad[j]
        this.v[i][j] = this.beta2 * this.v[i][j] + (1 - this.beta2) * grad[j] ** 2

        const mHat = this.m[i][j] / (1 - this.beta1 ** this.t)
        const vHat = this.v[i][j] / (1 - this.beta2 ** this.t)

        param.data[j] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps)
        param.data[j] -= this.lr * this.weightDecay * param.data[j]
      }
    }
  }
}
