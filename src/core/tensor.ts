/**
 * 연속된 `Float32Array` 버퍼를 기반으로 다차원 텐서를 표현합니다.
 *
 * 내부 데이터는 항상 1차원 버퍼에 저장되며, `shape`와 `strides`를 이용해
 * 다차원 좌표를 flat index로 변환합니다.
 *
 * 주요 속성:
 * - `data`: 실제 값이 저장된 1차원 버퍼
 * - `shape`: 각 차원의 길이
 * - `strides`: 각 차원의 메모리 보폭
 * - `grad`: 역전파 시 사용할 gradient 버퍼
 * - `_deps`: 현재 텐서가 의존하는 부모 텐서 목록
 * - `_backward`: 현재 텐서에서 부모 텐서로 gradient를 전파하는 함수
 *
 * 이 클래스는 텐서 생성, 브로드캐스팅 기반 원소 연산, 행렬 곱,
 * 정규화 계열 연산과 간단한 역전파 기능을 위한 최소 기능을 제공합니다.
 */
export class Tensor {
  /**
   * 텐서의 원소가 저장된 연속 메모리 버퍼입니다.
   */
  data: Float32Array;

  /**
   * 텐서의 각 차원 크기입니다.
   *
   * 예: `[2, 3]`은 2행 3열 텐서를 뜻합니다.
   */
  shape: number[];

  /**
   * 각 차원에서 다음 원소로 이동할 때 건너뛰어야 하는 원소 수입니다.
   *
   * 예: `shape = [2, 3, 4]`이면 `strides = [12, 4, 1]`입니다.
   */
  strides: number[];

  /**
   * 역전파 결과를 저장하는 그래디언트 버퍼입니다.
   *
   * 아직 그래디언트가 계산되지 않았다면 `null`입니다.
   */
  grad: Float32Array | null;

  /**
   * 현재 텐서의 local gradient를 부모 텐서들에 누적하는 backward 함수입니다.
   *
   * 각 연산 메서드는 결과 텐서를 만들 때 이 함수를 설정해 두고,
   * `backprop()` 호출 시 역위상 순서로 실행됩니다.
   */
  _backward: () => void

  /**
   * 현재 텐서가 직접 의존하는 부모 텐서 목록입니다.
   *
   * 계산 그래프를 위상 정렬할 때 사용됩니다.
   */
  _deps: Tensor[]

  /**
   * 주어진 버퍼와 형태로 텐서를 생성합니다.
   *
   * @param data 텐서 원소가 저장된 1차원 `Float32Array`
   * @param shape 텐서의 다차원 형태. 예를 들어 `[2, 3, 4]`는 `2 x 3 x 4` 텐서를 의미합니다.
   *
   * @example
   * const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
   */
  constructor(data: Float32Array, shape: number[]) {
    this.data = data;
    this.shape = shape;
    this.strides = Tensor.computeStrides(shape);
    this.grad = null;
    this._backward = () => { }
    this._deps = []
  }

  /**
   * `shape`에 대응하는 row-major stride 배열을 계산합니다.
   *
   * 마지막 차원의 stride는 항상 `1`이며, 앞 차원으로 갈수록
   * 뒤쪽 차원 크기의 누적 곱이 됩니다. 스칼라 shape(`[]`)는
   * stride도 빈 배열(`[]`)로 취급합니다.
   *
   * @param shape stride를 계산할 텐서 shape
   * @returns 입력 shape에 대한 stride 배열
   *
   * @example
   * Tensor.computeStrides([2, 3, 4]); // [12, 4, 1]
   */
  static computeStrides(shape: number[]): number[] {
    if (shape.length === 0) {
      return [];
    }

    const strides = new Array(shape.length);
    strides[shape.length - 1] = 1;
    for (let i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

  /**
   * 모든 원소가 `0`인 텐서를 생성합니다.
   *
   * @param shape 생성할 텐서의 shape
   * @returns 지정한 shape과 같은 크기의 zero tensor
   *
   * @example
   * const bias = Tensor.zeros([128]);
   */
  static zeros(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size).fill(0), shape);
  }

  /**
   * 모든 원소가 `1`인 텐서를 생성합니다.
   *
   * @param shape 생성할 텐서의 shape
   * @returns 지정한 shape과 같은 크기의 one tensor
   *
   * @example
   * const mask = Tensor.ones([2, 2]);
   */
  static ones(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size).fill(1), shape);
  }

  /**
   * 표준 정규분포(`mean=0`, `std=1`)를 따르는 난수로 텐서를 채웁니다.
   *
   * 내부적으로 Box-Muller 변환을 사용해 각 원소를 생성합니다.
   * 가중치 초기화처럼 평균 0 주변의 난수가 필요한 경우에 사용할 수 있습니다.
   *
   * @param shape 생성할 텐서의 shape
   * @returns 정규분포 난수로 채워진 텐서
   */
  static randn(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      const u1 = Math.max(Math.random(), 1e-7);
      const u2 = Math.random();
      data[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return new Tensor(data, shape);
  }

  /**
   * 두 shape의 배치 차원을 브로드캐스팅 규칙에 따라 결합합니다.
   *
   * 뒤쪽 축부터 비교하면서 각 차원이 같거나 둘 중 하나가 `1`이면
   * 더 큰 값을 결과 차원으로 선택합니다.
   *
   * @param leftShape 왼쪽 배치 shape
   * @param rightShape 오른쪽 배치 shape
   * @returns 브로드캐스팅된 배치 shape
   * @throws {Error} 브로드캐스팅이 불가능한 shape 조합이면 예외를 던집니다.
   *
   * @example
   * Tensor.broadcastShapes([4, 1, 8], [1, 3, 8]); // [4, 3, 8]
   */
  static broadcastShapes(leftShape: number[], rightShape: number[]): number[] {
    const maxRank = Math.max(leftShape.length, rightShape.length);
    const result = new Array(maxRank);

    for (let i = 0; i < maxRank; i++) {
      const leftDim = leftShape[leftShape.length - 1 - i] ?? 1;
      const rightDim = rightShape[rightShape.length - 1 - i] ?? 1;

      if (leftDim !== rightDim && leftDim !== 1 && rightDim !== 1) {
        throw new Error("브로드캐스팅할 수 없는 shape입니다.");
      }

      result[maxRank - 1 - i] = Math.max(leftDim, rightDim);
    }

    return result;
  }

  /**
   * shape에 포함된 모든 좌표를 row-major 순서로 순회합니다.
   *
   * 스칼라 shape(`[]`)는 빈 인덱스 배열을 한 번만 전달합니다.
   *
   * @param shape 순회할 대상 shape
   * @param callback 각 좌표마다 호출할 함수. 현재 좌표가 새 배열로 전달됩니다.
   */
  static forEachIndex(shape: number[], callback: (indices: number[]) => void): void {
    if (shape.length === 0) {
      callback([]);
      return;
    }

    const indices = new Array(shape.length).fill(0);

    while (true) {
      callback([...indices]);

      // 마지막 축부터 1씩 증가시키며 row-major 순서를 만듭니다.
      let dim = shape.length - 1;
      while (dim >= 0) {
        indices[dim] += 1;
        if (indices[dim] < shape[dim]) {
          break;
        }

        indices[dim] = 0;
        dim -= 1;
      }

      if (dim < 0) {
        return;
      }
    }
  }

  /**
   * 브로드캐스팅된 배치 인덱스를 원본 배치 shape에 맞는 인덱스로 변환합니다.
   *
   * 원본 shape의 차원이 `1`이면 해당 축은 브로드캐스팅된 축이므로 항상 `0`을 사용합니다.
   * rank가 더 작은 텐서는 앞쪽에 가상의 차원 `1`이 붙어 있다고 가정합니다.
   *
   * @param targetIndices 브로드캐스팅 결과 기준 배치 인덱스
   * @param sourceShape 원본 배치 shape
   * @returns 원본 텐서에서 사용할 배치 인덱스
   */
  static projectBroadcastIndices(targetIndices: number[], sourceShape: number[]): number[] {
    const offset = targetIndices.length - sourceShape.length;

    return sourceShape.map((dim, i) => {
      if (dim === 1) {
        return 0;
      }

      return targetIndices[i + offset];
    });
  }

  /**
   * 다차원 좌표에 해당하는 원소 값을 읽습니다.
   *
   * 내부적으로 `indices[i] * strides[i]`의 합을 계산해
   * 1차원 버퍼의 실제 위치를 구합니다.
   *
   * @param indices 각 차원의 인덱스 배열
   * @returns 지정한 좌표의 스칼라 값
   *
   * @example
   * const value = tensor.get([1, 2]);
   */
  get(indices: number[]): number {
    const flatIndex = indices.reduce((acc, idx, i) => acc + idx * this.strides[i], 0);
    return this.data[flatIndex];
  }

  /**
   * 다차원 좌표에 해당하는 위치에 값을 기록합니다.
   *
   * 내부 메모리는 1차원 버퍼이므로 전달된 좌표를 stride를 사용해
   * flat index로 변환한 뒤 값을 기록합니다.
   *
   * @param indices 각 차원의 인덱스 배열
   * @param value 기록할 스칼라 값
   *
   * @example
   * tensor.set([1, 2], 3.1415);
   */
  set(indices: number[], value: number): void {
    const flatIndex = indices.reduce((acc, idx, i) => acc + idx * this.strides[i], 0);
    this.data[flatIndex] = value;
  }

  /**
   * 동일한 데이터 버퍼를 유지한 채 텐서의 shape만 변경합니다.
   *
   * 새 텐서는 기존 `data`를 그대로 공유하므로 복사가 아니라 view에 가깝습니다.
   * 원소 수가 달라지는 reshape는 허용하지 않습니다.
   *
   * @param newShape 변경할 대상 shape
   * @returns 같은 데이터를 공유하는 새로운 텐서 뷰
   * @throws {Error} 기존 원소 수와 새 원소 수가 다르면 예외를 던집니다.
   */
  reshape(newShape: number[]): Tensor {
    const oldSize = this.shape.reduce((pre, crr) => pre * crr, 1);
    const newSize = newShape.reduce((a, b) => a * b, 1);

    if (oldSize !== newSize) throw new Error("shape의 크기가 맞지 않습니다.");
    return new Tensor(this.data, newShape);
  }

  /**
   * 두 텐서에 브로드캐스팅 기반 원소별 덧셈을 적용합니다.
   *
   * 결과 shape는 두 텐서 shape의 브로드캐스팅 결과로 결정됩니다.
   * 각 결과 좌표는 브로드캐스팅 규칙에 맞는 원본 좌표로 투영한 뒤 계산합니다.
   * 또한 결과 텐서에 계산 그래프 정보를 기록해, 역전파 시 출력 gradient를
   * 두 입력 텐서의 gradient 버퍼에 누적합니다.
   *
   * @param other 더할 대상 텐서
   * @returns 원소별 덧셈 결과 텐서
   * @throws {Error} 두 shape가 브로드캐스팅되지 않으면 예외를 던집니다.
   */
  add(other: Tensor): Tensor {
    const resultShape = Tensor.broadcastShapes(this.shape, other.shape);
    const result = Tensor.zeros(resultShape);

    Tensor.forEachIndex(resultShape, (indices) => {
      const aIndices = Tensor.projectBroadcastIndices(indices, this.shape);
      const bIndices = Tensor.projectBroadcastIndices(indices, other.shape);

      result.set(indices, this.get(aIndices) + other.get(bIndices));
    });
    result._deps = [this, other]
    result._backward = () => {
      // result.grad를 this.grad와 other.grad에 누적
      if (!result.grad) return

      if (!this.grad) this.grad = new Float32Array(this.data.length)
      if (!other.grad) other.grad = new Float32Array(other.data.length)

      for (let i = 0; i < this.data.length; i++) {
        this.grad[i] += result.grad[i]
      }
      for (let i = 0; i < other.data.length; i++) {
        other.grad[i] += result.grad[i]
      }
    }
    return result;
  }

  /**
   * 두 텐서에 브로드캐스팅 기반 원소별 곱셈을 적용합니다.
   *
   * 결과 shape는 두 텐서 shape의 브로드캐스팅 결과로 결정됩니다.
   * 각 결과 좌표는 브로드캐스팅 규칙에 맞는 원본 좌표로 투영한 뒤 계산합니다.
   *
   * 결과 텐서는 계산 그래프에 부모 텐서를 등록하며, 역전파 시
   * 각 입력에 대해 상대 텐서 값을 곱한 gradient를 계산해 누적합니다.
   *
   * @param other 곱할 대상 텐서
   * @returns 원소별 곱셈 결과 텐서
   * @throws {Error} 두 shape가 브로드캐스팅되지 않으면 예외를 던집니다.
   */
  mul(other: Tensor): Tensor {
    const resultShape = Tensor.broadcastShapes(this.shape, other.shape);
    const result = Tensor.zeros(resultShape);

    Tensor.forEachIndex(resultShape, (indices) => {
      const aIndices = Tensor.projectBroadcastIndices(indices, this.shape);
      const bIndices = Tensor.projectBroadcastIndices(indices, other.shape);

      result.set(indices, this.get(aIndices) * other.get(bIndices));
    });
    result._deps = [this, other]
    result._backward = () => {
      if (!result.grad) return
      if (!this.grad) this.grad = new Float32Array(this.data.length)
      if (!other.grad) other.grad = new Float32Array(other.data.length)

      for (let i = 0; i < this.data.length; i++) {
        this.grad[i] += result.grad[i] * other.data[i]
      }
      for (let i = 0; i < other.data.length; i++) {
        other.grad[i] += result.grad[i] * this.data[i]
      }
    }
    return result
  }

  /**
   * 두 텐서의 행렬 곱을 계산합니다.
   *
   * 연산 규칙은 NumPy의 `matmul`과 비슷하게 동작합니다.
   * 1차원 입력은 계산 편의를 위해 내부적으로 2차원으로 승격한 뒤,
   * 결과에서 불필요한 축을 다시 제거합니다.
   *
   * - `1D x 1D` -> 스칼라(`[]`)
   * - `2D x 1D` -> 벡터
   * - `1D x 2D` -> 벡터
   * - `2D x 2D` -> 행렬
   * - `N-D x M-D` -> 마지막 두 축을 행렬로 보고, 앞 축은 배치 차원으로 브로드캐스팅
   *
   * shape 규칙:
   * - 왼쪽 텐서의 마지막 축 길이와 오른쪽 텐서의 뒤에서 두 번째 축 길이가 같아야 합니다.
   * - 두 텐서의 배치 차원은 브로드캐스팅 가능해야 합니다.
   *
   * 결과 텐서는 계산 그래프에 부모 텐서를 등록하며, 역전파 시
   * `dX = dZ.matmul(Y^T)`, `dY = X^T.matmul(dZ)` 형태로 gradient를 계산한 뒤
   * 각 입력 텐서의 gradient 버퍼에 누적합니다.
   *
   * @param otherTensor 오른쪽 피연산자 텐서
   * @returns 입력 형식에 맞게 축이 정리된 행렬 곱 결과 텐서
   * @throws {Error} 입력이 벡터 미만이거나, 내부 차원 또는 배치 브로드캐스팅이 맞지 않으면 예외를 던집니다.
   *
   * @example
   * const a = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
   * const b = new Tensor(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]);
   * const c = a.matmul(b); // shape: [2, 2]
   */
  matmul(otherTensor: Tensor): Tensor {
    if (this.shape.length === 0 || otherTensor.shape.length === 0) {
      throw new Error("matmul은 최소 1차원 텐서가 필요합니다.");
    }

    const leftIsVector = this.shape.length === 1;
    const rightIsVector = otherTensor.shape.length === 1;

    const leftShape = leftIsVector ? [1, this.shape[0]] : [...this.shape];
    const rightShape = rightIsVector ? [otherTensor.shape[0], 1] : [...otherTensor.shape];

    const rows = leftShape[leftShape.length - 2];
    const inner = leftShape[leftShape.length - 1];
    const otherRows = rightShape[rightShape.length - 2];
    const cols = rightShape[rightShape.length - 1];

    if (inner !== otherRows) {
      throw new Error("matmul의 내부 차원이 맞지 않습니다.");
    }

    const leftBatchShape = leftShape.slice(0, -2);
    const rightBatchShape = rightShape.slice(0, -2);
    const batchShape = Tensor.broadcastShapes(leftBatchShape, rightBatchShape);

    const resultShape = [...batchShape];
    if (!leftIsVector) {
      resultShape.push(rows);
    }
    if (!rightIsVector) {
      resultShape.push(cols);
    }

    const resultSize = resultShape.reduce((acc, dim) => acc * dim, 1);
    const result = new Tensor(new Float32Array(resultSize), resultShape);

    Tensor.forEachIndex(batchShape, (batchIndices) => {
      const leftBatchIndices = Tensor.projectBroadcastIndices(batchIndices, leftBatchShape);
      const rightBatchIndices = Tensor.projectBroadcastIndices(batchIndices, rightBatchShape);

      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          let sum = 0;

          // 각 배치 위치에서 (row, col) 원소를 만들기 위해
          // inner 축을 따라 내적을 계산합니다.
          for (let k = 0; k < inner; k++) {
            const leftIndices = leftIsVector
              ? [k]
              : [...leftBatchIndices, row, k];
            const rightIndices = rightIsVector
              ? [k]
              : [...rightBatchIndices, k, col];

            sum += this.get(leftIndices) * otherTensor.get(rightIndices);
          }

          const resultIndices = [...batchIndices];
          if (!leftIsVector) {
            resultIndices.push(row);
          }
          if (!rightIsVector) {
            resultIndices.push(col);
          }

          result.set(resultIndices, sum);
        }
      }
    });

    result._deps = [this, otherTensor]
    result._backward = () => {
      if (!result.grad) return
      const dZ = new Tensor(result.grad, result.shape)

      // dL/dX = dL/dZ · Yᵀ
      if (!this.grad) this.grad = new Float32Array(this.data.length)
      const dX = dZ.matmul(otherTensor.transpose())
      for (let i = 0; i < this.grad.length; i++) {
        this.grad[i] += dX.data[i]
      }

      // dL/dY = Xᵀ · dL/dZ
      if (!otherTensor.grad) otherTensor.grad = new Float32Array(otherTensor.data.length)
      const dY = this.transpose().matmul(dZ)
      for (let i = 0; i < otherTensor.grad.length; i++) {
        otherTensor.grad[i] += dY.data[i]
      }
    }

    return result;
  }

  /**
   * 지정한 축을 따라 softmax를 적용한 새 텐서를 반환합니다.
   *
   * 기본값은 마지막 축(`axis = -1`)이며, 해당 축을 제외한 나머지 좌표마다
   * 독립적인 확률 분포를 계산합니다. 수치 안정성을 위해 각 구간에서 최댓값을 먼저 빼고
   * `exp`를 계산합니다.
   *
   * 결과 텐서는 원본과 같은 shape를 유지하며, 지정한 축을 따라 더한 값이 `1`에 가까워집니다.
   *
   * @param axis softmax를 적용할 축. 음수면 뒤에서부터 계산합니다.
   * @returns 지정한 축 기준 softmax 확률값으로 변환된 텐서
   *
   * @example
   * const logits = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
   * const probs = logits.softmax();
   */
  softmax(axis: number = -1): Tensor {
    const ax = axis < 0 ? this.shape.length + axis : axis
    const axisSize = this.shape[ax]
    const result = Tensor.zeros(this.shape)

    // axis 제외한 나머지 인덱스 순회
    Tensor.forEachIndex(
      this.shape.filter((_, i) => i !== ax),
      (outerIndices) => {
        // 1. 최댓값 구하기
        let maxVal = -Infinity
        for (let i = 0; i < axisSize; i++) {
          const idx = [...outerIndices.slice(0, ax), i, ...outerIndices.slice(ax)]
          maxVal = Math.max(maxVal, this.get(idx))
        }

        // 2. exp 합 구하기
        let expSum = 0
        for (let i = 0; i < axisSize; i++) {
          const idx = [...outerIndices.slice(0, ax), i, ...outerIndices.slice(ax)]
          expSum += Math.exp(this.get(idx) - maxVal)
        }

        // 3. 각 원소 나누기
        for (let i = 0; i < axisSize; i++) {
          const idx = [...outerIndices.slice(0, ax), i, ...outerIndices.slice(ax)]
          result.set(idx, Math.exp(this.get(idx) - maxVal) / expSum)
        }
      }
    )

    return result
  }

  /**
   * 텐서 전체를 기준으로 layer normalization을 적용한 새 텐서를 반환합니다.
   *
   * 현재 구현은 특정 축을 기준으로 나누지 않고, 내부 버퍼 전체의 평균과 분산을 사용해
   * 정규화합니다. 이후 각 원소에 대해 `weight`로 스케일을 적용하고 `bias`를 더합니다.
   *
   * 수식은 다음과 같습니다.
   * `output[i] = weight[i] * ((x[i] - mean) / sqrt(variance + eps)) + bias[i]`
   *
   * 구현 제약:
   * - `weight`와 `bias`는 현재 텐서와 같은 길이의 버퍼를 가져야 합니다.
   * - transformer에서 흔히 쓰는 "마지막 축 기준 layer norm"과는 아직 다릅니다.
   * - 수치 안정성을 위해 `eps = 1e-5`를 사용합니다.
   *
   * @param weight 정규화된 값에 곱할 scale 텐서
   * @param bias 정규화 후 더할 shift 텐서
   * @returns layer normalization이 적용된 새 텐서
   *
   * @example
   * const x = new Tensor(new Float32Array([1, 2, 3]), [3]);
   * const gamma = Tensor.ones([3]);
   * const beta = Tensor.zeros([3]);
   * const y = x.layernorm(gamma, beta);
   */
  layernorm(weight: Tensor, bias: Tensor): Tensor {
    const average = this.data.reduce((a, b) => a + b, 0) / this.data.length;

    const variance = this.data.reduce((a, b) => a + Math.pow(b - average, 2), 0) / this.data.length;

    const data = new Float32Array(this.data.length);

    // 각 원소를 정규화한 뒤 gamma, beta를 적용합니다.
    for (let i = 0; i < this.data.length; i++) {
      data[i] =
        weight.data[i] * (this.data[i] - average) / Math.sqrt(variance + 1e-5) +
        bias.data[i];
    }

    return new Tensor(data, this.shape);
  }

  /**
   * 텐서의 축 순서를 재배열한 새 텐서를 반환합니다.
   *
   * `axes`를 지정하면 해당 순서대로 축을 재배치합니다.
   * `axes`를 생략하면 마지막 두 축만 서로 바꿉니다. 따라서
   * 2차원 텐서에서는 일반적인 행렬 전치와 동일하게 동작하고,
   * 3차원 이상에서는 배치 축은 유지한 채 마지막 두 축만 뒤집습니다.
   *
   * 내부적으로 원본 텐서의 모든 좌표를 순회하면서,
   * 새 축 순서에 맞는 위치로 값을 복사합니다.
   *
   * @param axes 새 축 순서를 나타내는 순열 배열
   * @returns 축 순서가 재배열된 새 텐서
   *
   * @example
   * const x = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
   * const y = x.transpose(); // shape: [3, 2]
   *
   * @example
   * const z = new Tensor(new Float32Array(24), [2, 3, 4]);
   * const t = z.transpose([1, 0, 2]); // shape: [3, 2, 4]
   */
  transpose(axes?: number[]): Tensor {
    const ndim = this.shape.length

    // axes 없으면 마지막 두 축만 뒤집기
    const perm = axes ?? [
      ...Array.from({ length: ndim - 2 }, (_, i) => i),
      ndim - 1,
      ndim - 2
    ]

    const newShape = perm.map(i => this.shape[i])
    const result = Tensor.zeros(newShape)

    Tensor.forEachIndex(this.shape, (indices) => {
      const newIndices = perm.map(i => indices[i])
      result.set(newIndices, this.get(indices))
    })

    return result
  }
  /**
   * Applies ReLU element-wise by clamping negative values to 0.
   * Returns a new tensor with the same shape.
   */
  relu(): Tensor {
    const data = this.data.map(x => Math.max(0, x))
    return new Tensor(new Float32Array(data), this.shape)
  }

  /**
   * tanh 기반 GELU 근사를 원소별로 적용한 새 텐서를 반환합니다.
   *
   * forward에서는 `0.5 * x * (1 + tanh(u))` 형태의 근사를 사용하고,
   * backward에서는 각 원소의 GELU 도함수를 계산해 입력 텐서의 gradient에
   * `result.grad[i] * geluPrime`를 누적합니다.
   *
   * @returns GELU가 적용된 새 텐서
   */
  gelu(): Tensor {
    const data = this.data.map(x =>
      0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
    )
    const result = new Tensor(new Float32Array(data), this.shape)

    result._deps = [this]
    result._backward = () => {
      if (!result.grad) return
      if (!this.grad) this.grad = new Float32Array(this.data.length)

      for (let i = 0; i < this.data.length; i++) {
        const x = this.data[i]
        const u = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)
        const tanhU = Math.tanh(u)
        const uPrime = Math.sqrt(2 / Math.PI) * (1 + 3 * 0.044715 * x ** 2)
        const geluPrime = 0.5 * (1 + tanhU) + 0.5 * x * (1 - tanhU ** 2) * uPrime
        this.grad[i] += result.grad[i] * geluPrime
      }
    }
    return result
  }

  /**
   * 현재 텐서를 시작점으로 계산 그래프 전체에 대해 역전파를 수행합니다.
   *
   * 먼저 `_deps`를 따라 그래프를 위상 정렬한 뒤, 시작 텐서의 gradient를
   * 모두 `1`로 초기화하고 역순으로 각 텐서의 `_backward()`를 실행합니다.
   * 각 `_backward()`는 gradient를 덮어쓰지 않고 누적하도록 설계되어 있어,
   * 하나의 텐서로 여러 경로가 합류하는 계산 그래프도 처리할 수 있습니다.
   * 일반적으로 loss 텐서에서 호출하는 것을 가정합니다.
   */
  backprop(): void {
    const topo: Tensor[] = []
    const visited = new Set<Tensor>()

    const build = (t: Tensor) => {
      if (visited.has(t)) return
      visited.add(t)
      for (const dep of t._deps) build(dep)
      topo.push(t)
    }

    build(this)

    // 시작 텐서의 grad를 1로 초기화
    this.grad = new Float32Array(this.data.length).fill(1)

    // 역순으로 backward 실행
    for (const t of topo.reverse()) {
      t._backward()
    }
  }

  crossEntropy(targets: number[]): Tensor {
    // 1. softmax로 확률 계산
    const probs = this.softmax(-1)

    // 2. 각 위치에서 정답 토큰의 확률만 뽑아서 -log
    const seqLen = this.shape[0]
    let loss = 0
    for (let i = 0; i < seqLen; i++) {
      const prob = probs.data[i * this.shape[1] + targets[i]]
      loss += -Math.log(prob + 1e-9)  // 1e-9 → log(0) 방지
    }
    loss /= seqLen

    const result = new Tensor(new Float32Array([loss]), [1])

    // backward
    result._deps = [this]
    result._backward = () => {
      if (!this.grad) this.grad = new Float32Array(this.data.length)
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < this.shape[1]; j++) {
          const idx = i * this.shape[1] + j
          const p = probs.data[idx]
          this.grad[idx] += (p - (j === targets[i] ? 1 : 0)) / seqLen
        }
      }
    }

    return result
  }
}
