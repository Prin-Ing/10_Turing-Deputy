/**
 * 토큰 id를 연속된 임베딩 벡터로 변환하는 임베딩 레이어입니다.
 *
 * 내부적으로 `[vocabSize, dModel]` shape의 weight 행렬을 가지며,
 * 각 토큰 id는 이 행렬의 한 행(row)을 가리킵니다.
 */
class Embedding {
  /**
   * 토큰 임베딩 weight 행렬입니다.
   *
   * shape: `[vocabSize, dModel]`
   */
  weight: Tensor;

  /**
   * 무작위 초기화된 토큰 임베딩 테이블을 생성합니다.
   *
   * @param vocabSize vocabulary에 포함된 전체 토큰 수
   * @param dModel 각 토큰을 표현할 임베딩 차원 수
   */
  constructor(vocabSize: number, dModel: number) {
    this.weight = Tensor.randn([vocabSize, dModel]);
  }

  /**
   * 토큰 id 시퀀스를 대응하는 임베딩 행렬로 변환합니다.
   *
   * 각 id에 대해 weight 행렬에서 해당 row를 복사해 꺼내고,
   * 최종적으로 `[seqLen, dModel]` shape의 텐서를 반환합니다.
   *
   * @param ids 임베딩할 토큰 id 배열
   * @returns 각 토큰의 임베딩 벡터를 담은 텐서
   *
   * @example
   * const emb = new Embedding(1000, 128);
   * const x = emb.forward([5, 10, 42]); // shape: [3, 128]
   */
  forward(ids: number[]): Tensor {
    const rows = ids.map((id) => {
      const start = id * this.weight.strides[0];
      return this.weight.data.slice(start, start + this.weight.shape[1]);
    });

    const seqLen = ids.length;
    const dModel = this.weight.shape[1];
    const data = new Float32Array(seqLen * dModel);

    // 각 토큰 임베딩 row를 결과 버퍼에 차례대로 복사합니다.
    rows.forEach((row, i) => data.set(row, i * dModel));

    return new Tensor(data, [seqLen, dModel]);
  }
}

/**
 * Transformer에서 사용하는 sinusoidal positional encoding을 생성합니다.
 *
 * 위치 정보는 학습 가능한 파라미터 대신 사인/코사인 함수로 계산되며,
 * 토큰 임베딩에 더해져 순서 정보를 제공합니다.
 */
class PositionalEncoding {
  /**
   * 주어진 시퀀스 길이와 모델 차원에 대한 positional encoding 텐서를 생성합니다.
   *
   * 짝수 인덱스에는 `sin`, 홀수 인덱스에는 `cos`를 사용합니다.
   *
   * @param seqLen 시퀀스 길이
   * @param dModel 모델 차원 수
   * @returns shape가 `[seqLen, dModel]`인 positional encoding 텐서
   */
  forward(seqLen: number, dModel: number): Tensor {
    const result = Tensor.zeros([seqLen, dModel]);

    for (let pos = 0; pos < seqLen; pos++) {
      for (let i = 0; i < dModel; i += 2) {
        const angle = pos / Math.pow(10000, i / dModel);
        result.set([pos, i], Math.sin(angle));

        if (i + 1 < dModel) {
          result.set([pos, i + 1], Math.cos(angle));
        }
      }
    }

    return result;
  }
}

/**
 * 토큰 임베딩과 positional encoding을 합쳐 최종 입력 임베딩을 만드는 레이어입니다.
 */
class EmbeddingLayer {
  /**
   * 토큰 id를 임베딩 벡터로 바꾸는 레이어입니다.
   */
  tokenEmbedding: Embedding;

  /**
   * 위치 정보를 생성하는 positional encoding 레이어입니다.
   */
  positionalEncoding: PositionalEncoding;

  /**
   * 토큰 임베딩 레이어와 positional encoding 레이어를 함께 초기화합니다.
   *
   * @param vocabSize vocabulary 크기
   * @param dModel 모델 차원 수
   */
  constructor(vocabSize: number, dModel: number) {
    this.tokenEmbedding = new Embedding(vocabSize, dModel);
    this.positionalEncoding = new PositionalEncoding();
  }

  /**
   * 입력 토큰 id 시퀀스에 대해 최종 임베딩 표현을 생성합니다.
   *
   * 먼저 토큰 임베딩을 구한 뒤, 같은 shape의 positional encoding을 생성하고
   * 둘을 원소별로 더해 반환합니다.
   *
   * @param ids 입력 토큰 id 배열
   * @returns token embedding과 positional encoding이 합쳐진 텐서
   *
   * @example
   * const layer = new EmbeddingLayer(5000, 256);
   * const x = layer.forward([1, 2, 3, 4]); // shape: [4, 256]
   */
  forward(ids: number[]): Tensor {
    const tokenEmb = this.tokenEmbedding.forward(ids);
    const posEnc = this.positionalEncoding.forward(ids.length, tokenEmb.shape[1]);
    return tokenEmb.add(posEnc);
  }
}
