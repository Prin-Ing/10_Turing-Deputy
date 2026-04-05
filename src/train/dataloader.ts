import * as fs from 'fs'
import { BPETokenizer } from './tokenizer';

/**
 * 학습용 텍스트 파일을 토큰 id 배열로 읽고,
 * 랜덤한 위치에서 next-token prediction 배치를 생성하는 데이터 로더입니다.
 */
class DataLoader {

  tokenizer: BPETokenizer
  contextLength: number
  batchSize: number
  data: number[]

  /**
   * 데이터 로더에 사용할 토크나이저와 배치 설정을 초기화합니다.
   *
   * @param tokenizer 텍스트를 토큰 id로 변환할 토크나이저
   * @param contextLength 입력 시퀀스 길이
   * @param batchSize 한 번에 생성할 샘플 수
   */
  constructor(tokenizer: BPETokenizer, contextLength: number, batchSize: number) {
    this.tokenizer = tokenizer
    this.contextLength = contextLength
    this.batchSize = batchSize
    this.data = []
  }

  /**
   * 지정한 텍스트 파일을 읽어 전체 데이터를 토큰 id 배열로 저장합니다.
   *
   * @param filePath `data` 디렉터리 기준의 입력 파일 경로
   */
  load(filePath: string): void {
    const text = fs.readFileSync(`../../data/${filePath}`, "utf-8")
    this.data = this.tokenizer.encode(text)
  }


  /**
   * 랜덤한 시작 위치에서 입력 시퀀스와 다음 토큰 정답 시퀀스를 배치로 생성합니다.
   *
   * `x`는 길이 `contextLength`의 입력 토큰들이고,
   * `y`는 각 위치에서 한 칸 오른쪽으로 이동한 next-token target입니다.
   *
   * @returns `x`, `y` 배치를 담은 객체
   */
  nextBatch(): { x: number[][], y: number[][] } {
    const x: number[][] = []
    const y: number[][] = []

    for (let b = 0; b < this.batchSize; b++) {
      const pos = Math.floor(Math.random() * (this.data.length - this.contextLength - 1))

      x.push(this.data.slice(pos, pos + this.contextLength))
      y.push(this.data.slice(pos + 1, pos + this.contextLength + 1))
    }

    return { x, y }
  }
}
