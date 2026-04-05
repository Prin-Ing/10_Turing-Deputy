/**
 * Byte Pair Encoding(BPE) 기반 토크나이저입니다.
 *
 * 이 구현은 다음 구조를 사용합니다.
 * - 초기 vocabulary는 0~255 바이트 단위 토큰으로 시작합니다.
 * - 학습 과정에서 가장 자주 등장하는 인접 토큰 쌍을 찾아 새 토큰으로 병합합니다.
 * - `vocab`, `id2Token`, `merges`를 함께 유지해 토큰 id와 병합 규칙을 관리합니다.
 *
 * 현재 구현은 BPE 학습의 핵심 루프까지 포함하며, pair 집계, 최빈 pair 선택,
 * 새 토큰 등록, 전체 시퀀스 치환을 반복해 vocabulary를 확장합니다.
 */

import * as fs from 'fs'

/**
 * 파일로 저장되는 BPE 토크나이저 직렬화 형식입니다.
 *
 * 현재는 학습된 merge 규칙의 순서만 저장합니다. 기본 byte vocabulary와
 * 특수 토큰은 `load()` 시 코드로 재구성됩니다.
 */
type SerializedBPETokenizer = {
  merges: Array<[string, number]>;
};

/**
 * UTF-8 바이트 기반 Byte Pair Encoding 토크나이저입니다.
 *
 * 설계 특징:
 * - 기본 vocabulary는 단일 바이트 256개와 미리 정의된 특수 토큰으로 시작합니다.
 * - 학습 시 가장 자주 등장하는 인접 token pair를 반복적으로 병합합니다.
 * - 인코딩 시 학습된 merge 순서를 그대로 재적용해 가능한 긴 토큰을 만듭니다.
 * - 디코딩 시 일반 토큰은 바이트 시퀀스로, 특수 토큰은 문자열 세그먼트로 복원합니다.
 */
class BPETokenizer {
  /**
   * 예약된 특수 토큰 목록입니다.
   *
   * 순서대로 vocabulary 뒤쪽에 추가되며, 저장 파일에는 포함되지 않습니다.
   */
  private static readonly specialTokens = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|system|>",
    "<|human|>",
    "<|assistant|>",
  ];

  /**
   * 토큰 문자열을 토큰 id로 매핑합니다.
   *
   * 예: `"ab" -> 300`
   */
  vocab: Map<string, number>;

  /**
   * 토큰 id를 원래 토큰 문자열로 역매핑합니다.
   *
   * 예: `300 -> "ab"`
   */
  id2Token: Map<number, string>;

  /**
   * 병합 규칙을 저장합니다.
   *
   * key는 병합 전 토큰 쌍, value는 병합 후 새 토큰 id입니다.
   * 예: `"97,98" -> 300`
   */
  merges: Map<string, number>;

  /**
   * 문자열을 UTF-8 바이트 시퀀스로 변환하는 인코더입니다.
   */
  private encoder = new TextEncoder();

  /**
   * BOS(beginning of sequence) 특수 토큰 id입니다.
   */
  bosId!: number;

  /**
   * EOS(end of sequence) 특수 토큰 id입니다.
   */
  eosId!: number;

  /**
   * PAD(padding) 특수 토큰 id입니다.
   */
  padId!: number;

  /**
   * UTF-8 바이트 시퀀스를 문자열로 되돌리는 디코더입니다.
   */
  private decoder = new TextDecoder();

  /**
   * 특수 토큰 문자열인지 여부를 빠르게 확인하기 위한 집합입니다.
   */
  private specialTokenSet = new Set(BPETokenizer.specialTokens);

  /**
   * 바이트 단위 기본 vocabulary 상태로 토크나이저를 초기화합니다.
   *
   * `load()`에서 기존 상태를 지우고 저장된 병합 규칙을 다시 적용하기 전에 사용합니다.
   */
  private resetBaseVocab(): void {
    this.vocab = new Map();
    this.id2Token = new Map();
    this.merges = new Map();

    for (let i = 0; i < 256; i++) {
      const byteStr = String.fromCharCode(i);
      this.vocab.set(byteStr, i);
      this.id2Token.set(i, byteStr);
    }

    // 2. 특수 토큰 등록
    for (const token of BPETokenizer.specialTokens) {
      const id = this.vocab.size  // 256부터 시작
      this.vocab.set(token, id)
      this.id2Token.set(id, token)
    }

    // 3. 편하게 쓰려고
    this.bosId = this.vocab.get('<|bos|>')!
    this.eosId = this.vocab.get('<|eos|>')!
    this.padId = this.vocab.get('<|pad|>')!
  }

  /**
   * 바이트 단위 vocabulary로 초기화된 BPE 토크나이저를 생성합니다.
   *
   * 생성 직후에는 `0~255`의 모든 단일 바이트가 기본 토큰으로 등록됩니다.
   * 또한 특수 토큰과 그에 대응하는 편의 id 필드도 함께 초기화됩니다.
   */
  constructor() {
    this.vocab = new Map();
    this.id2Token = new Map();
    this.merges = new Map();
    this.resetBaseVocab();

  }

  /**
   * 문자열을 UTF-8 바이트 배열로 변환합니다.
   *
   * BPE 학습은 문자열 자체가 아니라 바이트 시퀀스를 기준으로 진행됩니다.
   *
   * @param text 바이트 단위로 분해할 입력 문자열
   * @returns UTF-8로 인코딩된 바이트 배열
   */
  private text2Bytes(text: string): Uint8Array {
    return this.encoder.encode(text);
  }

  /**
   * UTF-8 바이트 배열을 문자열로 복원합니다.
   *
   * 병합된 토큰 id를 다시 텍스트 형태로 해석할 때 사용할 수 있습니다.
   *
   * @param bytes 문자열로 복원할 바이트 배열
   * @returns 디코딩된 문자열
   */
  private bytes2Text(bytes: Uint8Array): string {
    return this.decoder.decode(bytes);
  }

  /**
   * 토큰 시퀀스들에서 인접한 토큰 쌍의 빈도를 계산합니다.
   *
   * 각 시퀀스를 왼쪽부터 순회하면서 `(token[i], token[i + 1])` 형태의 쌍을 세고,
   * `"a,b"` 문자열 키로 누적합니다. BPE 학습에서는 이 빈도표를 기반으로
   * 가장 자주 등장한 쌍을 다음 merge 대상으로 선택합니다.
   *
   * @param sequences 토큰 id 시퀀스 목록
   * @returns `"left,right" -> count` 형태의 pair frequency 맵
   *
   * @example
   * // [[1, 2, 3], [1, 2]] -> "1,2"는 2번 등장
   */
  private countPairs(sequences: number[][]): Map<string, number> {
    const counts = new Map<string, number>();

    for (const seq of sequences) {
      for (let i = 0; i < seq.length - 1; i++) {
        const key = `${seq[i]},${seq[i + 1]}`;
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
    }

    return counts;
  }

  /**
   * pair frequency 맵에서 가장 자주 등장한 토큰 쌍을 찾습니다.
   *
   * `countPairs()`가 만든 `"left,right" -> count` 맵을 순회하면서
   * 가장 큰 count를 가진 key를 반환합니다. 학습 루프에서는 이 값이
   * 다음 merge 대상 pair가 됩니다.
   *
   * 현재 구현은 동점일 때 먼저 순회된 pair를 유지합니다.
   * 빈 맵이 들어오면 병합할 후보가 없으므로 `null`을 반환합니다.
   *
   * @param counts 토큰 쌍 빈도 맵
   * @returns 가장 빈도가 높은 pair key, 없으면 `null`
   *
   * @example
   * // Map { "1,2" => 4, "2,3" => 2 } -> "1,2"
   */
  private mostFreqPair(counts: Map<string, number>): string | null {
    let bestKey = null;
    let bestCount = 0;

    for (const [key, count] of counts) {
      if (count > bestCount) {
        bestCount = count;
        bestKey = key;
      }
    }

    return bestKey;
  }

  /**
   * 모든 시퀀스에서 특정 인접 pair를 새 토큰 id로 치환합니다.
   *
   * `pairKey`는 `"a,b"` 형식의 문자열이며, 각 시퀀스를 왼쪽부터 훑으면서
   * 정확히 일치하는 인접 토큰 쌍 `(a, b)`를 찾으면 두 토큰 대신 `newId` 하나를 넣습니다.
   * 한 번 병합된 위치는 건너뛰므로 같은 토큰이 겹치는 중복 병합은 발생하지 않습니다.
   *
   * BPE 학습 루프에서는 가장 자주 등장한 pair를 선택한 뒤 이 메서드로 전체 말뭉치의
   * 토큰 시퀀스를 갱신합니다.
   *
   * @param sequences 병합 대상 토큰 시퀀스 목록
   * @param pairKey 병합할 토큰 쌍을 나타내는 `"left,right"` 문자열
   * @param newId 병합 결과를 대표할 새 토큰 id
   * @returns 해당 pair가 `newId`로 치환된 새 시퀀스 목록
   *
   * @example
   * // pairKey = "1,2", newId = 300
   * // [1, 2, 3, 1, 2] -> [300, 3, 300]
   */
  private mergePair(sequences: number[][], pairKey: string, newId: number): number[][] {
    const [a, b] = pairKey.split(',').map(Number);

    return sequences.map((seq) => {
      const result: number[] = [];
      let i = 0;
      while (i < seq.length) {
        if (seq[i] === a && seq[i + 1] === b) {
          result.push(newId);
          i += 2;
        } else {
          result.push(seq[i]);
          i += 1;
        }
      }
      return result;
    });
  }

  /**
   * 말뭉치를 기반으로 BPE vocabulary를 학습합니다.
   *
   * 입력 텍스트를 UTF-8 바이트 시퀀스로 변환한 뒤, 목표 vocabulary 크기에
   * 도달할 때까지 가장 자주 등장한 인접 pair를 반복적으로 병합합니다.
   *
   * 내부 동작:
   * 1. 인접 토큰 쌍의 빈도 계산
   * 2. 가장 자주 등장한 pair 선택
   * 3. 새 토큰 id 할당 및 vocabulary/merge 규칙 갱신
   * 4. 전체 시퀀스에서 해당 pair를 새 토큰으로 치환
   *
   * @param corpus 학습에 사용할 원시 텍스트 말뭉치
   * @param vocabSize 목표 vocabulary 크기
   *
   * @example
   * const tokenizer = new BPETokenizer();
   * tokenizer.train(["hello hello", "hello world"], 300);
   */
  train(corpus: string[], vocabSize: number): void {
    if (vocabSize < this.vocab.size) {
      throw new Error(`vocabSize는 최소 ${this.vocab.size} 이상이어야 합니다.`);
    }

    let sequences: number[][] = corpus.map((text) =>
      Array.from(this.text2Bytes(text))
    );
    while (this.vocab.size < vocabSize) {
      const counts = this.countPairs(sequences);
      const bestKey = this.mostFreqPair(counts);

      if (bestKey === null) break;

      const newId = this.vocab.size;
      const [a, b] = bestKey.split(',').map(Number);
      const newToken = this.id2Token.get(a)! + this.id2Token.get(b)!;
      this.vocab.set(newToken, newId);
      this.id2Token.set(newId, newToken);
      this.merges.set(bestKey, newId);

      sequences = this.mergePair(sequences, bestKey, newId);
    }
  }

  /**
   * 입력 문자열을 학습된 BPE 규칙에 따라 토큰 id 시퀀스로 인코딩합니다.
   *
   * 먼저 문자열을 UTF-8 바이트 id 시퀀스로 바꾼 뒤, 학습된 `merges` 규칙을
   * 등록 순서대로 적용해 더 긴 토큰으로 병합합니다.
   * 특수 토큰 문자열은 일반 텍스트와 분리해 병합 없이 그대로 보존합니다.
   *
   * @param text 인코딩할 원시 문자열
   * @returns 토큰 id 배열
   *
   * @example
   * const ids = tokenizer.encode("hello");
   */
  encode(text: string): number[] {
    const pattern = new RegExp(`(${BPETokenizer.specialTokens.map((token) =>
      token.replace(/[|]/g, "\\|")
    ).join("|")})`, "g");

    const parts = text.split(pattern).filter((part) => part.length > 0);
    const result: number[] = [];

    for (const part of parts) {
      if (this.specialTokenSet.has(part)) {
        const specialId = this.vocab.get(part);
        if (specialId === undefined) {
          throw new Error(`등록되지 않은 특수 토큰입니다: ${part}`);
        }
        result.push(specialId);
        continue;
      }

      let ids = Array.from(this.text2Bytes(part));
      for (const [pairKey, newId] of this.merges) {
        ids = this.mergePair([ids], pairKey, newId)[0];
      }
      result.push(...ids);
    }

    return result;
  }

  /**
   * 토큰 id 시퀀스를 원래 문자열로 복원합니다.
   *
   * 각 id를 대응하는 토큰 문자열로 바꿔 이어 붙인 뒤, 이를 바이트 시퀀스로 해석해
   * UTF-8 문자열로 디코딩합니다. vocabulary에 없는 id가 들어오면 복원할 수 없으므로
   * 예외를 던집니다.
   *
   * @param ids 복원할 토큰 id 배열
   * @returns 디코딩된 원시 문자열
   * @throws {Error} vocabulary에 없는 token id가 포함되어 있으면 예외를 던집니다.
   *
   * @example
   * const text = tokenizer.decode([104, 101, 108, 108, 111]);
   */
  decode(ids: number[]): string {
    const segments: string[] = [];
    let byteBuffer: number[] = [];

    const flushBytes = () => {
      if (byteBuffer.length === 0) {
        return;
      }
      segments.push(this.bytes2Text(Uint8Array.from(byteBuffer)));
      byteBuffer = [];
    };

    for (const id of ids) {
      const token = this.id2Token.get(id);
      if (token === undefined) {
        throw new Error(`알 수 없는 token id입니다: ${id}`);
      }

      if (this.specialTokenSet.has(token)) {
        flushBytes();
        segments.push(token);
        continue;
      }

      for (const char of token) {
        byteBuffer.push(char.charCodeAt(0));
      }
    }

    flushBytes();
    return segments.join("");
  }

  /**
   * 현재 토크나이저 상태를 JSON 파일로 저장합니다.
   *
   * 현재 구현은 병합 규칙만 저장하며, base byte vocabulary는 `load()` 시 자동으로
   * 다시 구성됩니다. 파일 포맷은 사람이 읽을 수 있도록 들여쓰기된 JSON이며,
   * merge의 순서를 유지해야 하므로 배열 형태로 기록합니다.
   *
   * @param path 저장할 파일 경로
   */
  save(path: string): void {
    const data: SerializedBPETokenizer = {
      merges: Array.from(this.merges.entries()),
    };

    fs.writeFileSync(path, JSON.stringify(data, null, 2), "utf-8");
  }

  /**
   * 저장된 JSON 파일에서 토크나이저 상태를 복원합니다.
   *
   * 먼저 byte 단위 기본 vocabulary를 다시 만든 뒤, 저장된 `merges`를 순서대로
   * 재적용해 `vocab`, `id2Token`, `merges`를 일관된 상태로 재구성합니다.
   * 따라서 저장 시점과 동일한 merge 순서가 유지되어야 같은 인코딩 결과를 얻습니다.
   *
   * @param path 불러올 tokenizer JSON 파일 경로
   * @throws {Error} 파일 형식이 올바르지 않거나, 병합 규칙이 잘못되었으면 예외를 던집니다.
   */
  load(path: string): void {
    if (!fs.existsSync(path)) {
      throw new Error(`tokenizer 파일을 찾을 수 없습니다: ${path}`);
    }

    const raw = fs.readFileSync(path, "utf-8");
    const data = JSON.parse(raw) as SerializedBPETokenizer;

    if (!Array.isArray(data.merges)) {
      throw new Error("잘못된 tokenizer 파일 형식입니다.");
    }

    this.resetBaseVocab();

    for (const [pairKey, newId] of data.merges) {
      if (typeof pairKey !== "string" || typeof newId !== "number") {
        throw new Error("잘못된 merge 항목 형식입니다.");
      }

      const [a, b] = pairKey.split(",").map(Number);
      const leftToken = this.id2Token.get(a);
      const rightToken = this.id2Token.get(b);

      if (leftToken === undefined || rightToken === undefined) {
        throw new Error(`잘못된 merge 규칙입니다: ${pairKey}`);
      }

      const newToken = leftToken + rightToken;
      this.vocab.set(newToken, newId);
      this.id2Token.set(newId, newToken);
      this.merges.set(pairKey, newId);
    }
  }
}
