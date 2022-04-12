# Huggingface + ONNX + QuestionAnswering = OK

Huggingface 내 서비스되는 모델을 ONNX로 변환 후 테스트하는 예제입니다.

## Tutorial

### 1. Convert a model with Python

```bash
# 1. 필수 요소 설치: pytorch, transformers, onnxruntime
# - 원하시는 방식으로 설치를 진행해주세요.

# 2. Huggingface 모델 가져오기
python -m transformers.onnx \
    --model=deepset/roberta-base-squad2 \  # 사용하고자 하는 모델명
    --feature question-answering \  # 모델의 기능
    onnx  # 저장할 디렉토리

# 3. 테스트하기
python main.py
```

### 2. Run the model with Rust

토큰 분석을 위한 Vocabulary, 패치 등은 자동으로 다운로드됩니다.
(인터넷에 연결되어있어야 함)

실제로 서비스할 때는, 프로그램 실행 전에 파일을 미리 준비해두는 것이 보안상 좋습니다.

```bash
# 1. 필수 요소 설치: rust (cargo)
# - 원하시는 방식으로 설치를 진행해주세요.

# 2. 테스트하기
cd rust  # rust 코드베이스가 있는 디렉토리로 이동
cargo run  # Rust 패키지 빌드 및 설치
```

## References

* https://huggingface.co/docs/transformers/serialization#selecting-features-for-different-model-topologies
* https://github.com/nbigaouette/onnxruntime-rs

## LICENSE

* Code that refers to a particular reference is governed by the license of that reference.
    - Please check: https://github.com/nbigaouette/onnxruntime-rs#license
* Other than that => Unlicensed. Feel free to use.
