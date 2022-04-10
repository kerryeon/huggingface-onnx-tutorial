# Huggingface + ONNX + QuestionAnswering = OK

Huggingface 내 서비스되는 모델을 ONNX로 변환 후 테스트하는 예제입니다.

## Tutorial

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

## References

* https://huggingface.co/docs/transformers/serialization#selecting-features-for-different-model-topologies

## LICENSE

Unlicensed. Feel free to use.
