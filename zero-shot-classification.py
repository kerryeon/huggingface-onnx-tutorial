import numpy as np
from onnxruntime import InferenceSession
from transformers import AutoTokenizer
from transformers.onnx.features import FeaturesManager


if __name__ == '__main__':
    # 해당 모델로 가능한 기능의 목록을 볼 수 있습니다. (sequence-classification, ...)
    features = list(
        FeaturesManager.get_supported_features_for_model_type(
            "roberta",
        ).keys(),
    )
    print(features)

    # 단어 토큰화 모듈을 불러옵니다.
    tokenizer = AutoTokenizer.from_pretrained(
        "cross-encoder/nli-distilroberta-base",
    )

    # ONNX 세션을 생성합니다. 세션은 모델 정보를 가지고 있습니다.
    session = InferenceSession("onnx/model.onnx")

    # 세션의 입출력 정보를 확인할 수도 있습니다.
    print([e.name for e in session.get_inputs()])
    print([e.name for e in session.get_outputs()])

    # 입력값을 토큰화합니다. Q&A의 경우, 아래 예시와 같이 병렬로 입력합니다.
    inputs = tokenizer(
        # Context
        "Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app.",
        # Question
        "This example is mobile.",
        # ONNX에 넣기 위해 numpy 형식으로 결과를 반환합니다.
        return_tensors="np",
    )

    # 토큰화된 입력값 상태를 확인할 수 있습니다.
    print([tokenizer.decode(e) for e in inputs["input_ids"][0]])

    # 연산에 필요없는 입력값은 제거합니다.
    inputs = {
        key: value
        for key, value in inputs.items()
        if key in [str(e.name) for e in session.get_inputs()]
    }

    # 세션에 입력값을 넣어 ONNX 연산을 수행합니다. 출력값은 제공한 순서대로 반환됩니다.
    logits, = session.run(
        input_feed=inputs,
        output_names=["logits"],
    )

    # 결과값을 분석하여 정답 클래스를 획득합니다.
    # labels은 모델마다 구성과 인덱스가 다릅니다. `neutral`이 없을 수도 있습니다.
    contradiction, entailment, neutral = 0, 1, 2
    results = np.exp(logits[:, [contradiction, entailment]])
    prob = (results / np.sum(results))[0, 1]
    print(f'Probability: {prob * 100:.4f}%')
