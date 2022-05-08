from onnxruntime import InferenceSession
from transformers import AutoTokenizer
from transformers.onnx.features import FeaturesManager


if __name__ == '__main__':
    # 해당 모델로 가능한 기능의 목록을 볼 수 있습니다. (question-answering, ...)
    features = list(
        FeaturesManager.get_supported_features_for_model_type(
            "roberta",
        ).keys(),
    )
    print(features)

    # 단어 토큰화 모듈을 불러옵니다.
    tokenizer = AutoTokenizer.from_pretrained(
        "deepset/roberta-base-squad2",
    )

    # ONNX 세션을 생성합니다. 세션은 모델 정보를 가지고 있습니다.
    session = InferenceSession("onnx/model.onnx")

    # 세션의 입출력 정보를 확인할 수도 있습니다.
    print([e.name for e in session.get_inputs()])
    print([e.name for e in session.get_outputs()])

    # 입력값을 토큰화합니다. Q&A의 경우, 아래 예시와 같이 병렬로 입력합니다.
    inputs = tokenizer(
        # Question
        "Where do I live?",
        # Context
        "My name is Sarah and I live in London",
        # ONNX에 넣기 위해 numpy 형식으로 결과를 반환합니다.
        return_tensors="np",
    )

    # 토큰화된 입력값 상태를 확인할 수 있습니다.
    print([tokenizer.decode(e) for e in inputs["input_ids"][0]])

    # 세션에 입력값을 넣어 ONNX 연산을 수행합니다. 출력값은 제공한 순서대로 반환됩니다.
    start_logits, end_logits = session.run(
        input_feed=dict(inputs),
        output_names=["start_logits", "end_logits"],
    )

    # 결과값을 분석하여 정답 문자열을 획득합니다.
    answer = ' '.join(
        tokenizer.decode(e).strip()
        for e
        in inputs["input_ids"][0][
            # 구문에서 답이 존재하는 구역을 잘라옵니다.
            start_logits[0].argmax():end_logits[0].argmax()+1
        ]
    )
    print(answer)
