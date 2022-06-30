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
        "Helsinki-NLP/opus-mt-ko-en",
    )

    # ONNX 세션을 생성합니다. 세션은 모델 정보를 가지고 있습니다.
    session = InferenceSession("onnx/model.onnx")

    # 세션의 입출력 정보를 확인할 수도 있습니다.
    print([e.name for e in session.get_inputs()])
    print([e.name for e in session.get_outputs()])

    # 입력값을 토큰화합니다.
    inputs = tokenizer(
        # Source
        "아니 이게 될 리가 없잖아?",
        # ONNX에 넣기 위해 numpy 형식으로 결과를 반환합니다.
        return_tensors="np",
    )

    # 토큰화된 입력값 상태를 확인할 수 있습니다.
    print([tokenizer.decode(e) for e in inputs["input_ids"][0]])

    # EOS 토큰값을 가져옵니다.
    token_eos = inputs['input_ids'][0][-1]

    # 재귀 - 한 토큰씩 번역값을 생성합니다.
    past = []  # 이전 출력값
    while not past or past[-1] != token_eos:
        decoder_inputs = tokenizer(
            # Source
            "<pad> " * (len(past) + 1),
            # ONNX에 넣기 위해 numpy 형식으로 결과를 반환합니다.
            return_tensors="np",
            # EOS 등 필요없는 토큰은 버립니다.
            add_special_tokens=False,
        )

        # 토큰 결과물을 딕셔너리로 변환합니다.
        inputs = dict(inputs)
        for key, value in decoder_inputs.items():
            inputs[f'decoder_{key}'] = value

        # 이전 토큰값을 입력값에 반영합니다.
        # * 첫 번째 토큰은 <PAD> 이어야 합니다.
        for idx, token in enumerate(past, start=1):
            inputs["decoder_input_ids"][0][idx] = token

        # 토큰화된 입력값 상태를 확인할 수 있습니다.
        print([tokenizer.decode(e) for e in inputs["decoder_input_ids"][0]])

        # 세션에 입력값을 넣어 ONNX 연산을 수행합니다. 출력값은 제공한 순서대로 반환됩니다.
        logits, = session.run(
            input_feed=dict(inputs),
            output_names=["logits"],
        )

        # 새로 추론한 토큰값을 추가합니다.
        past.append(logits[0].argmax(-1)[-1])

    # 토큰화된 출력값 상태를 확인할 수 있습니다.
    print([tokenizer.decode(e) for e in past])
