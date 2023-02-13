import json

from transformers import AutoTokenizer, DebertaV2TokenizerFast


if __name__ == '__main__':
    # 단어 토큰화 모듈을 불러옵니다.
    tokenizer = AutoTokenizer.from_pretrained(
        "timpal0l/mdeberta-v3-base-squad2",
    )

    # 데이터를 JSON 형태로 저장합니다.
    tokenizer_dir = './tokenizer'
    tokenizer.save_pretrained(tokenizer_dir)
    raw = json.load(open(f'{tokenizer_dir}/tokenizer.json', 'r'))

    # SentencePiece 모델로 변환합니다.
    import tokenizer.sentencepiece_pb2 as model
    m = model.ModelProto()
    for piece, score in raw['model']['vocab']:
        p = model.ModelProto.SentencePiece()
        p.piece, p.score = piece, score
        m.pieces.append(p)

    # 변환한 모델을 저장합니다.
    with open(f'{tokenizer_dir}/spm.model', 'wb') as f:
        f.write(m.SerializeToString())
