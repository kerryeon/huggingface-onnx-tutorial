use std::{fs, io, path::Path, time::Duration};

use anyhow::Result;
use onnxruntime::{environment::Environment, ndarray};
use rust_tokenizers::{
    adapters::Example,
    tokenizer::{RobertaTokenizer, Tokenizer, TruncationStrategy},
    vocab::{BpePairVocab, RobertaVocab, Vocab},
};

fn argmax<S>(mat: &ndarray::ArrayBase<S, ndarray::Ix2>) -> ndarray::Array1<usize>
where
    S: ndarray::Data,
    S::Elem: PartialOrd,
{
    mat.rows()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        })
        .collect()
}

/// 구문에서 답이 존재하는 구역을 잘라옵니다.
fn find_answer<S>(
    mat: &ndarray::ArrayBase<S, ndarray::Ix2>,
    start_logits: &ndarray::CowArray<f32, ndarray::Ix2>,
    end_logits: &ndarray::CowArray<f32, ndarray::Ix2>,
) -> Vec<ndarray::Array1<S::Elem>>
where
    S: ndarray::Data,
    S::Elem: Copy + Clone + PartialOrd,
{
    let start_logits = argmax(start_logits);
    let end_logits = argmax(end_logits);
    mat.rows()
        .into_iter()
        .zip(start_logits)
        .zip(end_logits)
        .map(|((row, start), end)| {
            row.into_iter()
                .skip(start)
                .take(end - start + 1)
                .cloned()
                .collect()
        })
        .collect()
}

/// Source: https://github.com/nbigaouette/onnxruntime-rs/blob/88c6bab938f278c92b90ec4b43c40f47debb9fa6/onnxruntime/tests/integration_tests.rs#L44
fn download<'a>(filename: &'a str, url: &str) -> Result<&'a str> {
    // Download the ImageNet class labels, matching SqueezeNet's classes.
    let labels_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(filename);
    if !labels_path.exists() {
        println!("Downloading {:?} to {:?}...", url, labels_path);
        let resp = ureq::get(url)
            .timeout(Duration::from_secs(180)) // 3 minutes
            .call()?;

        assert!(resp.has("Content-Length"));
        let len = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap();
        println!("Downloading {} bytes...", len);

        let mut reader = resp.into_reader();

        let f = fs::File::create(&labels_path).unwrap();
        let mut writer = io::BufWriter::new(f);

        let bytes_io_count = io::copy(&mut reader, &mut writer).unwrap();

        assert_eq!(bytes_io_count, len as u64);
    }

    Ok(filename)
}

fn main() -> Result<()> {
    // 모델의 Vocabulary 정보를 가져옵니다.
    let vocab_path = download(
        "vocab.json",
        "https://huggingface.co/deepset/roberta-base-squad2/raw/main/vocab.json",
    )?;
    let vocab = RobertaVocab::from_file(vocab_path)?;

    // 모델의 업데이트 정보를 가져옵니다.
    let merges_path = download(
        "merges.txt",
        "https://huggingface.co/deepset/roberta-base-squad2/raw/main/merges.txt",
    )?;
    let merges = BpePairVocab::from_file(merges_path)?;

    // 단어 토큰화 모듈을 불러옵니다.
    let tokenizer = RobertaTokenizer::from_existing_vocab_and_merges(vocab, merges, false, false);

    // ONNX 세션을 생성합니다. 세션은 모델 정보를 가지고 있습니다.
    let env = Environment::builder().build()?;
    let session = env
        .new_session_builder()?
        .with_model_from_file("../onnx/model.onnx")?;

    // 세션의 입출력 정보를 확인할 수도 있습니다.
    dbg!(session.inputs.iter().map(|e| &e.name).collect::<Vec<_>>());
    dbg!(session.outputs.iter().map(|e| &e.name).collect::<Vec<_>>());

    // 입력값을 토큰화합니다. Q&A의 경우, 아래 예시와 같이 병렬로 입력합니다.
    let test_sentence = Example::new_from_strings(
        // Question
        "Where do I live?",
        // Context
        "My name is Sarah and I live in London",
    );
    let inputs = tokenizer.encode(
        &test_sentence.sentence_1,
        Some(&test_sentence.sentence_2),
        128,
        &TruncationStrategy::LongestFirst,
        0,
    );

    // 토큰화된 입력값 상태를 확인할 수 있습니다.
    println!("{:?}", &inputs);

    // 세션에 입력값을 넣어 ONNX 연산을 수행합니다. 출력값은 제공한 순서대로 반환됩니다.
    let input_length = inputs.token_ids.len();
    let input_ids = ndarray::Array::from_vec(inputs.token_ids).into_shape((1, input_length))?;
    let attention_mask = {
        let mut buf = input_ids.clone();
        buf.fill(1);
        buf
    };
    let outputs = session.run::<_, f32>(&[&input_ids, &attention_mask])?;

    // 결과값을 분석하여 정답 문자열을 획득합니다.
    let start_logits = outputs[0].to_shape((1, input_length))?;
    let end_logits = outputs[1].to_shape((1, input_length))?;
    let answer = find_answer(&input_ids, &start_logits, &end_logits);
    dbg!(answer
        .iter()
        .map(|row| tokenizer
            .decode(row.as_slice().unwrap(), true, true)
            .trim()
            .to_string())
        .collect::<Vec<_>>());

    Ok(())
}
