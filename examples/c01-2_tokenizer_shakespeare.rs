use neural_net::{
    transformer::{BytePairEncoder, Encoding},
    Result,
};
use reqwest::blocking::get;

fn get_shakespeare() -> Result<String> {
    let url =
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
    let input = get(url)
        .map_err(|err| -> Box<dyn std::error::Error> { err.into() })?
        .text()?;
    Ok(input)
}

fn main() -> Result<()> {
    let data = get_shakespeare()?;
    let input: String = data.chars().take(10_000).collect();
    // println!("input string: {:?}", input);
    let mut encoder = BytePairEncoder::default();
    encoder.train(input.to_string(), Encoding::Utf8, Some(1000))?;

    let encoded = encoder.encode(&input, Encoding::Utf8)?;

    println!("{:?}", encoder);
    // println!("last encoded string: {:?}", encoded);

    let decoded = encoder.decode(encoded)?;
    // println!("decoded string: {:?}", decoded);

    Ok(())
}
