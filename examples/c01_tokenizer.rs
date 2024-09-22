use neural_net::{
    transformer::{BytePairEncoder, Encoding},
    Result,
};

fn main() -> Result<()> {
    let input = "aabbabax";
    println!("input string: {input}");
    let mut encoder = BytePairEncoder::default();
    encoder.train(input.to_string(), Encoding::Utf8, None)?;

    let encoded = encoder.encode(input, Encoding::Utf8)?;

    println!("{:?}", encoder);
    println!("last encoded string: {:?}", encoded);

    let decoded = encoder.decode(encoded)?;
    println!("decoded string: {:?}", decoded);

    Ok(())
}
