use neural_net_rs::{
    transformer::{BytePairEncoder, Encoding},
    Result,
};

fn main() -> Result<()> {
    let input = "aabbabax";
    println!("input string: {input}");
    let mut encoder = BytePairEncoder::default();
    let encoded = encoder.encode(input, Encoding::Utf8)?;

    println!("{:?}", encoder);
    println!("last encoded string: {:?}", encoded);

    let decoded = encoder.decode(encoded)?;
    println!("decoded string: {:?}", decoded);
    let remade_string = encoder.to_utf8(decoded)?;
    println!("remade string: {:?}", remade_string);

    Ok(())
}
