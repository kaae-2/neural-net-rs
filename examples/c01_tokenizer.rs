use neural_net_rs::{transformer::BytePairEncoder, Result};

fn main() -> Result<()> {
    let input = "aabbabax";
    println!("input string: {input}");
    let mut encoder = BytePairEncoder::default();
    let encoded = encoder.from_utf8(input)?;
    println!("0th encoded string {:?}", encoded);
    let output = encoder.encode(encoded)?;

    println!("{:?}", encoder);
    println!("1st encoded string: {:?}", output);

    Ok(())
}
