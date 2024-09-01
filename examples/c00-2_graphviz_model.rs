use neural_net::{
    micrograd::{visualize_network, Value, MLP},
    Result,
};

fn main() -> Result<()> {
    let model = MLP::new(vec![2, 8, 8, 2]);
    let input = model.forward(vec![Value::from(0.0), Value::from(0.0)]);
    let _dot = visualize_network(input, "./test-model.png".to_string())?;
    // println!("{dot}");

    Ok(())
}
