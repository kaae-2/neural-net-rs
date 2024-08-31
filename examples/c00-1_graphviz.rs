use neural_net_rs::micrograd::{draw_dots, output_graph_file, Value};
use neural_net_rs::Result;

fn main() -> Result<()> {
    // Toy example making a graphviz
    let a = Value::from(2.0);
    let b = Value::from(-3.0);
    let c = Value::from(10.0);
    let e = &a * &b;
    let d = &e + &c;
    let f = Value::from(-2.0);
    let label = &d * &f;

    // Call the draw_dots function
    let dot = draw_dots(label).unwrap();

    println!("{}", &dot);

    let _ = output_graph_file(&dot, "./test.png".to_string());
    Ok(())
}
