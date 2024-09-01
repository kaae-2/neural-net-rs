use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use neural_net::{
    micrograd::{visualize_network, Value, MLP},
    Result,
};

pub struct DataPoint {
    x: f64,
    y: f64,
    label: f64,
}

fn read_csv_file(filename: &str) -> Result<Vec<DataPoint>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let data_points: Vec<DataPoint> = reader
        .lines()
        .enumerate()
        .filter(|&(index, _)| index != 0)
        .map(|(_, line)| {
            let value = line.expect("line cannot be read");
            let fields: Vec<&str> = value.split(',').collect();
            let data_point = DataPoint {
                x: fields[0].parse::<f64>().expect("field 0 can be read"),
                y: fields[1].parse::<f64>().expect("field 1 can be read"),
                label: fields[2].parse::<f64>().expect("field 2 can be read"),
            };
            data_point
        })
        .collect();

    Ok(data_points)
}

fn load_moon_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let data_points = read_csv_file("make_moons.csv").expect("moons file should exist");

    let (data, labels) = data_points.iter().fold(
        (Vec::new(), Vec::new()),
        |mut acc: (Vec<Vec<f64>>, Vec<f64>), data_point| {
            acc.0.push(vec![data_point.x, data_point.y]);
            acc.1.push(data_point.label);
            acc
        },
    );
    (data, labels)
}

fn loss(model: &MLP, data: &[Vec<f64>], labels: &[f64], alpha: f64) -> (Value, f64) {
    let inputs: Vec<Vec<Value>> = data
        .iter()
        .map(|row| vec![Value::from(row[0]), Value::from(row[1])])
        .collect();

    let scores: Vec<Value> = inputs
        .iter()
        .map(|row| model.forward(row.clone())[0].clone())
        .collect();
    let losses: Vec<Value> = labels
        .iter()
        .zip(&scores)
        .map(|(label, score)| {
            let label_value = Value::from(*label);
            (&Value::from(1.0) - &(&label_value * score)).relu()
        })
        .collect();
    let n = Value::from((&losses).len() as f64);
    let data_losses = &losses.into_iter().sum::<Value>() / &n;

    let reg_loss = &Value::from(alpha)
        * &model
            .parameters()
            .iter()
            .map(|p| p * p)
            .into_iter()
            .sum::<Value>();
    let total_loss = &data_losses + &reg_loss;
    let accuracies: Vec<bool> = labels
        .iter()
        .zip(scores.iter())
        .map(|(label, score)| (*label > 0.0) == (score.borrow().data > 0.0))
        .collect();

    let accuracy = accuracies.iter().filter(|&a| *a).count() as f64 / n.borrow().data;
    (total_loss, accuracy)
}

fn plot_ascii(model: &MLP, bound: isize) {
    let mut grid: Vec<Vec<String>> = Vec::new();
    for y in -bound..bound {
        let mut row: Vec<String> = Vec::new();
        for x in -bound..bound {
            let k = &model.forward(vec![
                Value::from(x as f64 / bound as f64 * 2.0),
                Value::from(-y as f64 / bound as f64 * 2.0),
            ])[0];
            row.push(if k.borrow().data > 0.0 {
                String::from("+")
            } else {
                String::from("0")
            });
        }
        grid.push(row.clone())
    }
    for row in grid {
        // println!("{:?}", row)
        for val in row {
            print!("{} ", val)
        }
        println!();
    }
}

fn main() -> Result<()> {
    let model = MLP::new(vec![2, 16, 16, 1]);
    let (data, labels) = load_moon_data();
    let alpha = 0.001;
    for k in 0..100 {
        let (total_loss, accuracy) = loss(&model, &data, &labels, alpha);
        model.zero_grad();
        total_loss.backward();
        let learning_rate = 1.0 - 0.9 * (k as f64) / 100.0;
        for p in &model.parameters() {
            let delta = learning_rate * p.borrow().grad;
            p.borrow_mut().data -= delta;
        }
        println!(
            "step {k} loss {:.3}, accuracy {:.2}%, learning rate: {:.3}",
            total_loss.borrow().data,
            accuracy * 100.0,
            learning_rate
        )
    }
    plot_ascii(&model, 20);
    let input = vec![Value::from(0.0), Value::from(0.0)];
    let _ = visualize_network(model.forward(input), "./moon_graph.png".to_string());
    Ok(())
}
