use std::{
    borrow::Borrow,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
};

use neural_net_rs::{engine::Value, neural_net::MLP};

pub struct DataPoint {
    x: f64,
    y: f64,
    label: f64,
}

pub fn read_csv_file(filename: &str) -> Result<Vec<DataPoint>, Box<dyn Error>> {
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

pub fn load_moon_data() -> (Vec<Vec<f64>>, Vec<f64>) {
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

pub fn loss(model: &MLP, data: &[Vec<f64>], labels: &[f64], alpha: f64) -> (Value, f64) {
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

fn main() {
    let model = MLP::new(vec![2, 16, 16, 1]);
    let (data, labels) = load_moon_data();
    let alpha = 0.001;
    for k in 0..100 {
        let (total_loss, accuracy) = loss(&model, &data, &labels, alpha);
        model.zero_grad();
        total_loss.backward();
        let learning_rate = 1.0 - 9.0 * (k as f64) / 100.0;
        for p in &model.parameters() {
            let delta = learning_rate * p.borrow().grad;
            p.borrow_mut().data -= delta;
        }

        println!(
            "step {k} loss {:.3}, accuracy, {:2}%",
            total_loss.borrow().data,
            accuracy * 100.0
        )
    }

    todo!()
}
