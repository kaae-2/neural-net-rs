use crate::engine::Value;
use rand::distributions::{Distribution, Uniform};

struct Module;

struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Neuron {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new_inclusive(-1.0, 1.0);
        //  TODO: make into 1 line loop
        let mut weights = vec![];
        for _ in 0..nin {
            weights.push(Value::from(uniform.sample(&mut rng)))
        }
        // println!("{:?}", weights);
        Neuron {
            weights: weights,
            bias: Value::from(0.0),
            nonlin,
        }
    }
    pub fn from(vals: Vec<Value>, nonlin: bool) -> Neuron {
        Neuron {
            weights: vals,
            bias: Value::from(0.0),
            nonlin,
        }
    }

    pub fn forward(&self, activations: Vec<Value>) -> Value {
        let mut result: Value = self
            .weights
            .iter()
            .zip(activations.iter())
            .map(|(wi, xi)| xi * wi)
            // .collect()
            .sum();
        result += &self.bias;
        let output = if self.nonlin { result.relu() } else { result };

        output
    }
}
// TODO: Fix from impl to take values
// impl<T: Into<Vec<Value>>> From<T> for Neuron {
//     fn from(t: T) -> Neuron {
//         Neuron::from(t, false)
//     }
// }
struct Layer;

struct MLP;

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn sanity_check_neuron() {
        let neuron = Neuron::new(10, false);
        assert_eq!(neuron.weights.len(), 10);
    }

    #[test]
    fn sanity_check_neuron_forward() {
        let activations = vec![Value::from(2.0); 2];
        let neuron = Neuron::from(vec![Value::from(1.0), Value::from(0.0)], false);
        let check = neuron.forward(activations);
        println!("{:?}", check)
    }
}
