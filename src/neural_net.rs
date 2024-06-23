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

    pub fn forward(&self, activations: &Vec<Value>) -> Value {
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
    pub fn parameters(&self) -> Vec<Value> {
        let mut output = self.weights.clone();
        output.push(self.bias.clone());
        output
    }
}
impl From<(Vec<f64>, f64, bool)> for Neuron {
    fn from(vals: (Vec<f64>, f64, bool)) -> Neuron {
        Neuron {
            weights: vals
                .0
                .iter()
                .map(|val| Value::from(val.to_owned()))
                .collect(),
            bias: Value::from(vals.1),
            nonlin: vals.2,
        }
    }
}
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin, nonlin)).collect(),
        }
    }

    // pub fn parameters(&self) {}

    // pub fn forward(&self, )
}

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
        let neuron = Neuron::from((vec![2.0, 1.0], 0.0, false));
        assert_eq!(neuron.forward(&activations).borrow().data, 6.0);
        let neuron2 = Neuron::from((vec![1.0, 2.0], 1.0, false));
        assert_eq!(neuron2.forward(&activations).borrow().data, 7.0);
        let neuron3 = Neuron::from((vec![-2.0, 1.0], 0.0, true));
        assert_eq!(neuron3.forward(&activations).borrow().data, 0.0);
    }
}
