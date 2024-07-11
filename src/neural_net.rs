use crate::engine::Value;
use rand::distributions::{Distribution, Uniform};

#[derive(Debug)]
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
        output.insert(0, self.bias.clone());
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

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
    nin: usize,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin, nonlin)).collect(),
            nin,
        }
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons
            .iter()
            .fold(Vec::new(), |mut acc: Vec<Value>, neuron: &Neuron| {
                acc.extend(neuron.parameters());
                acc
            })
    }

    pub fn forward(&self, activations: &Vec<Value>) -> Vec<Value> {
        assert_eq!(
            activations.len(),
            self.nin,
            "activations must be same length as inputs to neurons"
        );
        let out = self
            .neurons
            .iter()
            .map(|neuron| neuron.forward(activations))
            .collect();
        // .fold(Vec::new(), |mut acc: Vec<Value>, neuron: &Neuron| {
        //     acc.extend(neuron.forward());
        //     acc
        // });
        out
    }
}

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(n_layer_size: Vec<usize>) -> MLP {
        let layers = (0..n_layer_size.len() - 1)
            .map(|i| {
                Layer::new(
                    n_layer_size[i].clone(),
                    n_layer_size[i + 1].clone(),
                    i != n_layer_size.len() - 2,
                )
            })
            .collect();
        MLP { layers }
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .fold(Vec::new(), |mut acc: Vec<Value>, layer: &Layer| {
                acc.extend(layer.parameters());
                acc
            })
    }

    pub fn forward(&self, input: Vec<Value>) -> Vec<Value> {
        self.layers
            .iter()
            .fold(input, |acc, layer| layer.forward(&acc))
    }
    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.borrow_mut().grad = 0.0;
        }
    }
}

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
        println!("{:?}", neuron3.forward(&activations));
    }
    #[test]
    fn test_layer_parameters() {
        let layer = Layer::new(4, 2, true);
        let parameters = layer.parameters();
        assert_eq!(parameters.len(), 10);
    }

    #[test]
    fn sanity_check_layer_forward() {
        let activations = vec![Value::from(2.0); 2];
        let layer = Layer::new(2, 3, true);
        let output = layer.forward(&activations);
        // println!("{:?}", layer.parameters());
        println!("{:?}", output);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn sanity_check_mlp_parameters() {
        let mlp = MLP::new(vec![2, 3, 1]);
        let parameters = mlp.parameters();
        assert_eq!(parameters.len(), 13);
    }

    #[test]
    fn sanity_check_mlp_forward() {
        let input = vec![Value::from(2.0); 2];
        let mlp = MLP::new(vec![2, 3, 1]);
        let output = mlp.forward(input);
        assert_eq!(output.len(), 1);
        println!("{:?}", output)
    }
}
