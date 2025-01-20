use crate::micrograd::Value;
use crate::Result;

type Matrix = Vec<Vec<Value>>;

struct AttentionLayer {
    query_weights: Matrix,
    key_weights: Matrix,
    value_up_weights: Matrix,
    value_down_weights: Matrix,
}

impl AttentionLayer {
    fn get_dimension_k_size(&self) -> Value {
        Value::from(self.key_weights.len() as f64)
    }

    fn compute_attention(&self, input: Matrix) -> Result<Matrix> {
        let queries = matrix_mult(&input, self.query_weights);
        let keys = matrix_mult(&input, self.key_weights);
        let keys_transposed = transpose_matrix(&keys);
        let values = matrix_mult(self.value_up_weights, self.value_down_weights);

        let output = softmax((
            matrix_mult(queries, keys_transposed),
            sqrt(self.get_dimension_k_size()),
        ));

        Ok(input)
    }
}

fn matrix_mult(m1: Matrix, m2: Matrix) -> Matrix {
    m1
}

fn transpose_matrix(m1: &Matrix) -> Matrix {
    let mut output: Matrix = vec![vec![Value::from(0.0); m1.len()]; m1[0].len()];
    m1.iter().map(f)
}
