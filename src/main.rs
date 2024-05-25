use std::{
    cell::RefCell,
    collections::HashSet,
    hash::{Hash, Hasher},
    ops,
    rc::Rc,
};
use uuid::Uuid;

#[derive(Default)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    #[default]
    None,
}
pub struct ValueData {
    pub data: f64,
    pub grad: f64,
    pub uuid: Uuid,
    pub _backward: Option<fn(value: &ValueData)>,
    pub _prev: Vec<Value>,
    pub _op: Option<Operation>,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

impl ops::Deref for Value {
    type Target = Rc<RefCell<ValueData>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ValueData {
    fn new(data: f64) -> ValueData {
        ValueData {
            data,
            grad: 0.0,
            uuid: Uuid::new_v4(),
            _backward: None,
            _prev: Vec::new(),
            _op: None,
        }
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow().uuid.hash(state);
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().uuid == other.borrow().uuid
    }
}

impl Eq for Value {}

impl Value {
    pub fn new(value: ValueData) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }

    pub fn pow(&self, power: f64) -> Value {
        let result = Value::from(self.borrow().data.powf(power));
        result.borrow_mut()._op = Some(Operation::Pow);
        result.borrow_mut()._prev = vec![self.clone(), Value::from(power)];

        result.borrow_mut()._backward = Some(|val: &ValueData| {
            let base = val._prev[0].borrow().data;
            let p = val._prev[1].borrow().data;
            val._prev[0].borrow_mut().grad += p * base.powf(p - 1.0) * val.grad;
        });

        result
    }

    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<Value> = HashSet::new();
        self.build_topo(&mut topo, &mut visited);
        self.borrow_mut().grad = 1.0;
        topo.reverse();
        topo.iter().for_each(|v| {
            if let Some(backprop) = v.borrow()._backward {
                backprop(&v.borrow())
            }
        });
    }

    fn build_topo(&self, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if visited.insert(self.clone()) {
            self.borrow()
                ._prev
                .clone()
                .into_iter()
                .for_each(|child| child.build_topo(topo, visited));
            topo.push(self.to_owned())
        }
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(ValueData::new(t.into()))
    }
}

impl ops::Add<&Value> for &Value {
    type Output = Value;
    fn add(self, _rhs: &Value) -> Value {
        let result = Value::from(self.borrow().data + _rhs.borrow().data);
        result.borrow_mut()._op = Some(Operation::Add);
        // result.borrow_mut()._prev = vec![self, _rhs];
        result.borrow_mut()._prev = vec![self.clone(), _rhs.clone()];
        result.borrow_mut()._backward = Some(|val: &ValueData| {
            val._prev[0].borrow_mut().grad += val.grad;
            val._prev[1].borrow_mut().grad += val.grad;
        });

        result
    }
}

impl ops::Mul<&Value> for &Value {
    type Output = Value;
    fn mul(self, _rhs: &Value) -> Value {
        let result = Value::from(self.borrow().data * _rhs.borrow().data);
        result.borrow_mut()._op = Some(Operation::Add);
        // result.borrow_mut()._prev = vec![self, _rhs];
        result.borrow_mut()._prev = vec![self.clone(), _rhs.clone()];
        result.borrow_mut()._backward = Some(|val: &ValueData| {
            val._prev[0].borrow_mut().grad += val._prev[1].borrow().data * val.grad;
            val._prev[1].borrow_mut().grad += val._prev[0].borrow().data * val.grad;
        });

        result
    }
}

impl ops::Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value {
        let result = self * &Value::from(-1.0);

        result
    }
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sanity_check_add() {
        let a = Value::from(-4.0);
        let b = Value::from(2.0);
        let c = &a + &b;
        assert_eq!(c.borrow().data, -2.0);
        assert_ne!(c.borrow().data, 2.0);
    }
    #[test]
    fn sanity_check_mul() {
        let a = Value::from(-4.0);
        let b = Value::from(2.0);
        let c = &a * &b;
        assert_eq!(c.borrow().data, -8.0);
        assert_ne!(c.borrow().data, 2.0);
    }
    #[test]
    fn sanity_check_neg() {
        let a = Value::from(1.0);
        let b = -&a;
        assert_eq!(b.borrow().data, -1.0);
    }
    #[test]
    fn sanity_check_pow() {
        let a = Value::from(2.0);
        let b = 3.0;
        let c = a.pow(b);
        assert_eq!(c.borrow().data, 8.0);
    }
    #[test]
    fn sanity_check_backprop_add() {
        let a = Value::from(1.0);
        let b = Value::from(2.0);
        let c = Value::from(3.0);
        let d = &a + &b;
        let e = &c + &d;
        let vals = vec![&a, &b, &c, &d, &e];
        vals.clone()
            .iter()
            .for_each(|x| assert_eq!(x.borrow().grad, 0.0));
        e.backward();
        vals.clone()
            .iter()
            .for_each(|f| assert_eq!(f.borrow().grad, 1.0));
        // assert_eq!(a.borrow().grad, 0.0);
        // assert_eq!(b.borrow().grad, 0.0);
        // assert_eq!(c.borrow().grad, 0.0);
        // assert_eq!(d.borrow().grad, 0.0);
        // assert_eq!(e.borrow().grad, 0.0);
    }
    #[test]
    fn sanity_check_backprop() {
        let a = Value::from(1.0);
        let b = Value::from(2.0);
        let c = Value::from(3.0);
        let d = &a * &b;
        let e = &c * &d;
        let vals = vec![&a, &b, &c, &d, &e];
        vals.clone()
            .iter()
            .for_each(|x| assert_eq!(x.borrow().grad, 0.0));
        e.backward();
        //
        assert_eq!(a.borrow().grad, 6.0);
        assert_eq!(b.borrow().grad, 3.0);
        assert_eq!(c.borrow().grad, 2.0);
        assert_eq!(d.borrow().grad, 3.0);
        assert_eq!(e.borrow().grad, 1.0);
    }
}
