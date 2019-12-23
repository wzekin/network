use conv::prelude::*;
use nalgebra::allocator::Allocator;
use nalgebra::storage::Storage;
use nalgebra::{DefaultAllocator, Dim, DimName, Dynamic, Matrix, MatrixMN};
use rand::Rng;
use std::boxed::Box;

//type Mat = Vec<Vec<f64>>;

#[derive(Debug)]
pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
    pub loss: Box<dyn Loss>,
}

impl Model {
    pub fn new(loss: Box<dyn Loss>) -> Self {
        Model {
            layers: Vec::new(),
            loss,
        }
    }
    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer)
    }

    pub fn forward(
        &mut self,
        input: &MatrixMN<f64, Dynamic, Dynamic>,
        training: bool,
    ) -> MatrixMN<f64, Dynamic, Dynamic> {
        let mut pre = self.layers[0].forward(input, training);
        for i in 1..self.layers.len() {
            //println!("pre[{}]:{:?}", i - 1, pre);
            pre = self.layers[i].forward(&pre, training);
        }
        //println!("pre[{}]:{:?}", "out", pre);
        pre
    }

    pub fn predict(
        &mut self,
        input: &MatrixMN<f64, Dynamic, Dynamic>,
    ) -> MatrixMN<f64, Dynamic, Dynamic> {
        self.forward(input, false)
    }

    pub fn fit(
        &mut self,
        n: usize,
        x: &Vec<MatrixMN<f64, Dynamic, Dynamic>>,
        y: &Vec<MatrixMN<f64, Dynamic, Dynamic>>,
        lamada: f64,
    ) {
        //println!("X:{:?}", x);
        //println!("Y:{:?}", y);
        for _ in 0..n {
            for i in 0..x.len() {
                let y_ = self.forward(&x[i], true);
                let loss = self.loss.forward(&y_, &y[i]);
                println!("loss:{:?}", loss);
                println!("y_:{}   y:{}", y_[0], y[i]);
                let mut grads = self.loss.backward();
                //println!("grads[{}]:{:?}", "out", grads);
                for j in (0..self.layers.len()).rev() {
                    grads = self.layers[j].backward(&grads);
                    //println!("grads[{}]:{:?}", j, grads);
                }
                for layer in self.layers.iter_mut() {
                    layer.update(lamada);
                }
                for layer in self.layers.iter_mut() {
                    layer.clear();
                }
            }
        }
    }
}

pub trait Layer: std::fmt::Debug {
    fn forward(
        &mut self,
        input: &MatrixMN<f64, Dynamic, Dynamic>,
        training: bool,
    ) -> MatrixMN<f64, Dynamic, Dynamic>;
    fn backward(
        &mut self,
        grads: &MatrixMN<f64, Dynamic, Dynamic>,
    ) -> MatrixMN<f64, Dynamic, Dynamic>;
    fn update(&mut self, lamada: f64);
    fn clear(&mut self);
}

//pub trait Func: std::fmt::Debug {
//fn forward(
//&mut self,
//input: &MatrixMN<f64, Dynamic, Dynamic>,
//) -> MatrixMN<f64, Dynamic, Dynamic>;
//fn backward(
//&mut self,
//grads: &MatrixMN<f64, Dynamic, Dynamic>,
//) -> MatrixMN<f64, Dynamic, Dynamic>;
//}

pub trait Loss: std::fmt::Debug {
    fn forward(
        &mut self,
        input: &MatrixMN<f64, Dynamic, Dynamic>,
        true_value: &MatrixMN<f64, Dynamic, Dynamic>,
    ) -> f64;
    fn backward(&mut self) -> MatrixMN<f64, Dynamic, Dynamic>;
}
#[derive(Debug)]
pub struct Dense {
    units: usize,
    input_size: usize,
    grads: MatrixMN<f64, Dynamic, Dynamic>,
    weights: MatrixMN<f64, Dynamic, Dynamic>,
    input: MatrixMN<f64, Dynamic, Dynamic>,
    output: MatrixMN<f64, Dynamic, Dynamic>,
}
impl Dense {
    pub fn new(input: usize, units: usize) -> Dense {
        Dense {
            units: units,
            input_size: input,
            weights: MatrixMN::<f64, Dynamic, Dynamic>::new_random(input, units),
            grads: MatrixMN::<f64, Dynamic, Dynamic>::zeros(input, units),
            input: MatrixMN::<f64, Dynamic, Dynamic>::zeros(1, input),
            output: MatrixMN::<f64, Dynamic, Dynamic>::zeros(1, units),
        }
    }
}

impl Layer for Dense {
    fn forward(
        &mut self,
        input: &MatrixMN<f64, Dynamic, Dynamic>,
        training: bool,
    ) -> MatrixMN<f64, Dynamic, Dynamic> {
        self.input = input.clone();
        self.output = input * self.weights.clone();
        self.output.clone()
    }

    fn backward(
        &mut self,
        grads: &MatrixMN<f64, Dynamic, Dynamic>,
    ) -> MatrixMN<f64, Dynamic, Dynamic> {
        let trans = grads.transpose();
        //println!("grads:{}:{}", trans.ncols(), trans.nrows());
        //println!("input:{}:{}", self.input.ncols(), self.input.nrows());
        self.grads = trans * self.input.clone() ;
        //println!("grad:{}:{}", self.grads.ncols(), self.grads.nrows());
        self.grads.clone()
    }
    fn update(&mut self, lamada: f64) {
        self.weights += lamada * self.grads.clone().transpose();
    }
    fn clear(&mut self) {
        self.grads = MatrixMN::<f64, Dynamic, Dynamic>::zeros(self.units, self.input_size)
    }
}

#[derive(Debug)]
pub struct Sigmoid {
    output: MatrixMN<f64, Dynamic, Dynamic>,
}
impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {
            output: MatrixMN::<f64, Dynamic, Dynamic>::zeros(1, 1),
        }
    }
}
impl Layer for Sigmoid {
    fn forward(
        &mut self,
        input: &MatrixMN<f64, Dynamic, Dynamic>,
        _: bool,
    ) -> MatrixMN<f64, Dynamic, Dynamic> {
        self.output = input.map(|x| 1.0f64 / (1.0f64 + (-1.0f64 * x).exp()));
        self.output.clone()
    }

    fn backward(
        &mut self,
        grads: &MatrixMN<f64, Dynamic, Dynamic>,
    ) -> MatrixMN<f64, Dynamic, Dynamic> {
        let mut grad = MatrixMN::<f64, Dynamic, Dynamic>::zeros(1, self.output.ncols());
        //println!("grads:{}:{}", grads.ncols(), grad.nrows());
        //println!("output:{}:{}", self.output.ncols(), self.output.nrows());
        for i in 0..self.output.ncols() {
            grad[i] = grads.column(i).sum() * self.output[i] * (1.0 - self.output[i]);
        }
        grad
    }
    fn update(&mut self, lamada: f64) {}
    fn clear(&mut self) {}
}

#[derive(Debug)]
pub struct CrossEntropy {
    input: MatrixMN<f64, Dynamic, Dynamic>,
    true_value: MatrixMN<f64, Dynamic, Dynamic>,
}
impl CrossEntropy {
    pub fn new() -> Self {
        CrossEntropy {
            input: MatrixMN::<f64, Dynamic, Dynamic>::zeros(0, 0),
            true_value: MatrixMN::<f64, Dynamic, Dynamic>::zeros(0, 0),
        }
    }
}
impl Loss for CrossEntropy {
    fn forward(
        &mut self,
        input: &MatrixMN<f64, Dynamic, Dynamic>,
        true_value: &MatrixMN<f64, Dynamic, Dynamic>,
    ) -> f64 {
        self.input = input.clone();
        self.true_value = true_value.clone();
        let one =
            MatrixMN::<f64, Dynamic, Dynamic>::from_element(input.nrows(), input.ncols(), 1.0);
        -(input.map(|i| i.log2()).dot(&true_value)
            + (one.clone() - input.clone())
                .map(|i| i.log2())
                .dot(&(one - true_value)))
            / input.ncols().value_as::<f64>().unwrap()
        //return logistic_regression_loss(&self.input, &true_value[0]);
    }

    fn backward(&mut self) -> MatrixMN<f64, Dynamic, Dynamic> {
        let one = MatrixMN::<f64, Dynamic, Dynamic>::from_element(
            self.input.nrows(),
            self.input.ncols(),
            1.0,
        );
        self.true_value.zip_map(&self.input, |i, j| i / j)
            - (one.clone() - self.true_value.clone())
                .zip_map(&(one - self.input.clone()), |i, j| i / j)
    }
}

//pub fn point_mul(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
//assert_eq!(a.len(), b.len());

//let mut ans: f64 = 0.0;
//for i in 0..a.len() {
//ans += a[i] * b[i];
//}
//ans
//}

//pub fn vec_sigmoid(x: &Vec<f64>) -> Vec<f64> {
//x.iter().map(|v| sigmoid(v)).collect()
//}

pub fn logistic_regression_loss(input: &Vec<f64>, true_value: &Vec<f64>) -> f64 {
    assert_eq!(input.len(), true_value.len());

    let value: f64 = (0..input.len())
        .map(|i| input[i].log2() * true_value[i] + (1.0 - input[i]).log2() * (1.0 - true_value[i]))
        .sum();
    let len: f64 = input.len().value_as::<f64>().unwrap();
    -value / len
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nn_test() {
        let X: Vec<MatrixMN<f64, Dynamic, Dynamic>> = vec![
            MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 2, vec![100.0, 10.0]),
            MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 2, vec![10.0, 200.0]),
            MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 2, vec![10.0, 220.0]),
            MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 2, vec![200.0, 15.0]),
        ];
        let Y: Vec<MatrixMN<f64, Dynamic, Dynamic>> = vec![
            MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 1, vec![1.0]),
            MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 1, vec![0.0]),
            MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 1, vec![0.0]),
            MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 1, vec![1.0]),
        ];
        let mut model = Model::new(Box::new(CrossEntropy::new()));
        model.add(Box::new(Dense::new(2, 5)));
        model.add(Box::new(Sigmoid::new()));
        model.add(Box::new(Dense::new(5, 1)));
        model.add(Box::new(Sigmoid::new()));
        //println!("model init: {:?}", model);
        model.fit(1000, &X, &Y, 0.01);
        //println!("{:?}", model);
    }
}
