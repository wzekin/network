use conv::prelude::*;
use rand::Rng;
use std::boxed::Box;

type Mat = Vec<Vec<f64>>;

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

    pub fn forward(&mut self, input: &Mat, training: bool) -> Mat {
        let mut pre = self.layers[0].forward(input, training);
        for i in 1..self.layers.len() {
            //println!("pre[{}]:{:?}", i - 1, pre);
            pre = self.layers[i].forward(&pre, training);
        }
        //println!("pre[{}]:{:?}", "out", pre);
        pre
    }

    pub fn predict(&mut self, input: &Mat) -> Mat {
        self.forward(input, false)
    }

    pub fn fit(&mut self, n: usize, x: &Vec<Mat>, y: &Vec<Mat>, lamada: f64) {
        println!("X:{:?}", x);
        println!("Y:{:?}", y);
        for _ in 0..n {
            for i in 0..x.len() {
                let y_ = self.forward(&x[i], true);
                let loss = self.loss.forward(&y_, &y[i]);
                println!("loss:{:?}", loss);
                println!("y_:{}   y:{}", y_[0][0], y[i][0][0]);
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
    fn forward(&mut self, input: &Mat, training: bool) -> Mat;
    fn backward(&mut self, grads: &Mat) -> Mat;
    fn update(&mut self, lamada: f64);
    fn clear(&mut self);
}

pub trait Loss: std::fmt::Debug {
    fn forward(&mut self, input: &Mat, true_value: &Mat) -> f64;
    fn backward(&mut self) -> Mat;
}
#[derive(Debug)]
pub struct Dense {
    units: usize,
    input_size: usize,
    weights: Mat,
    grads: Mat,
    input: Vec<f64>,
    output: Vec<f64>,
}
impl Dense {
    pub fn new(input: usize, units: usize) -> Dense {
        let mut rng = rand::thread_rng();
        let weights = (0..units)
            .map(|_| {
                (0..input)
                    .map(|_| rng.gen_range::<f64, f64, f64>(-1.0, 1.0))
                    .collect()
            })
            .collect();
        Dense {
            units: units,
            input_size: input,
            weights: weights,
            grads: vec![vec![0.0f64; input]; units],
            input: Vec::new(),
            output: Vec::new(),
        }
    }
}
impl Layer for Dense {
    fn forward(&mut self, input: &Mat, _: bool) -> Mat {
        assert_eq!(1, input.len());
        assert_eq!(self.input_size, input[0].len());

        self.input = input[0].clone();
        let rs: Vec<f64> = (0..self.units)
            .map(|i| point_mul(&self.weights[i], &self.input))
            .collect();
        self.output = rs.clone();
        vec![rs]
    }

    fn backward(&mut self, grads: &Mat) -> Mat {
        assert_eq!(1, grads.len());
        assert_eq!(self.units, grads[0].len());

        for (i, grad) in grads[0].iter().enumerate() {
            for (j, input) in self.input.iter().enumerate() {
                self.grads[i][j] = grad * input;
            }
        }
        self.grads.clone()
    }
    fn update(&mut self, lamada: f64) {
        for i in 0..self.units {
            for j in 0..self.input_size {
                self.weights[i][j] += lamada * self.grads[i][j];
            }
        }
    }
    fn clear(&mut self) {
        self.grads = vec![vec![0.0f64; self.input_size]; self.units]
    }
}

#[derive(Debug)]
pub struct SigmoidLayer {
    input: Vec<f64>,
    output: Vec<f64>,
}
impl SigmoidLayer {
    pub fn new() -> Self {
        SigmoidLayer {
            input: vec![],
            output: vec![],
        }
    }
}
impl Layer for SigmoidLayer {
    fn forward(&mut self, input: &Mat, _: bool) -> Mat {
        assert_eq!(1, input.len());

        self.input = input[0].clone();
        let rs = vec_sigmoid(&self.input);
        self.output = rs.clone();
        vec![rs]
    }

    fn backward(&mut self, grads: &Mat) -> Mat {
        let mut rs: Mat = vec![vec![0.0f64; self.input.len()]; 1usize];
        for (i, out) in self.output.iter().enumerate() {
            for cell in grads.iter() {
                rs[0][i] = rs[0][i] + cell[i] * out * (1.0f64 - out);
            }
        }
        rs
    }
    fn update(&mut self, _: f64) {}
    fn clear(&mut self) {}
}

#[derive(Debug)]
pub struct CrossEntropyLayer {
    input: Vec<f64>,
    true_value: Vec<f64>,
}
impl CrossEntropyLayer {
    pub fn new() -> Self {
        CrossEntropyLayer {
            input: vec![],
            true_value: vec![],
        }
    }
}
impl Loss for CrossEntropyLayer {
    fn forward(&mut self, input: &Mat, true_value: &Mat) -> f64 {
        assert_eq!(1, input.len());
        assert_eq!(1, true_value.len());

        self.input = input[0].clone();
        self.true_value = true_value[0].clone();
        return logistic_regression_loss(&self.input, &true_value[0]);
    }

    fn backward(&mut self) -> Mat {
        let mut rs: Mat = vec![vec![0.0f64; self.input.len()]; 1usize];
        for (i, out) in self.true_value.iter().enumerate() {
            rs[0][i] = out / self.input[i] - (1.0 - out) / (1.0 - self.input[i]);
        }
        rs
    }
}

pub fn point_mul(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    assert_eq!(a.len(), b.len());

    let mut ans: f64 = 0.0;
    for i in 0..a.len() {
        ans += a[i] * b[i];
    }
    ans
}

pub fn sigmoid(x: &f64) -> f64 {
    1.0f64 / (1.0f64 + (-1.0f64 * x).exp())
}

pub fn vec_sigmoid(x: &Vec<f64>) -> Vec<f64> {
    x.iter().map(|v| sigmoid(v)).collect()
}

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
        let X: Vec<Mat> = vec![
            vec![vec![10.0, 20.0]],
            vec![vec![10.0, -20.0]],
            vec![vec![-20.0, 10.0]],
            vec![vec![20.0, 10.0]],
        ];
        let Y: Vec<Mat> = vec![
            vec![vec![1.0]],
            vec![vec![0.0]],
            vec![vec![0.0]],
            vec![vec![1.0]],
        ];
        let mut model = Model::new(Box::new(CrossEntropyLayer::new()));
        model.add(Box::new(Dense::new(2, 5)));
        model.add(Box::new(SigmoidLayer::new()));
        model.add(Box::new(Dense::new(5, 1)));
        model.add(Box::new(SigmoidLayer::new()));
        println!("model init: {:?}", model);
        model.fit(1000, &X, &Y, 0.01);
        println!("{:?}", model);
    }
}
