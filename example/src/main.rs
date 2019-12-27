use model_macro::model;
use nalgebra::{MatrixMN, U1, U2, U3,U5};
use network::layer::Dense;
use network::loss::Logistics;
use network::func::Sigmoid;
use std::boxed::Box;

model! {
    Dense::<U1,U2,U1,U5>::new(Box::new(Sigmoid::<U1,U5>::new()));
    Dense::<U1,U5,U1,U3>::new(Box::new(Sigmoid::<U1,U3>::new()));
    Dense::<U1,U3,U1,U1>::new(Box::new(Sigmoid::<U1,U1>::new()));
}

fn main() {
    let x: Vec<MatrixMN<f64, U1, U2>> = vec![
        MatrixMN::<f64, U1, U2>::from_vec(vec![2.0, 2.0]),
        MatrixMN::<f64, U1, U2>::from_vec(vec![-1.0, 2.0]),
        MatrixMN::<f64, U1, U2>::from_vec(vec![1.0, -2.2]),
        MatrixMN::<f64, U1, U2>::from_vec(vec![-2.3, -1.0]),
    ];
    let y: Vec<MatrixMN<f64, U1, U1>> = vec![
        MatrixMN::<f64, U1, U1>::from_vec(vec![1.0]),
        MatrixMN::<f64, U1, U1>::from_vec(vec![0.0]),
        MatrixMN::<f64, U1, U1>::from_vec(vec![0.0]),
        MatrixMN::<f64, U1, U1>::from_vec(vec![1.0]),
    ];
    let mut model = Model::new(Box::new(Logistics::new()));
    //println!("model init: {:?}", model);
    model.fit(1000, &x, &y, 1.0);
    //println!("{:?}", model);
}
