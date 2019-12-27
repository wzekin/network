use model_macro::model;
use nalgebra::{MatrixMN, U1, U2, U3};
use network::layer::{Dense, Sigmoid};
use network::loss::Logistics;

model! {
    Dense::<U1,U2,U1,U1>::new();
    Sigmoid::<U1,U1,U1,U1>::new();
    //Dense::<U1,U3,U1,U1>::new();
    //Sigmoid::<U1,U1,U1,U1>::new();
}

fn main() {
    let X: Vec<MatrixMN<f64, U1, U2>> = vec![
        MatrixMN::<f64, U1, U2>::from_vec(vec![2.0, 1.0]),
        MatrixMN::<f64, U1, U2>::from_vec(vec![1.0, 2.0]),
        MatrixMN::<f64, U1, U2>::from_vec(vec![1.0, 2.2]),
        MatrixMN::<f64, U1, U2>::from_vec(vec![2.3, 1.0]),
    ];
    let Y: Vec<MatrixMN<f64, U1, U1>> = vec![
        MatrixMN::<f64, U1, U1>::from_vec(vec![1.0]),
        MatrixMN::<f64, U1, U1>::from_vec(vec![0.0]),
        MatrixMN::<f64, U1, U1>::from_vec(vec![0.0]),
        MatrixMN::<f64, U1, U1>::from_vec(vec![1.0]),
    ];
    let mut model = Model::new(Box::new(Logistics::new()));
    //println!("model init: {:?}", model);
    model.fit(100, &X, &Y, 0.1);
    //println!("{:?}", model);
}
