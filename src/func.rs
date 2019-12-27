use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, DimName, MatrixMN};
pub trait Func<R, C>
where
    R: Dim + DimName,
    C: Dim + DimName,
    DefaultAllocator: Allocator<f64, R, C>,
{
    fn forward(&mut self, input: MatrixMN<f64, R, C>) -> MatrixMN<f64, R, C>;
    fn backward(&mut self, grads: MatrixMN<f64, R, C>) -> MatrixMN<f64, R, C>;
}
pub struct Sigmoid<R, C>
where
    R: DimName + Dim,
    C: DimName + Dim,
    DefaultAllocator: Allocator<f64, R, C>,
{
    output: MatrixMN<f64, R, C>,
}
impl<R, C> Sigmoid<R, C>
where
    R: DimName + Dim,
    C: DimName + Dim,
    DefaultAllocator: Allocator<f64, R, C>,
{
    pub fn new() -> Self {
        Sigmoid {
            output: MatrixMN::<f64, R, C>::zeros(),
        }
    }
}

impl<R, C> Func<R, C> for Sigmoid<R, C>
where
    R: DimName + Dim,
    C: DimName + Dim,
    DefaultAllocator: Allocator<f64, R, C>,
{
    fn forward(&mut self, input: MatrixMN<f64, R, C>) -> MatrixMN<f64, R, C> {
        // println!("Sigmoid_forward_in:{:?}", input);
        self.output = input.map(|x| 1.0 / (1.0 + (-1.0 * x).exp()));
        // println!("Sigmoid_forward_out:{:?}", self.output);
        self.output.clone()
    }

    fn backward(&mut self, grads: MatrixMN<f64, R, C>) -> MatrixMN<f64, R, C> {
        // println!("Sigmoid_in:{:?}", grads);
        let ans = grads.zip_map(&self.output, |i, j| i * j * (1.0 - j));
        // println!("Sigmoid_out:{:?}", ans);
        ans
    }
}
