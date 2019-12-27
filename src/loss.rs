use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, DimName, MatrixMN};

pub trait Loss<R, C>
where
    R: Dim + DimName,
    C: Dim + DimName,
    DefaultAllocator: Allocator<f64, R, C>,
{
    fn forward(&mut self, input: &MatrixMN<f64, R, C>, true_value: &MatrixMN<f64, R, C>) -> f64;
    fn backward(&mut self) -> MatrixMN<f64, R, C>;
}

pub struct Logistics<R, C>
where
    R: Dim + DimName,
    C: Dim + DimName,
    DefaultAllocator: Allocator<f64, R, C>,
{
    input: MatrixMN<f64, R, C>,
    true_value: MatrixMN<f64, R, C>,
}
impl<R, C> Logistics<R, C>
where
    R: Dim + DimName,
    C: Dim + DimName,
    DefaultAllocator: Allocator<f64, R, C>,
{
    pub fn new() -> Self {
        Logistics {
            input: MatrixMN::<f64, R, C>::zeros(),
            true_value: MatrixMN::<f64, R, C>::zeros(),
        }
    }
}
impl<R, C> Loss<R, C> for Logistics<R, C>
where
    R: Dim + DimName,
    C: Dim + DimName,
    DefaultAllocator: Allocator<f64, R, C>,
{
    fn forward(&mut self, input: &MatrixMN<f64, R, C>, true_value: &MatrixMN<f64, R, C>) -> f64 {
        self.input = input.clone();
        self.true_value = true_value.clone();
        let one = MatrixMN::<f64, R, C>::from_element(1.0);
        -(input.map(|i| i.log2()).dot(&true_value)
            + (one.clone() - input.clone())
                .map(|i| i.log2())
                .dot(&(one - true_value)))
        //return logistic_regression_loss(&self.input, &true_value[0]);
    }

    fn backward(&mut self) -> MatrixMN<f64, R, C> {
        let one = MatrixMN::<f64, R, C>::from_element(1.0);
        -self.true_value.zip_map(&self.input, |i, j| i / j)
            + (one.clone() - self.true_value.clone())
                .zip_map(&(one - self.input.clone()), |i, j| i / j)
    }
}
