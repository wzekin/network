use nalgebra::allocator::Allocator;
use nalgebra::constraint::{DimEq, ShapeConstraint};
use nalgebra::{DefaultAllocator, Dim, DimName, MatrixMN, U1};
use std::marker::PhantomData;

pub trait Layer<IR, IC, OR, OC>
where
    IR: Dim + DimName,
    IC: Dim + DimName,
    OR: Dim + DimName,
    OC: Dim + DimName,
    DefaultAllocator: Allocator<f64, IR, IC> + Allocator<f64, OR, OC>,
{
    fn forward(&mut self, input: MatrixMN<f64, IR, IC>, training: bool) -> MatrixMN<f64, OR, OC>;
    fn backward(&mut self, grads: MatrixMN<f64, OR, OC>) -> MatrixMN<f64, IR, IC>;
    fn update(&mut self, lamada: f64);
    fn clear(&mut self);
}

pub struct Dense<IR, IC, OR, OC>
where
    IR: DimName + Dim,
    IC: DimName + Dim,
    OR: DimName + Dim,
    OC: DimName + Dim,
    DefaultAllocator: Allocator<f64, U1, IC> + Allocator<f64, U1, OC> + Allocator<f64, IC, OC>,
    ShapeConstraint: DimEq<IR, U1> + DimEq<OR, U1>,
{
    grads: MatrixMN<f64, IC, OC>,
    bais_grads: MatrixMN<f64, U1, OC>,
    weights: MatrixMN<f64, IC, OC>,
    bais: MatrixMN<f64, U1, OC>,
    input: MatrixMN<f64, U1, IC>,
    output: MatrixMN<f64, U1, OC>,
    marker1: PhantomData<IR>,
    marker2: PhantomData<OR>,
}

impl<IR, IC, OR, OC> Dense<IR, IC, OR, OC>
where
    IR: DimName + Dim,
    IC: DimName + Dim,
    OR: DimName + Dim,
    OC: DimName + Dim,
    DefaultAllocator: Allocator<f64, U1, IC> + Allocator<f64, U1, OC> + Allocator<f64, IC, OC>,
    ShapeConstraint: DimEq<IR, U1> + DimEq<OR, U1>,
{
    pub fn new() -> Dense<IR, IC, OR, OC> {
        Dense {
            weights: MatrixMN::<f64, IC, OC>::new_random(),
            bais: MatrixMN::<f64, U1, OC>::new_random(),
            grads: MatrixMN::<f64, IC, OC>::zeros(),
            bais_grads: MatrixMN::<f64, U1, OC>::zeros(),
            input: MatrixMN::<f64, U1, IC>::zeros(),
            output: MatrixMN::<f64, U1, OC>::zeros(),
            marker1: PhantomData,
            marker2: PhantomData,
        }
    }
}

impl<IR, IC, OR, OC> Layer<U1, IC, U1, OC> for Dense<IR, IC, OR, OC>
where
    IR: DimName + Dim,
    IC: DimName + Dim,
    OR: DimName + Dim,
    OC: DimName + Dim,
    DefaultAllocator: Allocator<f64, U1, IC>
        + Allocator<f64, U1, OC>
        + Allocator<f64, IC, OC>
        + Allocator<f64, IC, U1>
        + Allocator<f64, OC, IC>,
    ShapeConstraint: DimEq<IR, U1> + DimEq<OR, U1>,
{
    fn forward(&mut self, input: MatrixMN<f64, U1, IC>, training: bool) -> MatrixMN<f64, U1, OC> {
        // println!("dense_forward_in:{:?}", input);
        self.input = input;
        self.output = &self.input * &self.weights + &self.bais;
        // println!("dense_forward_out:{:?}", self.output);
        self.output.clone()
    }

    fn backward(&mut self, grads: MatrixMN<f64, U1, OC>) -> MatrixMN<f64, U1, IC> {
        // println!("dense_in:{:?}", grads);
        self.grads = self.input.transpose() * &grads;
        self.bais_grads = grads;
        let out = MatrixMN::<f64, U1, OC>::from_element(1.0) * self.grads.transpose();
        // println!("dense_out:{:?}", out);
        out
    }
    fn update(&mut self, lamada: f64) {
        self.weights -= &self.grads * lamada;
        self.bais -= &self.bais_grads * lamada;
        // println!("dense_weights:{:?}", self.weights)
    }
    fn clear(&mut self) {
        self.grads = MatrixMN::<f64, IC, OC>::zeros()
    }
}

pub struct Sigmoid<R, C, R1, C1>
where
    R: DimName + Dim,
    C: DimName + Dim,
    R1: DimName + Dim,
    C1: DimName + Dim,
    DefaultAllocator: Allocator<f64, R, C>,
    ShapeConstraint: DimEq<R, R1> + DimEq<C, C1>,
{
    output: MatrixMN<f64, R, C>,
    marker1: PhantomData<R1>,
    marker2: PhantomData<C1>,
}
impl<R, C, R1, C1> Sigmoid<R, C, R1, C1>
where
    R: DimName + Dim,
    C: DimName + Dim,
    R1: DimName + Dim,
    C1: DimName + Dim,
    DefaultAllocator: Allocator<f64, R, C>,
    ShapeConstraint: DimEq<R, R1> + DimEq<C, C1>,
{
    pub fn new() -> Self {
        Sigmoid {
            output: MatrixMN::<f64, R, C>::zeros(),
            marker1: PhantomData,
            marker2: PhantomData,
        }
    }
}

impl<R, C, R1, C1> Layer<R, C, R, C> for Sigmoid<R, C, R1, C1>
where
    R: DimName + Dim,
    C: DimName + Dim,
    R1: DimName + Dim,
    C1: DimName + Dim,
    DefaultAllocator: Allocator<f64, R, C>,
    ShapeConstraint: DimEq<R, R1> + DimEq<C, C1>,
{
    fn forward(&mut self, input: MatrixMN<f64, R, C>, _: bool) -> MatrixMN<f64, R, C> {
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
    fn update(&mut self, _: f64) {}
    fn clear(&mut self) {}
}
