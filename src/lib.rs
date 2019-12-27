pub mod layer;
pub mod loss;

//struct Model1<IR, IC, OR, OC>
//where
    //IR: Dim + DimName,
    //IC: Dim + DimName,
    //OR: Dim + DimName,
    //OC: Dim + DimName,
    //DefaultAllocator: Allocator<f64, IR, IC> + Allocator<f64, OR, OC>,
//{
    //pub layer: Box<dyn Layer<IR, IC, OR, OC>>,
    //pub loss: Box<dyn Loss<OR, OC>>,
//}

//impl<IR, IC, OR, OC> Model1<IR, IC, OR, OC>
//where
    //IR: Dim + DimName,
    //IC: Dim + DimName,
    //OR: Dim + DimName,
    //OC: Dim + DimName,
    //DefaultAllocator: Allocator<f64, IR, IC> + Allocator<f64, OR, OC>,
//{
    //pub fn new(layer: Box<dyn Layer<IR, IC, OR, OC>>, loss: Box<dyn Loss<OR, OC>>) -> Self {
        //Model1 { layer, loss }
    //}

    //pub fn forward(
        //&mut self,
        //input: &MatrixMN<f64, IR, IC>,
        //training: bool,
    //) -> MatrixMN<f64, OR, OC> {
        //let pre = self.layer.forward(input, training);
        //pre
    //}

    //pub fn predict(&mut self, input: &MatrixMN<f64, IR, IC>) -> MatrixMN<f64, OR, OC> {
        //self.forward(input, false)
    //}

    //pub fn fit(
        //&mut self,
        //n: usize,
        //x: &Vec<MatrixMN<f64, IR, IC>>,
        //y: &Vec<MatrixMN<f64, OR, OC>>,
        //lamada: f64,
    //) {
        ////println!("X:{:?}", x);
        ////println!("Y:{:?}", y);
        //for _ in 0..n {
            //for i in 0..x.len() {
                //let y_ = self.forward(&x[i], true);
                //let loss = self.loss.forward(&y_, &y[i]);
                //println!("loss:{:?}", loss);
                //println!("y_:{}   y:{}", y_[0], y[i][0]);
                //let grads = self.loss.backward();
                //let _ = self.layer.backward(&grads);

                //self.layer.update(lamada);
                //self.layer.clear();
            //}
        //}
    //}
//}

//#[cfg(test)]
//mod tests {
//use super::*;
//#[test]
//fn nn_test() {
//let X: Vec<MatrixMN<f64, Dynamic, Dynamic>> = vec![
//MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 2, vec![100.0, 10.0]),
//MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 2, vec![10.0, 200.0]),
//MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 2, vec![10.0, 220.0]),
//MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 2, vec![200.0, 15.0]),
//];
//let Y: Vec<MatrixMN<f64, Dynamic, Dynamic>> = vec![
//MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 1, vec![1.0]),
//MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 1, vec![0.0]),
//MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 1, vec![0.0]),
//MatrixMN::<f64, Dynamic, Dynamic>::from_vec(1, 1, vec![1.0]),
//];
//let mut model = Model::new(Box::new(CrossEntropy::new()));
//model.add(Box::new(Dense::new(2, 5)));
//model.add(Box::new(Sigmoid::new()));
//model.add(Box::new(Dense::new(5, 1)));
//model.add(Box::new(Sigmoid::new()));
////println!("model init: {:?}", model);
//model.fit(1000, &X, &Y, 0.01);
////println!("{:?}", model);
//}
//}
