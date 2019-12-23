# BP神经网络实现

* github: github.com/wzekin/network

## 基本框架

* 基于链式求导法则实现求导，反向传播，每一层神经网络都实现了如下函数

  ```rust
  pub trait Layer: std::fmt::Debug {
      fn forward(&mut self, input: &Mat, training: bool) -> Mat;
      fn backward(&mut self, grads: &Mat) -> Mat;
      fn update(&mut self, lamada: f64);
      fn clear(&mut self);
  }
  ```

* Loss函数实现

  ```rust
   pub trait Loss: std::fmt::Debug {
       fn forward(&mut self, input: &Mat, true_value: &Mat) -> f64;
       fn backward(&mut self) -> Mat;
  }
  ```

  

* 总体实现

  ```rust
  impl Model {
      pub fn new(loss: Box<dyn Loss>) -> Self
      pub fn add(&mut self, layer: Box<dyn Layer>)
      pub fn forward(&mut self, input: &Mat, training: bool)
      pub fn predict(&mut self, input: &Mat) -> Mat
      pub fn fit(&mut self, n: usize, x: &Vec<Mat>, y: &Vec<Mat>, lamada: f64)
  }
  ```
  
  
  
* 使用

  ``` rust
  let mut model = Model::new(Box::new(CrossEntropyLayer::new()));
  model.add(Box::new(Dense::new(2, 5)));
  model.add(Box::new(SigmoidLayer::new()));
  model.add(Box::new(Dense::new(5, 1)));
  model.add(Box::new(SigmoidLayer::new()));
  model.fit(1000, &X, &Y, 0.01);
  ```

## 全连接层

* 正向传播
  $$
  z = w \sdot x
  $$
  
* 反向传播
  $$
  \delta = \sum\delta_{下层} * x
  $$

* 权值更新
  $$
  w = w - \eta \delta
  $$

## Sigmoid激活函数层

* 正向传播
  $$
  z = \frac1{1 + exp（-x）}
  $$
  
* 反向传播
  $$
  \delta = z_{输出}(1-z_{输出})\sum\delta_{下层}
  $$
  

## Logistics Loss损失函数层

* 正向传播
  $$
  \hat{y} = - \frac1m \sum (1-y)log(1-x) + ylogx
  $$
  
* 反向传播
  $$
  \delta = \sum \frac{\hat{y}}{x_i} + \frac{1-\hat{y}}{1-x_i}
  $$

## 接下来要做的

* 标准化权值**batch normalization**
* CNN
* RNN