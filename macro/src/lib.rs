extern crate proc_macro;
use self::proc_macro::TokenStream;
use quote::format_ident;
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::{parse_macro_input, ExprCall, Ident, Token, Type};

struct LayerMacro {
    name: Ident,
    layer_in_r: Type,
    layer_in_c: Type,
    layer_out_r: Type,
    layer_out_c: Type,
    init: ExprCall,
}

impl Parse for LayerMacro {
    fn parse(input: ParseStream) -> Result<Self> {
        let name = input.parse::<Ident>()?;
        input.parse::<Token![::]>()?;
        input.parse::<Token![<]>()?;
        let layer_in_r: Type = input.parse()?;
        input.parse::<Token![,]>()?;
        let layer_in_c: Type = input.parse()?;
        input.parse::<Token![,]>()?;
        let layer_out_r: Type = input.parse()?;
        input.parse::<Token![,]>()?;
        let layer_out_c: Type = input.parse()?;
        input.parse::<Token![>]>()?;
        input.parse::<Token![::]>()?;
        let init = input.parse::<ExprCall>()?;
        input.parse::<Token![;]>()?;
        Ok(LayerMacro {
            name,
            layer_in_c,
            layer_out_c,
            layer_in_r,
            layer_out_r,
            init,
        })
    }
}

struct ModelMacro {
    vec: Vec<LayerMacro>,
}

impl Parse for ModelMacro {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut vec = Vec::new();
        loop {
            match input.parse::<LayerMacro>() {
                Ok(lm) => vec.push(lm),
                Err(_) => return Ok(ModelMacro { vec }),
            }
        }
    }
}

#[proc_macro]
pub fn model(input: TokenStream) -> TokenStream {
    let array = parse_macro_input!(input as ModelMacro);
    let mut layers = Vec::new();
    let mut new_layers = Vec::new();
    let mut forwards = Vec::new();
    let mut backwards = Vec::new();
    let mut updates = Vec::new();
    let mut clears = Vec::new();
    let model_in_c = array.vec[0].layer_in_c.clone();
    let model_in_r = array.vec[0].layer_in_r.clone();
    let model_out_c = array.vec[array.vec.len() - 1].layer_out_c.clone();
    let model_out_r = array.vec[array.vec.len() - 1].layer_out_r.clone();
    for (i, l) in array.vec.iter().enumerate() {
        let varname = format_ident!("layer_{}", i);
        let LayerMacro {
            name,
            layer_in_c,
            layer_out_c,
            layer_in_r,
            layer_out_r,
            init,
        } = l;
        let layer = quote! {
            pub #varname: std::boxed::Box<dyn network::layer::Layer<#layer_in_r,#layer_in_c,#layer_out_r,#layer_out_c>>,
        };
        let new_layer = quote! {
            #varname: std::boxed::Box::new(#name::<#layer_in_r,#layer_in_c,#layer_out_r,#layer_out_c>::#init),
        };
        let forward = quote! {
            let input = self.#varname.forward(input, training);
        };
        let backward = quote! {
            let grads = self.#varname.backward(grads);
        };
        let update = quote! {
            self.#varname.update(lambda);
        };
        let clear = quote! {
            self.#varname.clear();
        };
        layers.push(layer);
        new_layers.push(new_layer);
        forwards.push(forward);
        backwards.insert(0, backward);
        updates.push(update);
        clears.push(clear);
    }
    let loss = quote! {
        std::boxed::Box<dyn network::loss::Loss<#model_out_r,#model_out_c>>
    };
    let expanded = quote! {
        struct Model{
            #(
                #layers
             )*
            pub loss: #loss,
        }
        impl Model{
            pub fn new(loss: #loss) -> Self {
                Model {
                    #(
                        #new_layers
                     )*
                    loss,
                }
            }

            pub fn forward(
                &mut self,
                input: nalgebra::MatrixMN<f64, #model_in_r, #model_in_c>,
                training: bool,
            ) -> nalgebra::MatrixMN<f64, #model_out_r, #model_out_c> {
                #(
                    #forwards
                 )*
                input
            }

            pub fn predict(&mut self, input: nalgebra::MatrixMN<f64, #model_in_r, #model_in_c>) -> nalgebra::MatrixMN<f64, #model_out_r, #model_out_c> {
                self.forward(input, false)
            }

            pub fn fit(
                &mut self,
                n: usize,
                x: &Vec<nalgebra::MatrixMN<f64, #model_in_r, #model_in_c>>,
                y: &Vec<nalgebra::MatrixMN<f64, #model_out_r, #model_out_c>>,
                lambda: f64,
            ) {
                for _ in 0..n {
                    for i in 0..x.len() {
                        let y_ = self.forward(x[i].clone(), true);
                        let loss = self.loss.forward(&y_, &y[i]);
                        println!("loss:{:?}", loss);
                        println!("y_:{}   y:{}", y_[0], y[i][0]);
                        let grads = self.loss.backward();
                        #(#backwards)*
                        #(#updates)*
                        #(#clears)*
                    }
                }
            }
        }
    };
    TokenStream::from(expanded)
}
