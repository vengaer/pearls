extern crate cv;

mod imgen;
use imgen::core::{CmpWeights, ExecutionPolicy, Filter, Image, Window};
use imgen::math::Size2u;

#[allow(unused_variables)]
fn main() {
    let max_dim = 1000;
    let min_dim = 100;

    let window = Window::new("Window1").unwrap();
    let mut img = Image::from_file("images/image1.jpg").unwrap();
    
    img.clamp_size(max_dim, min_dim);
    //window.show(&img).expect("Could not show image");
    
    let repr = img.reproduce(Size2u::new(4, 4), 
                             100, 
                             Size2u::new(24, 24), 
                             CmpWeights::new(0.5, 0.5).unwrap(),
                             Filter::Sdev,
                             ExecutionPolicy::Parallellx8).unwrap();
    repr.to_file("images/out.jpg").unwrap();
    //window.show(&repr).expect("Could not show image");
}
