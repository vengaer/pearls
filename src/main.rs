extern crate cv;

mod imgen;
use imgen::core::{Image, Window};
use imgen::math::Size2u;

fn main() {
    let max_dim = 1000;
    let min_dim = 100;

    let window = Window::new("Window1").unwrap();
    let mut img = Image::from_file("images/image1.jpg").unwrap();
    
    img.clamp_size(max_dim, min_dim);
    window.show(&img).expect("Could not show image");
    
    let repr = img.reproduce(Size2u::new(4, 4), 100, Size2u::new(24, 24)).unwrap();
    repr.to_file("images/out.jpg").unwrap();
    window.show(&repr).expect("Could not show image");
}
