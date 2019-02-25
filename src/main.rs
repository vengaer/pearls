extern crate cv;

mod imgen;
use imgen::core::{Image, Window};
use imgen::math::Size2u;
use std::cmp;

fn main() {

    let window = Window::new("Window1");
    let mut img = Image::from_file("images/image3.jpg").unwrap();

    let max_dim = 1000;
    let min_dim = 100;
    
    if img.data.cols > max_dim || 
       img.data.rows > max_dim {

        let factor = max_dim as f32 / cmp::max(img.data.cols, img.data.rows) as f32;
        println!("Resizing image by factor {}", factor);

        img.resize_by(factor, factor);
    }
    else if img.data.cols < min_dim ||
            img.data.rows < min_dim {
        let factor = min_dim as f32 / cmp::min(img.data.cols, img.data.rows) as f32;
        println!("Resizing image by factor {}", factor);

        img.resize_by(factor, factor);
    }

    window.show(&img).expect("Could not show image");
    
    let repr = img.reproduce(Size2u::new(4, 4), 100, Size2u::new(24, 24)).unwrap();
    repr.to_file("images/out.jpg").unwrap();
    window.show(&repr).expect("Could not show image");
}
