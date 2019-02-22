extern crate cv;

mod imgen;
use imgen::core::{Image, WindowFlag, highgui_named_window};
use imgen::math::Size2u;

fn main() {
    highgui_named_window("Display window", WindowFlag::Normal).unwrap();

    let mut img = Image::from_file("images/test_image.jpg").unwrap();
    img.show("Display window", 0).expect("Could not display window");
    img.resize(200, 350);
    
    let repr = img.reproduce(Size2u::new(4, 4), 100, Size2u::new(24, 24)).unwrap();
    repr.show("Display window", 0).unwrap();
    repr.to_file("images/out.jpg").unwrap();
}
