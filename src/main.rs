extern crate cv;

use std::process::Command;

mod imgen;
use imgen::core::{Image, WindowFlag, highgui_named_window};
use imgen::math::Size2u;

fn main() {
    highgui_named_window("Display window", WindowFlag::Normal).unwrap();

    let mut img = Image::from_file("images/test_image.jpg").unwrap();
    img.show("Display window", 0).expect("Could not display window");
    
    let repr = img.reproduce(Size2u::new(4, 4), 100, Size2u::new(24, 24)).unwrap();
    repr.show("Display window", 0).unwrap();
    let _ = Command::new("pause").status();
}
