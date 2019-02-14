extern crate cv;

use std::process::Command;

mod imgen;
use imgen::core::{Image, Scalar, Show, WindowFlag, highgui_named_window};
use imgen::color::SampleSpace;

fn main() {
    let img = Image::from_file("/home/vilhelm/repos/rust/pearls/images/test_image.jpg");

    highgui_named_window("Display window", WindowFlag::Normal).unwrap();
    img.show("Display window", 0).expect("Could not display window");

    let _ = Command::new("pause").status();
    let space = SampleSpace::new(100);
    let img = Image::new(23, 23);
    highgui_named_window("Display window", WindowFlag::Normal).unwrap();

    for color in space {
        img.pearl(color, Scalar::new(255,255,255,255));
        img.show("Display window", 0).expect("Could not display window");
        let _ = Command::new("pause").status();
    }
    

}
