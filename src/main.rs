extern crate cv;
use cv::highgui::*;
use cv::imgcodecs::ImageReadMode;
use cv::core::Scalar;

use std::process::Command;

mod imgen;
use imgen::{Image, ColorSpace};

fn main() {
    let img = Image::from_file("/home/vilhelm/repos/rust/pearls/images/test_image.jpg", ImageReadMode::Color);

    highgui_named_window("Display window", WindowFlag::Normal).unwrap();
    img.show("Display window", 0).expect("Could not display window");

    let _ = Command::new("pause").status();
    let space = ColorSpace::new(30);
    //let img = Image::new(500, 500);
    highgui_named_window("Display window", WindowFlag::Normal).unwrap();

    for color in space {
        img.pearl(color, Scalar::new(255,255,255,255));
        img.show("Display window", 0).expect("Could not display window");
        let _ = Command::new("pause").status();
    }
    

}
