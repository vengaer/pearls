extern crate cv;

use std::process::Command;

mod imgen;
use imgen::core::{Image, WindowFlag, highgui_named_window};
use imgen::math::Size2u;

fn main() {
    
    let mut img = Image::from_file("/home/vilhelm/repos/rust/pearls/images/test3.jpg").unwrap();
    highgui_named_window("Display window", WindowFlag::Normal).unwrap();
    img.show("Display window", 0).expect("Could not display window");
    
    let repr = img.reproduce(Size2u::new(4, 4), 200, Size2u::new(24, 24)).unwrap();
    repr.show("Display window", 0).expect("Error");
    let _ = Command::new("pause").status();
    //let sub_img = img.subsection(24..28, 0..4).unwrap();
    //sub_img.show("Display window", 0).expect("Could not display window");

    //let means = img.lab_means();

    /*
    let _ = Command::new("pause").status();
    let space = SampleSpace::new(100);
    let img = Image::new(23, 23);
    highgui_named_window("Display window", WindowFlag::Normal).unwrap();

    for color in space {
        img.pearl(color, Scalar::new(255,255,255,255));
        img.show("Display window", 0).expect("Could not display window");
    }
    */
    

}
