extern crate cv;

mod imgen;
use imgen::parser;
use imgen::core::{Image, Window};
use imgen::parser::Show;

#[allow(unused_variables)]
fn main() {
    let max_dim = 1000;
    let min_dim = 100;

    let pres = parser::parse("config.toml").unwrap();

    let input    = pres.input;
    let output   = pres.output;
    let subsize  = pres.subsize;
    let ncolors  = pres.ncolors;
    let circsize = pres.circsize;
    let weights  = pres.weights;
    let filter   = pres.filter;
    let exec     = pres.exec;
    let proc     = pres.postproc;
    let show     = pres.show;

    let mut window = Window::new("Window");

    let mut img = Image::from_file(&input).expect("Input image could not be opened");
    img.clamp_size(max_dim, min_dim);

    let show_orig = match show {
        Show::Orig => true,
        Show::Both => true,
        _ => false,
    };

    if show_orig {
        window.show(&img).unwrap();
    }

    let repr = img.reproduce(subsize,
                             ncolors,
                             circsize,
                             weights,
                             filter,
                             exec,
                             proc).unwrap();

    repr.to_file(&output).expect("Could not write reproduced image");

    let show_repr = match show {
        Show::Res => true,
        Show::Both => true,
        _ => false,
    };

    if show_repr {
        window.show(&repr).unwrap();
    }
}
