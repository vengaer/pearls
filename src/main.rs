extern crate cv;

mod imgen;
use imgen::parser;
use std::env;

use imgen::core::{Image, Window};
use imgen::parser::Show;
use std::error::Error;

#[allow(unused_variables)]
fn main() {
    let args: Vec<String> = env::args().collect();

    let input_idx = args
        .iter()
        .enumerate()
        .find(|&e| e.1.to_string() ==  "-i")
        .map_or(args.len() + 1, |e| e.0 + 1);

    let output_idx = args
        .iter()
        .enumerate()
        .find(|&e| e.1.to_string() == "-o")
        .map_or(args.len() + 1, |e| e.0 + 1);

    let config_idx = args
        .iter()
        .enumerate()
        .find(|&e| e.1.to_string() == "-c")
        .map_or(args.len() + 1, |e| e.0 + 1);

    let config = match config_idx {
        len if len < args.len() => &args[config_idx],
        _ => "config.toml",
    };


    let max_dim = 1000;
    let min_dim = 100;

    let pres = parser::parse(config).map_err(|error| {
        panic!("Fatal: {}", error.description());
    }).unwrap();

    let input = match input_idx {
        len if len < args.len() => &args[input_idx],
        _ => { 
            if pres.input.is_empty() {
                panic!("Fatal: input file not specified neither in config.toml or from command line");
            };
            &pres.input
        },

    };

    let output = match output_idx {
        len if len < args.len() => &args[output_idx],
        _ => &pres.output,

    };

    let subsize  = pres.subsize;
    let ncolors  = pres.ncolors;
    let circsize = pres.circsize;
    let weights  = pres.weights;
    let filter   = pres.filter;
    let exec     = pres.exec;
    let proc     = pres.postproc;
    let show     = pres.show;

    let mut window = Window::new("Window");

    let mut img = Image::from_file(&input).map_err(|error| {
        panic!("Fatal: {}", error.description());
    }).unwrap();

    img.clamp_size(max_dim, min_dim);

    let show_orig = match show {
        Show::Orig => true,
        Show::Both => true,
        _ => false,
    };

    if show_orig {
        window.show(&img).map_err(|error| {
            eprintln!("Warning: could not display original image, {}", error.description());
        }).unwrap();
    }

    let repr = img.reproduce(subsize,
                             ncolors,
                             circsize,
                             weights,
                             filter,
                             exec,
                             proc)
                    .map_err(|error| {
                        panic!("Fatal: {}", error.description());
                    }).unwrap();

    repr.to_file(&output).map_err(|error| {
        panic!("Fatal: {}", error.description());
    }).unwrap();

    let show_repr = match show {
        Show::Res => true,
        Show::Both => true,
        _ => false,
    };

    if show_repr {
        window.show(&repr).map_err(|error| {
            eprintln!("Warning: could not display result, {}", error.description());
        }).unwrap();
    }
}
