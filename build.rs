use std::process::Command;
use std::env;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    Command::new("g++").args(&["src/bindings.cc", "-c", "-fPIC", "-o"])
                       .arg(&format!("{}/bindings.o", out_dir))
                       .status().unwrap();

    Command::new("ar").args(&["crus", "libbindings.a", "bindings.o"])
                      .current_dir(&Path::new(&out_dir))
                      .status().unwrap();

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=bindings");
}
