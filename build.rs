extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/ffs.cc")
        .compile("ffs");
}
