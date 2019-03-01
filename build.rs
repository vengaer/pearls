extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/native/ffs.cc")
        .compile("ffs");
}
