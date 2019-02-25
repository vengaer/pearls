extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/ffis.cc")
        .compile("ffis");
}
