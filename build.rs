extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/bindings/cv.cc")
        .compile("cv");
}
