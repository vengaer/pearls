extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/bindings.cc")
        .compile("bindings");
}
