use std::{env};

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-search={}/lib", out_dir);
}
