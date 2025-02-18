extern crate prost_build;

fn main() {
    prost_build::compile_protos(&["../backend-api/sdi.proto"], &["../backend-api"]).unwrap();
}
