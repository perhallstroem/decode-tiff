extern crate tiff;

use std::{fs::File, path::PathBuf};

use tiff::{decoder::Decoder, tags::Tag, ColorType};

const TEST_IMAGE_DIR: &str = "./tests/images/bigtiff";

#[test]
fn test_big_tiff() {
  todo!("Specific test for decoding BigTIFFs may be missing")
}
