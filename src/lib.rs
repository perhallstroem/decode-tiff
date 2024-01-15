//! Decoding and Encoding of TIFF Images
//!
//! TIFF (Tagged Image File Format) is a versatile image format that supports
//! lossless and lossy compression.
//!
//! # Related Links
//! * <https://web.archive.org/web/20210108073850/https://www.adobe.io/open/standards/TIFF.html> -
//!   The TIFF specification

extern crate weezl;

#[cfg(test)]
use std::{fs::File, path::PathBuf};

pub use self::error::{TiffError, TiffFormatError, TiffResult, TiffUnsupportedError, UsageError};
#[cfg(test)]
use crate::{
  decoder::{ifd::Value, Decoder, DecodingResult},
  tags::Tag,
};

mod bytecast;
pub mod decoder;
mod error;
pub mod tags;

const TEST_IMAGE_DIR: &str = "./tests/images";

#[ignore]
#[test]
fn example() {
  let path = PathBuf::from(TEST_IMAGE_DIR).join("tiled-rect-rgb-u8.tif");
  let img_file = File::open(path).expect("Cannot find test image!");
  let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");
  assert_eq!(decoder.dimensions().expect("Cannot get dimensions"), (490, 367));

  let tile_count = decoder.tile_count().unwrap();
  assert_eq!(48, tile_count);

  for t in 0..tile_count {
    let tile = decoder.read_chunk(t).unwrap();
    let dim = decoder.chunk_data_dimensions(t);
    assert_eq!((32, 128), dim);

    let pixels = decoder.read_chunk(t).unwrap();

    match pixels {
      DecodingResult::U8(bytes) => {
        assert_eq!(4096, bytes.len());
      }
      _ => panic!("not the expected type"),
    }
  }
}

#[ignore]
#[test]
fn example2() {
  let path = PathBuf::from("/Users/perh/Downloads/000_pq4fee8bf1-DSM.tiff");
  let img_file = File::open(path).expect("Cannot find test image!");
  let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");

  let dim = decoder.dimensions().unwrap();
  eprintln!("dimensions: {dim:?}");

  let tiles = decoder.tile_count().unwrap();

  let tag = decoder.get_tag(Tag::GdalNodata).unwrap();
  let tag = match tag {
    Value::Ascii(s) => s,
    _ => panic!("expected GdalNodata tag to be of type ASCII"),
  };

  for t in 0..tiles {
    let tile = decoder.read_chunk(t).unwrap();
    let dim = decoder.chunk_data_dimensions(t);

    match tile {
      DecodingResult::F32(x) => {
        let no_data: f32 = tag.parse().unwrap();

        let c_data = x.iter().filter(|value| **value != no_data).count();
        let c_no = x.len() - c_data;

        eprintln!("T{t:03} decoding result of type F32, width {dim:?} len {} (nodata {c_no}, data {c_data})", x.len());
      }
      _ => panic!("only handling F32 for now"),
    }
    // break;
  }
}

#[ignore]
#[test]
fn example3() {
  let path = PathBuf::from("/Users/perh/Downloads/001_pq36e2a169-ortho.tiff");
  let img_file = File::open(path).expect("Cannot find test image!");
  let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");

  let dim = decoder.dimensions().unwrap();
  eprintln!("dimensions: {dim:?}");

  let tiles = decoder.tile_count().unwrap();

  let tag = decoder.get_tag(Tag::GdalNodata).unwrap();
  let tag = match tag {
    Value::Ascii(s) => s,
    _ => panic!("expected GdalNodata tag to be of type ASCII"),
  };

  for t in 0..tiles {
    let tile = decoder.read_chunk(t).unwrap();
    let dim = decoder.chunk_data_dimensions(t);

    match tile {
      DecodingResult::F32(x) => {
        let no_data: f32 = tag.parse().unwrap();

        let c_data = x.iter().filter(|value| **value != no_data).count();
        let c_no = x.len() - c_data;

        eprintln!("T{t:03} decoding result of type F32, width {dim:?} len {} (nodata {c_no}, data {c_data})", x.len());
      }
      _ => panic!("only handling F32 for now"),
    }
    // break;
  }
}
