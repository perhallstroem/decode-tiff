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
use std::{
  io,
  io::{Read, Seek},
};

pub use self::error::{TiffError, TiffFormatError, TiffResult, TiffUnsupportedError};
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

mod public_api {
  use std::{
    fmt::{Display, Formatter},
    io::{Read, Seek},
  };

  pub use decoded::Decoded;
  pub use error::Error;
  pub use tiff::Tiff;

  mod error {
    use std::fmt::{Display, Formatter};

    use crate::TiffError;

    type BoxedError = Box<dyn std::error::Error>;

    #[derive(Debug)]
    pub enum Error {
      IoError(std::io::Error),
      InvalidFormat(BoxedError),
      UnsupportedFeature(BoxedError),
      Internal(BoxedError),
      UnsupportedBandType { format: String, bit_width: u8 },
    }

    impl Display for Error {
      fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
          Error::IoError(e) => write!(f, "I/O error: {e}"),
          Error::InvalidFormat(e) => write!(f, "invalid TIFF format: {e}"),
          Error::UnsupportedFeature(e) => write!(f, "unsupported TIFF feature: {e}"),
          Error::Internal(e) => write!(f, "internal error: {e}"),
          Error::UnsupportedBandType { format, bit_width } => {
            write!(f, "unsupported {bit_width}-bit band type {format}")
          }
        }
      }
    }

    impl std::error::Error for Error {}

    impl From<TiffError> for Error {
      fn from(value: TiffError) -> Self {
        match value {
          TiffError::FormatError(e) => Self::InvalidFormat(Box::new(e)),
          TiffError::UnsupportedError(e) => Self::UnsupportedFeature(Box::new(e)),
          TiffError::IoError(e) => Self::IoError(e),
          _ => Self::Internal(Box::new(value)),
        }
      }
    }
  }

  pub mod types {
    use std::fmt::{Display, Formatter};

    macro_rules! impl_numeric_newtype {
      ($name:ident, $t:ty ) => {
        #[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
        pub struct $name($t);

        impl Display for $name {
          fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
          }
        }

        impl From<usize> for $name {
          fn from(value: usize) -> Self {
            $name(value)
          }
        }

        impl From<$name> for usize {
          fn from(value: $name) -> usize {
            value.0
          }
        }

        impl $name {
          pub fn checked_sub(self, other: Self) -> Option<Self> {
            self.0.checked_sub(other.0).map(Self)
          }

          pub fn checked_add(self, other: Self) -> Option<Self> {
            self.0.checked_add(other.0).map(Self)
          }

          pub fn checked_mul(self, other: Self) -> Option<Self> {
            self.0.checked_mul(other.0).map(Self)
          }
        }
      };
    }

    impl_numeric_newtype!(Bit, usize);
    impl_numeric_newtype!(Byte, usize);

    impl Byte {
      fn to_bits(self) -> Bit {
        self.0.checked_mul(8).map(Bit).expect("multiplication should not overflow")
      }
    }

    #[cfg(test)]
    mod tests {
      use crate::public_api::types::{Bit, Byte};

      #[test]
      fn bytes_to_bits() {
        assert_eq!(Bit(88), Byte(11).to_bits());
      }
    }
  }

  mod decoded {
    use crate::public_api::{band_type::BandType, types::Byte};

    pub struct Decoded {
      bands: Box<[BandType]>,
      data: Box<[u8]>,
      pixel_len: Byte,
      width: usize,
      height: usize,
    }

    // TODO: assertions that
    // data.len() = width * height * pixel_len
    // bands...width()...sum() = pixel_len
    // ...

    impl Decoded {
      pub(crate) fn new<B, D, W, H>(bands: B, data: D, width: W, height: H) -> Self
      where
        B: Into<Box<[BandType]>>,
        D: Into<Box<[u8]>>,
        W: Into<usize>,
        H: Into<usize>,
      {
        let (width, height, bands, data) = (width.into(), height.into(), bands.into(), data.into());
        let pixel_len: usize = bands.iter().map(|b| usize::from(b.width())).sum();

        assert_eq!(data.len(), width * height * pixel_len);

        Self { bands, data, pixel_len: pixel_len.into(), width, height }
      }

      pub fn width(&self) -> usize {
        self.width
      }

      pub fn height(&self) -> usize {
        self.height
      }
    }

    #[derive(Debug)]
    pub enum Sample {
      U08(u8),
      I32(i32),
      F32(f32),
    }

    impl TryFrom<Sample> for u8 {
      type Error = ();

      fn try_from(value: Sample) -> Result<Self, Self::Error> {
        match value {
          Sample::U08(v) => Ok(v),
          _ => Err(()),
        }
      }
    }

    impl TryFrom<Sample> for f32 {
      type Error = ();

      fn try_from(value: Sample) -> Result<Self, Self::Error> {
        match value {
          Sample::F32(v) => Ok(v),
          _ => Err(()),
        }
      }
    }

    impl TryFrom<Sample> for i32 {
      type Error = ();

      fn try_from(value: Sample) -> Result<Self, Self::Error> {
        match value {
          Sample::I32(v) => Ok(v),
          _ => Err(()),
        }
      }
    }

    #[derive(Debug)]
    pub struct Pixel<'d> {
      band_types: &'d [BandType],
      sample_slice: &'d [u8],
    }

    impl<'d> Pixel<'d> {
      fn sample(&self, idx: usize) -> Sample {
        todo!()
      }
    }

    pub trait GetPixel<'a, C> {
      fn get_pixel(&'a self, coord: C) -> Option<Pixel<'a>>;
    }

    impl<'a> GetPixel<'a, usize> for Decoded {
      fn get_pixel(&'a self, coord: usize) -> Option<Pixel<'a>> {
        let len = self.height * self.width;
        if coord >= len {
          return None;
        }

        let slc_len: usize = self.pixel_len.into();
        let start = coord * slc_len;

        let sample_slice = &self.data[start..start + slc_len];
        assert_eq!(sample_slice.len(), slc_len);

        let pix = Pixel { band_types: self.bands.as_ref(), sample_slice };

        Some(pix)
      }
    }

    impl<'a> GetPixel<'a, (usize, usize)> for Decoded {
      fn get_pixel(&'a self, coord: (usize, usize)) -> Option<Pixel<'a>> {
        if coord.0 >= self.width || coord.1 >= self.height {
          None
        } else {
          self.get_pixel(coord.0 + self.width * coord.1)
        }
      }
    }

    pub trait GetSample<'a> {
      fn get_sample(&'a self, idx: usize) -> Option<Sample>;
    }

    impl<'a> GetSample<'a> for Pixel<'a> {
      fn get_sample(&'a self, idx: usize) -> Option<Sample> {
        if idx >= self.band_types.len() {
          None
        } else {
          let rel_offset =
            self.band_types[..idx].iter().map(|b| usize::from(b.width())).sum::<usize>();
          let band_type = self.band_types[idx];
          let sample_size = usize::from(band_type.width());
          assert!(self.sample_slice.len() >= rel_offset + sample_size);

          match band_type {
            BandType::U8 => Some(Sample::U08(self.sample_slice[rel_offset])),
            BandType::F32 => {
              let slc = self.sample_slice[rel_offset..rel_offset + sample_size].try_into().unwrap();
              let value = f32::from_ne_bytes(slc);
              Some(Sample::F32(value))
            }
          }
        }
      }
    }

    impl<'a> GetSample<'a> for Option<Pixel<'a>> {
      fn get_sample(&'a self, idx: usize) -> Option<Sample> {
        self.as_ref().and_then(|some| GetSample::get_sample(some, idx))
      }
    }
  }

  mod band_type {
    use crate::public_api::types::Byte;

    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    pub enum BandType {
      U8,
      F32,
    }

    impl BandType {
      pub fn width(&self) -> Byte {
        match self {
          BandType::U8 => 1,
          BandType::F32 => 4,
        }
        .into()
      }
    }
  }

  mod tiff {
    use std::{
      arch::aarch64::vaba_s16,
      io::{Read, Seek},
    };

    use super::{Decoded, Error};
    use crate::{
      decoder::Decoder,
      public_api::{band_type::BandType, types::Byte},
      tags::SampleFormat,
    };

    pub type Result<T> = std::result::Result<T, Error>;

    pub struct Tiff<R> {
      decoder: Decoder<R>,
      band_types: Vec<BandType>,
    }

    impl<R> Tiff<R> {
      pub fn dimensions(&self) -> (usize, usize) {
        self.decoder.dimensions()
      }

      pub fn pixel_width(&self) -> Byte {
        let samples = usize::from(self.decoder.image().samples);
        let bits_per_sample = usize::from(self.decoder.image().bits_per_sample);

        assert!(
          bits_per_sample % 8 == 0,
          "bits per sample is {bits_per_sample}, but can only handle even octets"
        );

        samples
          .checked_mul(bits_per_sample)
          .map(|total_bits| Byte::from(total_bits / 8))
          .expect("usize should be wide enough to contain pixel width")
      }

      pub fn band_types(&self) -> &[BandType] {
        self.band_types.as_slice()
      }
    }

    struct CalculatedFetchRegion {
      img_dim: (usize, usize),
      tile_dim: (usize, usize),
      origin: (usize, usize),
      fetch_rect: (usize, usize),
    }

    impl CalculatedFetchRegion {
      fn new(t: VerifiedTileCount, origin: (u32, u32), fetch_rect: (u32, u32)) -> Result<Self> {
        let origin_x = usize::try_from(origin.0).expect("u32 should fit in usize");
        let origin_y = usize::try_from(origin.1).expect("u32 should fit in usize");
        let len_x = usize::try_from(fetch_rect.0).expect("u32 should fit in usize");
        let len_y = usize::try_from(fetch_rect.1).expect("u32 should fit in usize");

        Ok(Self {
          img_dim: t.img_dim,
          tile_dim: t.tile_dim,
          origin: (origin_x, origin_y),
          fetch_rect: (len_x, len_y),
        })
      }

      fn fetch_rect(&self) -> (usize, usize) {
        self.fetch_rect
      }

      fn rect_width(&self) -> usize {
        self.fetch_rect.0
      }

      fn rect_height(&self) -> usize {
        self.fetch_rect.1
      }

      fn origin(&self) -> (usize, usize) {
        self.origin
      }

      fn origin_x(&self) -> usize {
        self.origin.0
      }

      fn origin_y(&self) -> usize {
        self.origin.1
      }

      fn tile_width(&self) -> usize {
        self.tile_dim.0
      }

      fn tile_height(&self) -> usize {
        self.tile_dim.1
      }

      fn start_x_tile(&self) -> usize {
        self.origin.0 / self.tile_dim.0
      }

      fn start_y_tile(&self) -> usize {
        self.origin.1 / self.tile_dim.1
      }

      fn end_x_tile(&self) -> usize {
        (self.origin.0 + self.fetch_rect.0) / self.tile_dim.0
      }

      fn end_y_tile(&self) -> usize {
        (self.origin.1 + self.fetch_rect.1) / self.tile_dim.1
      }

      fn x_tiles(&self) -> usize {
        self.img_dim.0.div_ceil(self.tile_dim.0)
      }

      fn y_tiles(&self) -> usize {
        self.img_dim.1.div_ceil(self.tile_dim.1)
      }
    }

    struct VerifiedTileCount {
      img_dim: (usize, usize),
      tile_dim: (usize, usize),
    }

    impl VerifiedTileCount {
      fn new(
        img_dim: (usize, usize), tile_dim: (usize, usize), actual_tile_count: usize,
      ) -> Result<Self> {
        let (img_width, img_height) = img_dim;
        let (tile_width, tile_height) = tile_dim;

        let expected_tiles = img_width
          .div_ceil(tile_width)
          .checked_mul(img_height.div_ceil(tile_height))
          .expect("pixel count calculation cannot reasonably overflow");
        if expected_tiles != actual_tile_count {
          return Err(
            Error::Internal(
              format!("given image dimension expected there to be {expected_tiles}, but found {actual_tile_count}").into()
            )
          );
        }

        Ok(Self { img_dim, tile_dim })
      }
    }

    impl<R> Tiff<R>
    where
      R: Read + Seek,
    {
      pub fn new(r: R) -> Result<Self> {
        let decoder = Decoder::new(r)?;
        let band_types = convert_band_types(&decoder)?;
        Ok(Self { decoder, band_types })
      }

      pub fn read(&mut self, corner: (u32, u32), size: (u32, u32)) -> Result<Decoded> {
        // TODO: convert into errors
        assert!(size.0 > 0);
        assert!(size.1 > 0);

        let verified = VerifiedTileCount::new(
          self.dimensions(),
          self.decoder.chunk_dimensions(),
          self.decoder.image().chunk_offsets.len(),
        )?;

        let calculated = CalculatedFetchRegion::new(verified, corner, size)?;

        let sample_size = usize::from(self.pixel_width());

        eprintln!("Sample size is {sample_size}");
        let mut result_vec =
          vec![0u8; calculated.rect_width() * calculated.rect_height() * sample_size];
        eprintln!("Result buffer size is {}", result_vec.len());

        let tile_width = calculated.tile_width();
        let tile_height = calculated.tile_height();

        for absyt in calculated.start_y_tile()..=calculated.end_y_tile() {
          for absxt in calculated.start_x_tile()..=calculated.end_x_tile() {
            let tile_src_x_corner = absxt * tile_width;
            let tile_src_first_x = calculated.origin_x().saturating_sub(tile_src_x_corner);
            let l_right_x = ((calculated.origin_x() + calculated.rect_width()) - tile_src_x_corner)
              .min(tile_width);

            // eprintln!("Tile X corner is at {}; read range is {}..{}", tile_src_x_corner,
            // tile_src_first_x, l_right_x);

            let tile_src_y_corner = absyt * tile_height;
            let tile_src_first_y = calculated.origin_y().saturating_sub(tile_src_y_corner);
            let l_lower_y = ((calculated.origin_y() + calculated.rect_height())
              - tile_src_y_corner)
              .min(tile_height);

            // eprintln!("Tile Y corner is at {}; read range is {}..{}", tile_src_y_corner,
            // tile_src_first_y, l_lower_y);

            let src_first_x = tile_src_x_corner + tile_src_first_x;
            let dst_first_x = src_first_x - calculated.origin_x();
            let dst_right_x = dst_first_x + (l_right_x - tile_src_first_x);

            let src_first_y = tile_src_y_corner + tile_src_first_y;
            let dst_first_y = src_first_y - calculated.origin_y();
            let dst_lower_y = dst_first_y + (l_lower_y - tile_src_first_y);

            // eprintln!("First 'source X': {src_first_x}");
            // eprintln!("First 'dest X': {dst_first_x}");
            // eprintln!("Right 'dest X': {dst_right_x}");

            let src_range_x = (tile_src_first_x..l_right_x);
            let dst_range_x = (dst_first_x..dst_right_x);
            let src_range_y = (tile_src_first_y..l_lower_y);
            let dst_range_y = (dst_first_y..dst_lower_y);

            assert_eq!(src_range_x.len(), dst_range_x.len());
            assert_eq!(src_range_y.len(), dst_range_y.len());

            // eprintln!("Reading ({tile_src_first_x}, {tile_src_first_y})..({l_right_x},
            // {l_lower_y}) from tile ({absxt}, {absyt}) into result region ({dst_first_x},
            // {dst_first_y})..({dst_right_x}, {dst_lower_y})");

            let tile_idx = absxt + absyt * calculated.x_tiles();

            let toile_data = self.decoder.read_chunk(tile_idx)?;
            let bufbuf = toile_data.as_bytes();

            let (tile_width, tile_height) = self.decoder.chunk_data_dimensions(tile_idx);

            assert_eq!(bufbuf.len(), tile_width * tile_height * sample_size);

            // eprintln!("READ BUFFER OF {} bytes", bufbuf.len());

            for (sy, dy) in src_range_y.clone().zip(dst_range_y.clone()) {
              for (sx, dx) in src_range_x.clone().zip(dst_range_x.clone()) {
                let sp = sx + tile_width * sy;
                let dp = dx + calculated.rect_width() * dy;
                // eprintln!("{sx},{sy} ({sp}) => {dx},{dy} ({dp})");

                let sp_offset = sample_size * sp;
                let dp_offset = sample_size * dp;
                let sp_range = sp_offset..sp_offset + sample_size;
                let dp_range = dp_offset..dp_offset + sample_size;

                assert_eq!(sp_range.len(), dp_range.len());

                // TODO: because handling of "nodata" values is not yet handled etc, fixing this
                // seems like a lower priority   Note that this seems a bit
                // wasteful, because it copies adjacent slices:     result_vec[0..
                // 4].copy_from_slice(&bufbuf[261112..261116])     result_vec[4..8].
                // copy_from_slice(&bufbuf[261116..261120])     …
                result_vec[dp_range].copy_from_slice(&bufbuf[sp_range]);
              }
            }
          }
        }

        // let bands = self.band_types().to_vec().into_boxed_slice();
        // let data = result_vec.into_boxed_slice();
        let width = calculated.rect_width();
        let height = calculated.rect_height();

        Ok(Decoded::new(self.band_types(), result_vec, width, height))
      }
    }

    fn convert_band_types<R: Read + Seek>(decoder: &Decoder<R>) -> Result<Vec<BandType>> {
      let bits_per_sample = decoder.image().bits_per_sample;
      let sample_formats = decoder.image().sample_format.as_slice();

      let mut result = Vec::with_capacity(sample_formats.len());

      for sf in sample_formats {
        result.push(match (sf, bits_per_sample) {
          (SampleFormat::Uint, 8) => BandType::U8,
          (SampleFormat::IEEEFP, 32) => BandType::F32,
          _ => {
            return Err(Error::UnsupportedBandType {
              format: format!("{:?}", sf),
              bit_width: bits_per_sample,
            });
          }
        });
      }

      Ok(result)
    }

    #[cfg(test)]
    mod tests {
      use crate::public_api::tiff::{CalculatedFetchRegion, VerifiedTileCount};

      #[test]
      fn verified_tile_count() {
        let img_dim = (8922, 5907);
        let tile_dim = (256, 256);
        let vtc = VerifiedTileCount::new(img_dim, tile_dim, 840).unwrap();
        assert_eq!(tile_dim, vtc.tile_dim);
        assert_eq!(img_dim, vtc.img_dim);
      }

      #[test]
      fn calculated_fetch_region() {
        let img_dim = (8922, 5907);
        let tile_dim = (256, 256);
        let vtc = VerifiedTileCount { img_dim, tile_dim };
        let cfr = CalculatedFetchRegion::new(vtc, (128, 128), (256, 256)).unwrap();

        assert_eq!(0, cfr.start_x_tile());
        assert_eq!(0, cfr.start_y_tile());

        assert_eq!(1, cfr.end_x_tile());
        assert_eq!(1, cfr.end_y_tile());
      }
    }
  }

  #[cfg(test)]
  mod tests {
    use std::{fs::File, time::Instant};

    use super::*;
    use crate::public_api::decoded::{GetPixel, GetSample};

    #[test]
    fn api_example() {
      let path = "/Users/perh/Downloads/000_pq4fee8bf1-DSM.tiff";
      let img_file = File::open(path).expect("image should exist");
      let mut tiff = Tiff::new(img_file).expect("image should be valid");

      let dimensions = tiff.dimensions();

      assert_eq!((8922, 5907), dimensions);

      let band_types = tiff.band_types();

      eprintln!("band_types: {band_types:?}");

      let start = Instant::now();
      let sloice = tiff.read((0, 0), (8922, 5907)).unwrap();
      let dur = Instant::now() - start;

      eprintln!("It took {dur:?} to read {} pixlurs", sloice.width() * sloice.height());

      eprintln!("slice dimensions: height={}, width={}", sloice.height(), sloice.width());

      let pixel = sloice.get_pixel((1, 1));
      eprintln!("pixel: {pixel:?}");

      // let pixel = sloice.get_pixel((7, 3));
      // let pixel = sloice.get_pixel(899);

      let value = pixel.get_sample(0).expect("I am sure there is one");
      eprintln!("value: {value:?}");

      todo!("this is where it continues")

      // let value: i32 = value.try_into().expect("know what I am doing");

      // let tile_dim = decoder.tile_dimensions();
    }
  }
}

#[ignore]
#[test]
fn example() {
  let path = PathBuf::from(TEST_IMAGE_DIR).join("tiled-rect-rgb-u8.tif");
  let img_file = File::open(path).expect("Cannot find test image!");
  let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");
  assert_eq!(decoder.dimensions(), (490, 367));

  let tile_count = decoder.tile_count();
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

  let dim = decoder.dimensions();
  eprintln!("dimensions: {dim:?}");

  let tiles = decoder.tile_count();

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

  let dim = decoder.dimensions();
  eprintln!("dimensions: {dim:?}");

  let tiles = decoder.tile_count();

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
