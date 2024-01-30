// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ rustc ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
// forbid unused `Result`s etc
#![forbid(unused_must_use)]
// do not disallow identifiers like "max_𝜀"
#![allow(uncommon_codepoints)]
#![warn(rust_2018_idioms)]
#![warn(rust_2021_compatibility)]
//
// ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ Clippy ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
#![warn(clippy::pedantic)]
#![warn(clippy::missing_docs_in_private_items)]
#![allow(clippy::must_use_candidate)]
// instead use expect() to provide rationale
#![warn(clippy::unwrap_used)]
// ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ rustdoc ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
//
// Documentation is primarily for contributors, so allowing links to private items is a must.
#![allow(rustdoc::private_intra_doc_links)]
//
// Broken links are quickly fixed if caught immediately, so just deny them
#![forbid(rustdoc::broken_intra_doc_links)]
//
// Do not allow invalid/empty code blocks. Code blocks are meant to look like this:
//
//   ┆
//   ┆ /// This is a rustdoc comment documenting something. It goes on and on and on …
//   ┆ /// ```
//   ┆ ///   inside.is(code);
//   ┆ /// ```
//   ┆ /// The documentation may, or may not, continue down here.
//   ┆
//
// If the newline is omitted, the code will be neither formatted nor executed:
//
//   ┆
//   ┆ /// This is a rustdoc comment documenting something. It goes on and on and on …```
//   ┆ ///   inside.is(code);
//   ┆ /// ```
//   ┆ /// The documentation may, or may not, continue down here.
//   ┆
//
// Hence the rule.
#![deny(rustdoc::invalid_rust_codeblocks)]
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub use weezl;

#[cfg(test)]
use std::{fs::File, path::PathBuf};

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
  pub use decoded::Decoded;
  pub use error::Error;
  pub use tiff::Tiff;
  pub use types::Rectangle;

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
      UsageError(String),
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
          Error::UsageError(str) => write!(f, "usage error: {str}")
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
    use std::ops::Add;

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

    #[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
    pub struct Rectangle {
      corner: Origin,
      size: Size
    }

    impl Rectangle {
      pub fn corner(&self) -> Origin {
        self.corner
      }

      pub fn size(&self) -> Size {
        self.size
      }

      pub fn corner_x(&self) -> usize { self.corner.x() }
      pub fn corner_y(&self) -> usize { self.corner.y() }
      pub fn width(&self) -> usize { self.size.width() }
      pub fn height(&self) -> usize { self.size.height() }
    }

    #[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
    pub struct Origin((usize, usize));

    #[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
    pub struct Size((usize, usize));

    impl Origin {
      fn x(&self) -> usize { self.0.0 }
      fn y(&self) -> usize { self.0.1 }
    }

    impl Size {
      fn width(&self) -> usize { self.0.0 }
      fn height(&self) -> usize { self.0.1 }
    }

    impl From<(usize, usize)> for Origin {
      fn from(value: (usize, usize)) -> Self {
        Self(value)
      }
    }

    impl From<(usize, usize)> for Size {
      fn from(value: (usize, usize)) -> Self {
        Self(value)
      }
    }

    impl From<(Origin, Size)> for Rectangle {
      fn from(value: (Origin, Size)) -> Self {
        Rectangle {
          corner: value.0,
          size: value.1,
        }
      }
    }

    impl Add<Origin> for Size {
      type Output = Rectangle;

      fn add(self, rhs: Origin) -> Self::Output {
        Rectangle::from((rhs, self))
      }
    }

    impl Add<Size> for Origin {
      type Output = Rectangle;

      fn add(self, rhs: Size) -> Self::Output {
        rhs.add(self)
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
    use std::iter::{FilterMap, FlatMap};
    use std::marker::PhantomData;
    use std::ops::Range;
    use crate::public_api::{band_type::BandType, Rectangle, types::Byte};

    pub struct Decoded {
      bands: Box<[BandType]>,
      data: Box<[u8]>,
      nodata_values: Box<[u8]>,
      pixel_len: Byte,
      rectangle: Rectangle
    }

    impl Decoded {
      pub(crate) fn new<B, N, D>(
        bands: B, nodata_values: N, data: D, rect: Rectangle
      ) -> Self
      where
        B: Into<Box<[BandType]>>,
        N: Into<Box<[u8]>>,
        D: Into<Box<[u8]>>,
      {
        let (width, height, bands, data) = (rect.width(), rect.height(), bands.into(), data.into());
        let pixel_len: usize = bands.iter().map(|b| usize::from(b.width())).sum();
        let nodata_values = nodata_values.into();

        assert_eq!(nodata_values.len(), pixel_len.into());
        assert_eq!(data.len(), width * height * pixel_len);

        Self { bands, data, nodata_values, pixel_len: pixel_len.into(), rectangle: rect }
      }

      pub fn rectangle(&self) -> Rectangle {
        self.rectangle
      }

      pub fn width(&self) -> usize {
        self.rectangle().width()
      }

      pub fn height(&self) -> usize {
        self.rectangle().height()
      }

      pub fn pixels(&self) -> impl Iterator<Item=((usize, usize), Pixel<'_>)> {
          (0..self.height()).flat_map(move |rel_y| {
            (0..self.width()).filter_map(move |rel_x| self.get_pixel((rel_x, rel_y)).map(|p| {
              let abs_x = rel_x + self.rectangle().corner_x();
              let abs_y = rel_y + self.rectangle().corner_y();
              ((abs_x, abs_y), p)
            }))
          })
      }
    }

    #[derive(Debug, PartialOrd, PartialEq)]
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
      nodata_slice: &'d [u8],
    }

    pub trait GetPixel<'a, C> {
      fn get_pixel(&'a self, coord: C) -> Option<Pixel<'a>>;
    }

    impl<'a> GetPixel<'a, usize> for Decoded {
      fn get_pixel(&'a self, coord: usize) -> Option<Pixel<'a>> {
        let len = self.height() * self.width();
        if coord >= len {
          return None;
        }

        let slc_len: usize = self.pixel_len.into();
        let start = coord * slc_len;

        let sample_slice = &self.data[start..start + slc_len];
        assert_eq!(sample_slice.len(), slc_len);

        let nodata_slice = self.nodata_values.as_ref();

        assert_eq!(nodata_slice.len(), sample_slice.len());

        let pix = Pixel { band_types: self.bands.as_ref(), sample_slice, nodata_slice };

        Some(pix)
      }
    }

    impl<'a> GetPixel<'a, (usize, usize)> for Decoded {
      fn get_pixel(&'a self, coord: (usize, usize)) -> Option<Pixel<'a>> {
        if coord.0 >= self.width() || coord.1 >= self.height() {
          None
        } else {
          self.get_pixel(coord.0 + self.width() * coord.1)
        }
      }
    }

    pub trait GetSample<'a> {
      fn get_sample(&'a self, idx: usize) -> Option<Sample>;
    }

    impl<'a> GetSample<'a> for Pixel<'a> {
      fn get_sample(&'a self, idx: usize) -> Option<Sample> {
        if idx >= self.band_types.len() {
          todo!("handle this better perhaps");
        } else {
          let rel_offset =
            self.band_types[..idx].iter().map(|b| usize::from(b.width())).sum::<usize>();
          let band_type = self.band_types[idx];
          let sample_size = usize::from(band_type.width());
          assert!(self.sample_slice.len() >= rel_offset + sample_size);

          let s_slc = &self.sample_slice[rel_offset..rel_offset + sample_size];
          let n_slc = &self.nodata_slice[rel_offset..rel_offset + sample_size];

          if s_slc == n_slc {
            return None;
          }

          match band_type {
            BandType::U8 => {
              let arr = s_slc.try_into().expect("slice len should be correct at this point");
              let value = u8::from_le_bytes(arr);
              Some(Sample::U08(value))
            }
            BandType::F32 => {
              let arr = s_slc.try_into().expect("slice len should be correct at this point");
              let value = f32::from_le_bytes(arr);
              Some(Sample::F32(value))
            }
            BandType::I32 => {
              let arr = s_slc.try_into().expect("slice len should be correct at this point");
              let value = i32::from_le_bytes(arr);
              Some(Sample::I32(value))
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
      I32,
      F32,
    }

    impl BandType {
      pub fn width(&self) -> Byte {
        match self {
          Self::U8 => 1,
          Self::I32 | Self::F32 => 4,
        }
        .into()
      }
    }
  }

  mod tiff {
    use std::io::{Read, Seek};

    use super::{Decoded, Error};
    use crate::{
      decoder::{ifd::Value, Decoder},
      public_api::{band_type::BandType, types::Byte},
      tags::{SampleFormat, Tag},
    };
    use crate::public_api::types::{Origin, Size};

    pub type Result<T> = std::result::Result<T, Error>;

    pub struct Tiff<R> {
      decoder: Decoder<R>,
      band_types: Vec<BandType>,
      nodata_value: String,
    }

    impl<R> Tiff<R> {
      pub fn dimensions(&self) -> (usize, usize) {
        self.decoder.dimensions()
      }

      pub fn pixel_width(&self) -> Byte {
        let samples = usize::from(self.decoder.image().samples);
        let bits_per_sample = usize::from(self.decoder.image().bits_per_sample);

        assert_eq!(
          bits_per_sample % 8,
          0,
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

      fn nodata_value(&self) -> &str {
        self.nodata_value.as_str()
      }
    }

    struct CalculatedFetchRegion {
      img_dim: (usize, usize),
      tile_dim: (usize, usize),
      origin: (usize, usize),
      fetch_rect: (usize, usize),
    }

    impl CalculatedFetchRegion {
      fn new(
        t: VerifiedTileCount, origin: (usize, usize), fetch_rect: (usize, usize),
      ) -> Result<Self> {
        Ok(Self { img_dim: t.img_dim, tile_dim: t.tile_dim, origin, fetch_rect })
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
        let mut decoder = Decoder::new(r)?;
        let band_types = convert_band_types(&decoder)?;

        let nodata_value = match decoder.get_tag(Tag::GdalNodata)? {
          Value::Ascii(s) => s,
          _ => panic!("expected GdalNodata tag to be of type ASCII"),
        };

        Ok(Self { decoder, band_types, nodata_value })
      }

      pub fn read(&mut self, corner: (usize, usize), size: (usize, usize)) -> Result<Decoded> {
        // TODO: convert into errors
        assert!(size.0 > 0);
        assert!(size.1 > 0);

        let (dx, dy) = self.dimensions();
        if corner.0 + size.0 >= dx {
          return Err(Error::UsageError(format!("image X dimension {dx} < requested {}", corner.0+size.0)));
        }
        if corner.1 + size.1 >= dy {
          return Err(Error::UsageError(format!("image Y dimension {dy} < requested {}", corner.1+size.1)));
        }


        let verified = VerifiedTileCount::new(
          self.dimensions(),
          self.decoder.chunk_dimensions(),
          self.decoder.image().chunk_offsets.len(),
        )?;

        let calculated = CalculatedFetchRegion::new(verified, corner, size)?;

        let sample_size = usize::from(self.pixel_width());

        let mut result_vec =
          vec![0u8; calculated.rect_width() * calculated.rect_height() * sample_size];

        let tile_width = calculated.tile_width();
        let tile_height = calculated.tile_height();

        for absyt in calculated.start_y_tile()..=calculated.end_y_tile() {
          for absxt in calculated.start_x_tile()..=calculated.end_x_tile() {
            let tile_src_x_corner = absxt * tile_width;
            let tile_src_first_x = calculated.origin_x().saturating_sub(tile_src_x_corner);
            let l_right_x = ((calculated.origin_x() + calculated.rect_width()) - tile_src_x_corner)
              .min(tile_width);

            let tile_src_y_corner = absyt * tile_height;
            let tile_src_first_y = calculated.origin_y().saturating_sub(tile_src_y_corner);
            let l_lower_y = ((calculated.origin_y() + calculated.rect_height())
              - tile_src_y_corner)
              .min(tile_height);

            let src_first_x = tile_src_x_corner + tile_src_first_x;
            let dst_first_x = src_first_x - calculated.origin_x();
            let dst_right_x = dst_first_x + (l_right_x - tile_src_first_x);

            let src_first_y = tile_src_y_corner + tile_src_first_y;
            let dst_first_y = src_first_y - calculated.origin_y();
            let dst_lower_y = dst_first_y + (l_lower_y - tile_src_first_y);

            let src_range_y = (tile_src_first_y..l_lower_y);
            let dst_range_y = (dst_first_y..dst_lower_y);

            assert_eq!(src_range_y.len(), dst_range_y.len());

            let tile_idx = absxt + absyt * calculated.x_tiles();

            let _d = self.decoder.read_chunk(tile_idx)?;
            let source_octet_slice = _d.as_bytes();

            let (tile_width, tile_height) = self.decoder.chunk_data_dimensions(tile_idx);

            assert_eq!(source_octet_slice.len(), tile_width * tile_height * sample_size);

            for (sy, dy) in src_range_y.clone().zip(dst_range_y.clone()) {
              let sp_start = tile_src_first_x + tile_width * sy;
              let sp_count = l_right_x - tile_src_first_x;
              let sp_offset = sp_start * sample_size;
              let sp_byte_range = sp_offset..sp_offset + (sample_size * sp_count);

              let dp_start = dst_first_x + calculated.rect_width() * dy;
              let dp_count = dst_right_x - dst_first_x;
              let dp_offset = dp_start * sample_size;
              let dp_byte_range = dp_offset..dp_offset + (sample_size * dp_count);

              result_vec[dp_byte_range].copy_from_slice(&source_octet_slice[sp_byte_range]);
            }
          }
        }

        let mut ndv = Vec::with_capacity(usize::from(self.pixel_width()));
        for band in self.band_types() {
          let x = match band {
            BandType::U8 => {
              let value: u8 = self.nodata_value().parse().unwrap(); // TODO
              value.to_le_bytes().to_vec()
            }
            BandType::F32 => {
              let value: f32 = self.nodata_value().parse().unwrap(); // TODO
              value.to_le_bytes().to_vec()
            }
            BandType::I32 => {
              let value: i32 = self.nodata_value().parse().unwrap(); // TODO
              value.to_le_bytes().to_vec()
            }
          };
          ndv.extend(x);
        }

        let size = Size::from(size);
        let origin = Origin::from(corner);
        let rect = size + origin;

        Ok(Decoded::new(self.band_types(), ndv, result_vec, rect))
      }
    }

    fn convert_band_types<R: Read + Seek>(decoder: &Decoder<R>) -> Result<Vec<BandType>> {
      let bits_per_sample = decoder.image().bits_per_sample;
      let sample_formats = decoder.image().sample_format.as_slice();

      let mut result = Vec::with_capacity(sample_formats.len());

      for sf in sample_formats {
        result.push(match (sf, bits_per_sample) {
          (SampleFormat::Uint, 8) => BandType::U8,
          (SampleFormat::Int, 32) => BandType::I32,
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
    use std::{fs::File, path::PathBuf};
    use std::ops::Neg;

    use rand::{thread_rng, Rng};

    use super::*;
    use crate::{
      public_api::{
        band_type::BandType,
        decoded::{GetPixel, GetSample, Sample},
      },
      TEST_IMAGE_DIR,
    };

    #[test]
    fn api_example() {
      let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
      let img_file = File::open(path).expect("image should exist");
      let mut tiff = Tiff::new(img_file).expect("image should be valid");

      let region = tiff.read((10, 10), (5, 5)).unwrap();

      // There are 25 pixels in the read region
      let pixels : Vec<_> = region.pixels().collect();
      assert_eq!(25, pixels.len());

      // 20 of the pixels have a defined sample(0); the others are undefined (nodata)
      let samples : Vec<_> = region.pixels().filter_map(|(_, pixel)| pixel.get_sample(0)).collect();
      assert_eq!(20, samples.len());
    }

    #[test]
    fn e2e_smoke() {
      let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
      let img_file = File::open(path).expect("image should exist");
      let mut tiff = Tiff::new(img_file).expect("image should be valid");

      let dimensions = tiff.dimensions();
      assert_eq!((192, 256), dimensions);

      let mut rng = thread_rng();
      for _ in 0..10_000 {
        let lo_x = rng.gen_range(0..dimensions.0 - 1);
        let left_x = dimensions.0 - lo_x;
        let sx = rng.gen_range(1..left_x);

        let lo_y = rng.gen_range(0..dimensions.1 - 1);
        let left_y = dimensions.1 - lo_y;
        let sy = rng.gen_range(1..left_y);

        eprintln!("{lo_x}, {lo_y} .. {}, {}", lo_x + sx, lo_y + sy);

        let decoded = tiff.read((lo_x, lo_y), (sx, sy)).unwrap();

        for _ in 0..100 {
          let rand_x = rng.gen_range(lo_x..lo_x + sx);
          let rand_y = rng.gen_range(lo_y..lo_y + sy);
          let rel_x = rand_x - lo_x;
          let rel_y = rand_y - lo_y;

          let maybe_sample = decoded.get_pixel((rel_x, rel_y)).get_sample(0);
          if rand_x % 3 == 0 {
            // If pixel X is multiple of 3 expect nodata
            assert!(maybe_sample.is_none());
          } else {
            // else expect sample to be defined and have a specific value
            let val: i32 = maybe_sample
              .expect("sample should be defined")
              .try_into()
              .expect("sample should be i32");

            let expected = 9_000_000 + rand_x as i32 * 1000 + rand_y as i32;
            let expected = if rand_x % 4 == 0 { expected.neg() } else { expected };

            assert_eq!(expected, val);
          }
        }
      }
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
