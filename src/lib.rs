#![allow(clippy::all)]

pub mod decoder;
mod tags;

mod bytecast;
mod error;

use error::{TiffError, TiffFormatError, TiffResult, TiffUnsupportedError};
pub use public_api::{Error, GetPixel, GetSample, Sample, Tiff};

pub const TEST_IMAGE_DIR: &str = "./tests/images";

mod public_api {

  //! The limited public API, wrapping the less-ergonomic [`decoder`][crate::decoder]

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //
  // ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ rustc ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
  // forbid unused `Result`s etc
  #![forbid(unused_must_use)]
  //
  // ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ Clippy ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
  #![warn(clippy::pedantic)]
  #![warn(clippy::missing_docs_in_private_items)]
  #![allow(clippy::must_use_candidate)]
  // instead use expect() to provide rationale
  #![warn(clippy::unwrap_used)]
  // ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ rustdoc ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
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
  pub use decoded::{Decoded, GetPixel, GetSample, Sample};
  pub use error::Error;
  pub use tiff::Tiff;
  pub use types::Rectangle;

  mod error {
    //! The [`Error`] type and its implementations

    use std::fmt::{Display, Formatter};

    use crate::error::TiffError;

    /// Convenience alias for arbitrary boxed errors
    type BoxedError = Box<dyn std::error::Error>;

    /// The error type returned from this [crate]
    #[derive(Debug)]
    pub enum Error {
      /// An [I/O error][std::io::Error] occurred while reading from the source
      IoError(std::io::Error),
      /// The source TIFF had an invalid or incompatible format
      InvalidFormat(BoxedError),
      /// The source TIFF used a feature that is unsupported by this crate
      UnsupportedFeature(BoxedError),
      /// The source TIFF contained a band of an unsupported type/format and/or width
      UnsupportedBandType { format: String, bit_width: u8 },
      /// The library was used incorrectly; for example, an attempt was made to read a 0×0-pixel
      /// region
      UsageError(String),
      /// An internal (rare/unexpected) error occurred; this may indicate an error in the wrapped
      /// TIFF library
      Internal(BoxedError),
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
          Error::UsageError(str) => write!(f, "usage error: {str}"),
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
    //! Simple common types used throughout the crate

    use std::{
      fmt::{Display, Formatter},
      ops::Add,
    };

    /// Generates a simple numeric newtype
    ///
    /// This macro generates a simple named numeric newtype, with some rudimentary basic features.
    macro_rules! impl_numeric_newtype {
      ($name:ident, $t:ty ) => {
        /// Generated numeric newtype
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
          /// Checked integer subtraction, with standard-library semantics
          pub fn checked_sub(self, other: Self) -> Option<Self> {
            self.0.checked_sub(other.0).map(Self)
          }

          /// Checked integer addition, with standard-library semantics
          pub fn checked_add(self, other: Self) -> Option<Self> {
            self.0.checked_add(other.0).map(Self)
          }

          /// Checked integer multiplication, with standard-library semantics
          pub fn checked_mul(self, other: Self) -> Option<Self> {
            self.0.checked_mul(other.0).map(Self)
          }
        }
      };
    }

    impl_numeric_newtype!(Bit, usize);
    impl_numeric_newtype!(Byte, usize);

    /// Describes a rectangle of a known extent
    #[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
    pub struct Rectangle {
      /// The corner (low-x, low-y) coordinate, or origin
      corner: Origin,
      /// The size, in number of pixels, of the rectangle
      size: Size,
    }

    #[allow(dead_code)]
    impl Rectangle {
      /// Returns the [`Origin`] of `self`
      pub fn corner(&self) -> Origin {
        self.corner
      }

      /// Returns the [`Size`] of self
      pub fn size(&self) -> Size {
        self.size
      }

      /// Returns the X corner coordinate
      pub fn corner_x(&self) -> usize {
        self.corner.x()
      }

      /// Returns the Y corner coordinate
      pub fn corner_y(&self) -> usize {
        self.corner.y()
      }

      /// Returns the width of `self`
      pub fn width(&self) -> usize {
        self.size.width()
      }

      /// Returns the height of `self`
      pub fn height(&self) -> usize {
        self.size.height()
      }
    }

    /// Newtype describing an origin
    ///
    /// This newtype is used to avoid confusing `(usize, usize)` tuples of different purposes
    #[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
    pub struct Origin((usize, usize));

    /// Newtype describing a size, in pixels
    ///
    /// This newtype is used to avoid confusing `(usize, usize)` tuples of different purposes
    #[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
    pub struct Size((usize, usize));

    impl Origin {
      /// Returns `self`'s X coordinate
      fn x(&self) -> usize {
        self.0 .0
      }

      /// Returns `self`'s Y coordinate
      fn y(&self) -> usize {
        self.0 .1
      }
    }

    impl Size {
      /// Returns the width of `self`
      fn width(&self) -> usize {
        self.0 .0
      }

      /// Returns the height of `self`
      fn height(&self) -> usize {
        self.0 .1
      }
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
        Rectangle { corner: value.0, size: value.1 }
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
  }

  mod decoded {
    //! Types and implementations related to fetching [regions of data][Decoded], [pixels][Pixel]
    //! from those regions and [samples][Sample] from those pixels
    //!
    //! # Example: reading a single sample from a single pixel
    //! ```
    //! use std::{fs::File, path::PathBuf};
    //!
    //! use tiff::{GetPixel, GetSample, Tiff, TEST_IMAGE_DIR};
    //!
    //! let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
    //! let img_file = File::open(path).expect("image should exist");
    //! let mut tiff = Tiff::new(img_file).expect("image should be valid");
    //!
    //! let region = tiff.read((0, 0), (2, 2)).expect("unable to read region");
    //!
    //! let sample = region.get_pixel((1, 1)).get_sample(0).unwrap();
    //! let expected: i32 = sample.try_into().unwrap();
    //! assert_eq!(expected, 9001001i32);
    //! ```

    use crate::public_api::{band_type::BandType, types::Byte, Rectangle};

    /// Defines a [`get_pixel`][GetPixel::get_pixel] method for accessing a pixel by coordinate
    pub trait GetPixel<'a, C> {
      /// Returns the [`Pixel`] at the specified coordinate `coord`, if one exists
      ///
      /// In the fashion of [`Vec::get`], implementations of this method should return `None` if the
      /// specified `coord` is out of bounds
      fn get_pixel(&'a self, coord: C) -> Option<Pixel<'a>>;
    }

    /// Defines a [`get_sample`][GetSample::get_sample] method for access a sample by index
    pub trait GetSample<'a> {
      /// Returns the [`Sample`] at the specified index `idx`, if one exists
      ///
      /// This module implements [`GetSample`] for both [`Pixel`] and [`Option<Pixel>`], allowing
      /// chaining of calls without `?`.
      ///
      /// # Example: implementation for [`Pixel`]
      /// ```
      /// # use std::{fs::File, path::PathBuf};
      /// # use tiff::{GetPixel, Tiff, TEST_IMAGE_DIR, GetSample};
      /// # let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
      /// # let img_file = File::open(path).expect("image should exist");
      /// # let mut tiff = Tiff::new(img_file).expect("image should be valid");
      /// # let some_region = tiff.read((0, 0), (2, 2)).expect("unable to read region");
      ///
      /// let pixel = some_region.get_pixel((1, 1)).unwrap();
      /// assert!(pixel.get_sample(0).is_some());
      /// ```
      ///
      /// # Example: implementation for [`Option<Pixel>`]
      /// ```
      /// # use std::{fs::File, path::PathBuf};
      /// # use tiff::{GetPixel, Tiff, TEST_IMAGE_DIR, GetSample};
      /// # let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
      /// # let img_file = File::open(path).expect("image should exist");
      /// # let mut tiff = Tiff::new(img_file).expect("image should be valid");
      /// # let some_region = tiff.read((0, 0), (2, 2)).expect("unable to read region");
      ///
      /// let pixel = some_region.get_pixel((1, 1));
      /// assert!(pixel.get_sample(0).is_some());
      /// ```
      fn get_sample(&'a self, idx: usize) -> Option<Sample>;
    }

    /// A sample, i.e. a "band value", from a pixel
    #[derive(Debug, PartialOrd, PartialEq)]
    pub enum Sample {
      /// The value, of type `u8`
      U08(u8),
      /// The value, of type `i32`
      I32(i32),
      /// The value, of type `f32`
      F32(f32),
    }

    /// A decoded region read from a TIFF
    pub struct Decoded {
      /// The array of [band types][BandType] the TIFF contains
      bands: Box<[BandType]>,
      /// The raw pixel data read from the TIFF
      data: Box<[u8]>,
      /// The parsed per-band nodata values
      ///
      /// An important note about peculiarities: by convention a TIFF has a single `GdalNodata`
      /// tag, containing the nodata value **as an ASCII string**. Because a TIFF's band types may
      /// differ, this array has the same length as a single pixel (i.e. its length is equal to the
      /// sum of the bands' widths), and contains, without padding, the string parsed into the
      /// respective data type. For example, if the nodata value is "0" and the TIFF contains a
      /// single unsigned 8-bit integer band followed by a single unsigned 16-bit integer band, the
      /// content of this array will be `[0, 0, 0]`: the first byte is "0" parsed into a `u8` and
      /// the following two bytes "0" parsed into a u16.
      nodata_values: Box<[u8]>,
      /// The length of a pixel, equal to the sum of the bands' widths, and equal to the length of
      /// the `nodata_values` array
      pixel_len: Byte,
      /// The [`Rectangle`] of the source TIFF this decoded region covers
      rectangle: Rectangle,
    }

    /// A single pixel read from a [decoded region][Decoded]
    #[derive(Debug)]
    pub struct Pixel<'d> {
      /// Reference to the array of band types
      band_types: &'d [BandType],
      /// The raw byte slice containing the pixel's samples
      sample_slice: &'d [u8],
      /// The parsed nodata values (see [`Decoded`])
      nodata_slice: &'d [u8],
    }

    impl Decoded {
      /// Instantiates a new [`Decoded`] with the supplied fields
      ///
      /// # Panics
      /// This crate-internal method may panic if the supplied values are not correct with respect
      /// to each other; for example, the supplied sample data must be of exactly the right length
      /// with respect to the supplied [`rect`][Rectangle].
      pub(crate) fn new<B, N, D>(bands: B, nodata_values: N, data: D, rect: Rectangle) -> Self
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

        Self {
          bands,
          data,
          nodata_values,
          pixel_len: pixel_len.into(),
          rectangle: rect,
        }
      }

      /// Returns the [`Rectangle`] whence this decoded region was read from the TIFF source
      ///
      /// # Example
      /// ```
      /// # use std::{fs::File, path::PathBuf};
      /// # use tiff::{GetPixel, Tiff, TEST_IMAGE_DIR, GetSample};
      /// # let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
      /// # let img_file = File::open(path).expect("image should exist");
      /// # let mut tiff = Tiff::new(img_file).expect("image should be valid");
      ///
      /// let region = tiff.read((11, 13), (3, 7)).expect("unable to read region");
      /// assert_eq!(3, region.rectangle().width());
      /// assert_eq!(7, region.rectangle().height());
      /// ```
      pub fn rectangle(&self) -> Rectangle {
        self.rectangle
      }

      /// Returns the width, in pixels, of this decoded region
      ///
      /// # Example
      /// ```
      /// # use std::{fs::File, path::PathBuf};
      /// # use tiff::{GetPixel, Tiff, TEST_IMAGE_DIR, GetSample};
      /// # let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
      /// # let img_file = File::open(path).expect("image should exist");
      /// # let mut tiff = Tiff::new(img_file).expect("image should be valid");
      ///
      /// let region = tiff.read((0, 0), (2, 4)).expect("unable to read region");
      /// assert_eq!(2, region.width());
      /// ```
      pub fn width(&self) -> usize {
        self.rectangle().width()
      }

      /// Returns the height, in pixels, of this decoded region
      ///
      /// # Example
      /// ```
      /// # use std::{fs::File, path::PathBuf};
      /// # use tiff::{GetPixel, Tiff, TEST_IMAGE_DIR, GetSample};
      /// # let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
      /// # let img_file = File::open(path).expect("image should exist");
      /// # let mut tiff = Tiff::new(img_file).expect("image should be valid");
      ///
      /// let region = tiff.read((0, 0), (2, 4)).expect("unable to read region");
      /// assert_eq!(4, region.height());
      /// ```
      pub fn height(&self) -> usize {
        self.rectangle().height()
      }

      /// Returns an iterator over the pixels in `self`
      ///
      /// The returned iterator yields tuples that contain the **absolute** `(x, y)` coordinates
      /// (i.e. relative to the TIFF source, rather than the [decoded region][Decoded]).
      ///
      /// # Example
      /// ```
      /// # use std::{fs::File, path::PathBuf};
      /// # use tiff::{GetPixel, Tiff, TEST_IMAGE_DIR, GetSample};
      /// # let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
      /// # let img_file = File::open(path).expect("image should exist");
      /// # let mut tiff = Tiff::new(img_file).expect("image should be valid");
      ///
      /// let region = tiff.read((0, 0), (2, 4)).expect("unable to read region");
      /// let pixels = region.pixels().collect::<Vec<_>>();
      /// assert_eq!(8, pixels.len());
      /// assert_eq!((1, 1), pixels[3].0);
      /// ```
      pub fn pixels(&self) -> impl Iterator<Item = ((usize, usize), Pixel<'_>)> {
        (0..self.height()).flat_map(move |rel_y| {
          (0..self.width()).filter_map(move |rel_x| {
            self.get_pixel((rel_x, rel_y)).map(|p| {
              ((rel_x + self.rectangle().corner_x(), rel_y + self.rectangle().corner_y()), p)
            })
          })
        })
      }
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

    impl<'a> GetPixel<'a, usize> for Decoded {
      /// Returns the pixel by index (in TIFF order)
      fn get_pixel(&'a self, index: usize) -> Option<Pixel<'a>> {
        let len = self.height() * self.width();
        if index >= len {
          return None;
        }

        let slc_len: usize = self.pixel_len.into();
        let start = index * slc_len;

        let sample_slice = &self.data[start..start + slc_len];
        assert_eq!(sample_slice.len(), slc_len);

        let nodata_slice = self.nodata_values.as_ref();

        assert_eq!(nodata_slice.len(), sample_slice.len());

        let pix = Pixel { band_types: self.bands.as_ref(), sample_slice, nodata_slice };

        Some(pix)
      }
    }

    impl<'a> GetPixel<'a, (usize, usize)> for Decoded {
      /// Returns the [`Pixel`] at the specified `(x, y)` coordinates, if one exists
      ///
      /// Note that the coordinates are specified relative to the decoded region. If either
      /// coordinate is out of bounds, `None` is returned.
      ///
      /// # Example: relative coordinates
      ///
      /// ```
      /// # use std::{fs::File, path::PathBuf};
      /// # use tiff::{GetPixel, Tiff, TEST_IMAGE_DIR, GetSample, Sample};
      /// # let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
      /// # let img_file = File::open(path).expect("image should exist");
      /// # let mut tiff = Tiff::new(img_file).expect("image should be valid");
      ///
      /// let region = tiff.read((50, 50), (7, 3)).expect("unable to read region");
      /// let sample = region.get_pixel((2, 1)).get_sample(0).unwrap();
      ///
      /// assert_eq!(Sample::I32(-9052051), sample);
      /// ```
      fn get_pixel(&'a self, coord: (usize, usize)) -> Option<Pixel<'a>> {
        if coord.0 >= self.width() || coord.1 >= self.height() {
          None
        } else {
          self.get_pixel(coord.0 + self.width() * coord.1)
        }
      }
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

          let s_slc = &self.sample_slice[rel_offset..rel_offset + sample_size];
          let n_slc = &self.nodata_slice[rel_offset..rel_offset + sample_size];

          if s_slc == n_slc {
            return None;
          }

          match band_type {
            BandType::U08 => {
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

    /// Implementation of [`GetSample`] for [`Option<Pixel>`], to allow chaining
    /// [`get_pixel`][GetPixel] and [`get_sample`][GetSample]
    impl<'a> GetSample<'a> for Option<Pixel<'a>> {
      /// Returns the [`Sample`] for the band at the specified index, or `None`
      fn get_sample(&'a self, idx: usize) -> Option<Sample> {
        self.as_ref().and_then(|some| GetSample::get_sample(some, idx))
      }
    }
  }

  mod band_type {
    //! Module containing [`BandType`] and its implementation(s)

    use crate::public_api::types::Byte;

    /// The types of supported bands, expressed in terms of primitive Rust types
    ///
    /// The TIFF specification permits bands to be, for example, 14-bit floating point numbers or
    /// 63-bit signed integers. This crate supports only types that map cleanly to Rust primitives.
    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    pub enum BandType {
      /// The band contains unsigned 8-bit integers
      U08,
      /// The band contains signed 32-bit integers
      I32,
      /// The band contains 32-bit IEEE floating point numbers
      F32,
    }

    impl BandType {
      /// Returns the width, i.e. the number of bytes that one value occupies, of the band
      pub fn width(self) -> Byte {
        match self {
          Self::U08 => 1,
          Self::I32 | Self::F32 => 4,
        }
        .into()
      }
    }
  }

  mod tiff {
    //! Hyper-rudimentary API for reading data from TIFFs
    //!
    //! # Example: instantiation
    //! ```
    //! use std::{
    //!   fs::File,
    //!   io::{Cursor, Read},
    //!   path::PathBuf,
    //! };
    //!
    //! use tiff::{Tiff, TEST_IMAGE_DIR};
    //!
    //! let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
    //! let mut file = File::open(path).expect("unable to open file");
    //! let mut image_data = Vec::new();
    //! file.read_to_end(&mut image_data).unwrap();
    //!
    //! let data = Cursor::new(image_data.as_slice());
    //! let tiff = Tiff::new(data).unwrap();
    //! ```

    use std::{
      any,
      io::{Read, Seek},
      str::FromStr,
    };

    use super::{Decoded, Error};
    use crate::{
      decoder::{ifd::Value, Decoder},
      public_api::{
        band_type::BandType,
        types::{Byte, Origin, Size},
      },
      tags::{SampleFormat, Tag},
    };

    /// An object providing access to pixels in a TIFF
    ///
    /// # Example
    ///
    /// ```
    /// use std::{fs::File, io::Read, path::PathBuf};
    ///
    /// use tiff::{Tiff, TEST_IMAGE_DIR};
    ///
    /// let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
    /// let file = File::open(path).expect("unable to open file");
    /// let tiff = Tiff::new(&file).unwrap();
    ///
    /// assert_eq!((192, 256), tiff.dimensions());
    /// ```
    pub struct Tiff<R> {
      /// The wrapped [`Decoder`], which performs the actual low-level reading and decoding
      decoder: Decoder<R>,
      /// The array of band types contained in this TIFF
      band_types: Box<[BandType]>,
      /// The value of the `GdalNodata` tag — which by convention is stored as an ASCII string (!).
      nodata_value: String,
    }

    /// A specialized [Result][std::result::Result] type for fallible operations in this module.
    /// This typedef is primarily used to avoid writing out [`Error`] everywhere, and is otherwise a
    /// direct mapping to [Result][std::result::Result].
    type Result<T> = std::result::Result<T, Error>;

    /// Basic properties of the fetch region and therefrom derived values
    struct CalculatedFetchRegion {
      /// The dimensions of the whole image
      img_dim: (usize, usize),
      /// The dimensions of each tile in the image
      tile_dim: (usize, usize),
      /// The origin of the fetch region
      origin: (usize, usize),
      /// The size of the fetch region
      fetch_rect: (usize, usize),
    }

    /// Data on the image/tile dimensions
    struct VerifiedTileCount {
      /// The dimensions of the whole image
      img_dim: (usize, usize),
      /// The dimensions of each tile in the image
      tile_dim: (usize, usize),
    }

    impl<R> Tiff<R> {
      /// Returns the image dimensions, in pixels
      pub fn dimensions(&self) -> (usize, usize) {
        self.decoder.dimensions()
      }

      /// Returns the length per pixel, in bytes (for example, an RGB image with 8-bit samples has
      /// 3-byte pixels)
      ///
      /// # Panics
      /// If the image on which this is called has a sample size that is not an even number of
      /// bytes, this function panics.
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

      /// Returns the array of [`self`]'s [band types][BandType]
      pub fn band_types(&self) -> &[BandType] {
        self.band_types.as_ref()
      }

      /// Returns the nodata value read from the TIFF header
      fn nodata_value(&self) -> &str {
        self.nodata_value.as_str()
      }
    }

    impl<R> Tiff<R>
    where
      R: Read + Seek,
    {
      /// Instantiates a new [`Tiff`]
      ///
      /// # Errors
      /// Creating a [`Tiff`] instance requires seeking and reading from the source TIFF, which
      /// means that [I/O errors][Error::IoError] may occur. Other typical errors are
      /// [`Error::UnsupportedFeature`], [`Error::UnsupportedBandType`] or [`Error::InvalidFormat`].
      pub fn new(r: R) -> Result<Self> {
        let mut decoder = Decoder::new(r)?;
        let band_types = convert_band_types(&decoder)?;

        let value = decoder.get_tag(Tag::GdalNodata)?;
        let nodata_value = match value {
          Value::Ascii(s) => Ok(s),
          _ => Err(Error::InvalidFormat("expected GdalNodata tag to be of type ASCII".into())),
        }?;

        Ok(Self { decoder, band_types, nodata_value })
      }

      /// Reads a [region][Decoded] from `self`
      ///
      /// # Errors
      /// A very large number of errors can possibly occur while reading a region, including the
      /// obvious I/O errors.
      pub fn read(&mut self, corner: (usize, usize), size: (usize, usize)) -> Result<Decoded> {
        /// Attempts to parse the supplied nodata string into a `T`, returning on error
        /// [`Error::InvalidFormat`]
        ///
        /// # Errors
        /// If the string cannot be parsed into a `T`, [`Error::InvalidFormat`] is returned
        fn try_parse<T>(val: &str) -> Result<T>
        where
          T: FromStr,
          <T as FromStr>::Err: std::error::Error,
        {
          T::from_str(val).map_err(|e| {
            Error::InvalidFormat(
              format!("nodata value '{val}' cannot be parsed into {}: {e}", any::type_name::<T>())
                .into(),
            )
          })
        }

        if size.0 == 0 || size.1 == 0 {
          return Err(Error::UsageError(
            "region to fetch must be at minimum 1×1 pixels".to_string(),
          ));
        }

        let (dx, dy) = self.dimensions();
        if corner.0 + size.0 >= dx {
          return Err(Error::UsageError(format!(
            "image X dimension {dx} < requested {}",
            corner.0 + size.0
          )));
        }
        if corner.1 + size.1 >= dy {
          return Err(Error::UsageError(format!(
            "image Y dimension {dy} < requested {}",
            corner.1 + size.1
          )));
        }

        let verified = VerifiedTileCount::new(
          self.dimensions(),
          self.decoder.chunk_dimensions(),
          self.decoder.image().chunk_offsets.len(),
        )?;

        let freg = CalculatedFetchRegion::new(&verified, corner, size);

        let sample_size = usize::from(self.pixel_width());

        let mut result_vec = vec![0u8; freg.rect_width() * freg.rect_height() * sample_size];

        let tile_width = freg.tile_width();
        let tile_height = freg.tile_height();

        for absyt in freg.start_y_tile()..=freg.end_y_tile() {
          for absxt in freg.start_x_tile()..=freg.end_x_tile() {
            let tsrc_hor_corner = absxt * tile_width;
            let tile_src_first_x = freg.origin_x().saturating_sub(tsrc_hor_corner);
            let l_right_x =
              ((freg.origin_x() + freg.rect_width()) - tsrc_hor_corner).min(tile_width);

            let tsrc_vert_corner = absyt * tile_height;
            let tile_src_first_y = freg.origin_y().saturating_sub(tsrc_vert_corner);
            let l_lower_y =
              ((freg.origin_y() + freg.rect_height()) - tsrc_vert_corner).min(tile_height);

            let src_first_x = tsrc_hor_corner + tile_src_first_x;
            let dst_first_x = src_first_x - freg.origin_x();
            let dst_right_x = dst_first_x + (l_right_x - tile_src_first_x);

            let src_first_y = tsrc_vert_corner + tile_src_first_y;
            let dst_first_y = src_first_y - freg.origin_y();
            let dst_lower_y = dst_first_y + (l_lower_y - tile_src_first_y);

            let src_range_y = tile_src_first_y..l_lower_y;
            let dst_range_y = dst_first_y..dst_lower_y;

            debug_assert_eq!(src_range_y.len(), dst_range_y.len());

            let tile_idx = absxt + absyt * freg.x_tiles();

            let chunk = self.decoder.read_chunk(tile_idx)?;
            let source_octet_slice = chunk.as_bytes();

            let (tile_width, tile_height) = self.decoder.chunk_data_dimensions(tile_idx);

            let actual = source_octet_slice.len();
            let expected = tile_width * tile_height * sample_size;
            if actual != expected {
              return Err(Error::Internal(
                format!("decoded result size incorrect: expected {expected} ≠ actual {actual}")
                  .into(),
              ));
            }

            for (sy, dy) in src_range_y.clone().zip(dst_range_y.clone()) {
              let src_start = tile_src_first_x + tile_width * sy;
              let src_count = l_right_x - tile_src_first_x;
              let src_offset = src_start * sample_size;
              let src_bytes = src_offset..src_offset + (sample_size * src_count);

              let dst_start = dst_first_x + freg.rect_width() * dy;
              let dst_count = dst_right_x - dst_first_x;
              let dst_offset = dst_start * sample_size;
              let dst_bytes = dst_offset..dst_offset + (sample_size * dst_count);

              result_vec[dst_bytes].copy_from_slice(&source_octet_slice[src_bytes]);
            }
          }
        }

        let mut ndv = Vec::with_capacity(usize::from(self.pixel_width()));

        for band in self.band_types() {
          let x = match band {
            BandType::U08 => try_parse::<u8>(self.nodata_value())?.to_le_bytes().to_vec(),
            BandType::F32 => try_parse::<f32>(self.nodata_value())?.to_le_bytes().to_vec(),
            BandType::I32 => try_parse::<i32>(self.nodata_value())?.to_le_bytes().to_vec(),
          };
          ndv.extend(x);
        }

        let size = Size::from(size);
        let origin = Origin::from(corner);
        let rect = size + origin;

        Ok(Decoded::new(self.band_types(), ndv, result_vec, rect))
      }
    }

    /// Calculates basic properties of the fetch region, based on the provided dimensions etc
    #[allow(dead_code)]
    impl CalculatedFetchRegion {
      /// Instantiates a new [`CalculatedFetchRegion`] for the provided image, tile and fetch
      /// configuration
      fn new(t: &VerifiedTileCount, origin: (usize, usize), fetch_rect: (usize, usize)) -> Self {
        Self { img_dim: t.img_dim, tile_dim: t.tile_dim, origin, fetch_rect }
      }

      /// The width, in pixels, of the fetch rectangle
      fn rect_width(&self) -> usize {
        self.fetch_rect.0
      }

      /// The height, in pixels, of the fetch rectangle
      fn rect_height(&self) -> usize {
        self.fetch_rect.1
      }

      /// The lowest X coordinate of the fetch rectangle
      fn origin_x(&self) -> usize {
        self.origin.0
      }

      /// The lowest Y coordinate of the fetch rectangle
      fn origin_y(&self) -> usize {
        self.origin.1
      }

      /// The width of the image's tiles
      fn tile_width(&self) -> usize {
        self.tile_dim.0
      }

      /// The height of the image's tiles
      fn tile_height(&self) -> usize {
        self.tile_dim.1
      }

      /// The X index of the tile that contains the fetch region's origin pixel
      fn start_x_tile(&self) -> usize {
        self.origin.0 / self.tile_dim.0
      }

      /// The Y index of the tile that contains the fetch region's origin pixel
      fn start_y_tile(&self) -> usize {
        self.origin.1 / self.tile_dim.1
      }

      /// The X index of the tile that contains the fetch region's last pixel
      fn end_x_tile(&self) -> usize {
        (self.origin.0 + self.fetch_rect.0) / self.tile_dim.0
      }

      /// The Y index of the tile that contains the fetch region's last pixel
      fn end_y_tile(&self) -> usize {
        (self.origin.1 + self.fetch_rect.1) / self.tile_dim.1
      }

      /// The number of tiles, in the X dimension, that the fetch region covers
      fn x_tiles(&self) -> usize {
        self.img_dim.0.div_ceil(self.tile_dim.0)
      }

      /// The number of tiles, in the Y dimension, that the fetch region covers
      fn y_tiles(&self) -> usize {
        self.img_dim.1.div_ceil(self.tile_dim.1)
      }
    }

    impl VerifiedTileCount {
      /// Returns a new instance of [`VerifiedTileCount`], if the supplied numbers match up
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
          return Err(Error::Internal(
            format!("expected {expected_tiles} but had {actual_tile_count} tiles").into(),
          ));
        }

        Ok(Self { img_dim, tile_dim })
      }
    }

    /// Returns the array of [band types][BandType] present in the source TIFF
    ///
    /// The TIFF specification allows bands to be signed/unsigned integers or floats, and each band
    /// can have an arbitrary width. This crate restricts the types of bands to combinations of
    /// sample formats and sample widths that match Rust-native primitives.
    ///
    /// # Errors
    /// Returns [`Error::UnsupportedBandType`] if the TIFF contains any band with a format that
    /// cannot be represented by [`BandType`]
    fn convert_band_types<R: Read + Seek>(decoder: &Decoder<R>) -> Result<Box<[BandType]>> {
      let bits_per_sample = decoder.image().bits_per_sample;
      let sample_formats = decoder.image().sample_format.as_slice();

      let mut result = Vec::with_capacity(sample_formats.len());

      for sf in sample_formats {
        result.push(match (sf, bits_per_sample) {
          (SampleFormat::Uint, 8) => BandType::U08,
          (SampleFormat::Int, 32) => BandType::I32,
          (SampleFormat::IEEEFP, 32) => BandType::F32,
          _ => {
            return Err(Error::UnsupportedBandType {
              format: format!("{sf:?}"),
              bit_width: bits_per_sample,
            });
          }
        });
      }

      Ok(result.into_boxed_slice())
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
        let cfr = CalculatedFetchRegion::new(&vtc, (128, 128), (256, 256));

        assert_eq!(0, cfr.start_x_tile());
        assert_eq!(0, cfr.start_y_tile());

        assert_eq!(1, cfr.end_x_tile());
        assert_eq!(1, cfr.end_y_tile());
      }
    }
  }

  #[cfg(test)]
  mod tests {
    use std::{fs::File, ops::Neg, path::PathBuf};

    use rand::{thread_rng, Rng};

    use super::*;
    use crate::{
      public_api::decoded::{GetPixel, GetSample},
      TEST_IMAGE_DIR,
    };

    #[test]
    fn api_example() {
      let path = PathBuf::from(TEST_IMAGE_DIR).join("fixture.tiff");
      let img_file = File::open(path).expect("image should exist");
      let mut tiff = Tiff::new(img_file).expect("image should be valid");

      let region = tiff.read((10, 10), (5, 5)).unwrap();

      // There are 25 pixels in the read region
      let pixels: Vec<_> = region.pixels().collect();
      assert_eq!(25, pixels.len());

      // 20 of the pixels have a defined sample(0); the others are undefined (nodata)
      let samples: Vec<_> = region.pixels().filter_map(|(_, pixel)| pixel.get_sample(0)).collect();
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
