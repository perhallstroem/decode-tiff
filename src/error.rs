use std::{error::Error, fmt, io, str, string};

use weezl::LzwError;

use crate::{
  decoder::ifd::Value,
  tags::{CompressionMethod, SampleFormat, Tag},
};

/// Tiff error kinds.
#[derive(Debug)]
pub enum TiffError {
  /// The Image is not formatted properly.
  FormatError(TiffFormatError),

  /// The Decoder does not support features required by the image.
  UnsupportedError(TiffUnsupportedError),

  /// An I/O Error occurred while decoding the image.
  IoError(io::Error),

  /// The Limits of the Decoder is exceeded.
  LimitsExceeded,

  /// An integer conversion to or from a platform size failed, either due to
  /// limits of the platform size or limits of the format.
  IntSizeError,
}

/// The image is not formatted properly.
///
/// This indicates that the encoder producing the image might behave incorrectly or that the input
/// file has been corrupted.
///
/// The list of variants may grow to incorporate errors of future features. Matching against this
/// exhaustively is not covered by interface stability guarantees.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TiffFormatError {
  TiffSignatureNotFound,
  TiffSignatureInvalid,
  ImageFileDirectoryNotFound,
  InconsistentSizesEncountered,
  UnexpectedCompressedData { actual_bytes: usize, required_bytes: usize },
  InconsistentStripSamples { actual_samples: usize, required_samples: usize },
  InvalidDimensions(u32, u32),
  InvalidTag,
  InvalidTagValueType(Tag),
  RequiredTagNotFound(Tag),
  UnknownPredictor(u16),
  UnknownPlanarConfiguration(u16),
  ByteExpected(Value),
  UnsignedIntegerExpected(Value),
  SignedIntegerExpected(Value),
  Format(String),
  RequiredTagEmpty(Tag),
  RequiredTileInformationNotFound,
  CycleInOffsets,
  SamplesPerPixelIsZero,
}

impl Error for TiffFormatError {}

impl fmt::Display for TiffFormatError {
  fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
    use self::TiffFormatError::*;
    match *self {
      TiffSignatureNotFound => write!(fmt, "TIFF signature not found."),
      TiffSignatureInvalid => write!(fmt, "TIFF signature invalid."),
      ImageFileDirectoryNotFound => write!(fmt, "Image file directory not found."),
      InconsistentSizesEncountered => write!(fmt, "Inconsistent sizes encountered."),
      UnexpectedCompressedData { actual_bytes, required_bytes } => {
        write!(
          fmt,
          "Decompression returned different amount of bytes than expected: got {}, expected {}.",
          actual_bytes, required_bytes
        )
      }
      InconsistentStripSamples { actual_samples, required_samples } => {
        write!(
          fmt,
          "Inconsistent elements in strip: got {}, expected {}.",
          actual_samples, required_samples
        )
      }
      InvalidDimensions(width, height) => write!(fmt, "Invalid dimensions: {}x{}.", width, height),
      InvalidTag => write!(fmt, "Image contains invalid tag."),
      InvalidTagValueType(ref tag) => {
        write!(fmt, "Tag `{:?}` did not have the expected value type.", tag)
      }
      RequiredTagNotFound(ref tag) => write!(fmt, "Required tag `{:?}` not found.", tag),
      UnknownPredictor(ref predictor) => {
        write!(fmt, "Unknown predictor “{}” encountered", predictor)
      }
      UnknownPlanarConfiguration(ref planar_config) => {
        write!(fmt, "Unknown planar configuration “{}” encountered", planar_config)
      }
      ByteExpected(ref val) => write!(fmt, "Expected byte, {:?} found.", val),
      UnsignedIntegerExpected(ref val) => {
        write!(fmt, "Expected unsigned integer, {:?} found.", val)
      }
      SignedIntegerExpected(ref val) => {
        write!(fmt, "Expected signed integer, {:?} found.", val)
      }
      Format(ref val) => write!(fmt, "Invalid format: {:?}.", val),
      RequiredTagEmpty(ref val) => write!(fmt, "Required tag {:?} was empty.", val),
      RequiredTileInformationNotFound => write!(fmt, "TIFF must be tiled"),
      CycleInOffsets => write!(fmt, "File contained a cycle in the list of IFDs"),
      SamplesPerPixelIsZero => write!(fmt, "Samples per pixel is zero"),
    }
  }
}

/// The Decoder does not support features required by the image.
///
/// This only captures known failures for which the standard either does not require support or an
/// implementation has been planned but not yet completed. Some variants may become unused over
/// time and will then get deprecated before being removed.
///
/// The list of variants may grow. Matching against this exhaustively is not covered by interface
/// stability guarantees.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TiffUnsupportedError {
  FloatingPointPredictor,
  HorizontalPredictor,
  InconsistentBitsPerSample(Vec<u8>),
  UnsupportedCompressionMethod(CompressionMethod),
  UnsupportedSampleDepth(u8),
  UnsupportedSampleFormat(Vec<SampleFormat>),
  UnsupportedBitsPerChannel(u8),
}

impl Error for TiffUnsupportedError {}

impl fmt::Display for TiffUnsupportedError {
  fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
    use self::TiffUnsupportedError::*;
    match *self {
      FloatingPointPredictor => {
        write!(fmt, "Floating point predictor for is unsupported for this image")
      }
      HorizontalPredictor => {
        write!(fmt, "Horizontal predictor for is unsupported for this image")
      }
      InconsistentBitsPerSample(ref bits_per_sample) => {
        write!(fmt, "Inconsistent bits per sample: {:?}.", bits_per_sample)
      }
      UnsupportedCompressionMethod(method) => {
        write!(fmt, "Compression method {:?} is unsupported", method)
      }
      UnsupportedSampleDepth(samples) => {
        write!(fmt, "{} samples per pixel is unsupported.", samples)
      }
      UnsupportedSampleFormat(ref formats) => {
        write!(fmt, "Sample format {:?} is unsupported.", formats)
      }
      UnsupportedBitsPerChannel(bits) => {
        write!(fmt, "{} bits per channel not supported", bits)
      }
    }
  }
}

impl fmt::Display for TiffError {
  fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
    match *self {
      TiffError::FormatError(ref e) => write!(fmt, "Format error: {}", e),
      TiffError::UnsupportedError(ref f) => write!(
        fmt,
        "The Decoder does not support the \
                 image format `{}`",
        f
      ),
      TiffError::IoError(ref e) => e.fmt(fmt),
      TiffError::LimitsExceeded => write!(fmt, "The Decoder limits are exceeded"),
      TiffError::IntSizeError => write!(fmt, "Platform or format size limits exceeded"),
    }
  }
}

impl Error for TiffError {
  fn description(&self) -> &str {
    match *self {
      TiffError::FormatError(..) => "Format error",
      TiffError::UnsupportedError(..) => "Unsupported error",
      TiffError::IoError(..) => "IO error",
      TiffError::LimitsExceeded => "Decoder limits exceeded",
      TiffError::IntSizeError => "Platform or format size limits exceeded",
    }
  }

  fn cause(&self) -> Option<&dyn Error> {
    match *self {
      TiffError::IoError(ref e) => Some(e),
      _ => None,
    }
  }
}

impl From<io::Error> for TiffError {
  fn from(err: io::Error) -> TiffError {
    TiffError::IoError(err)
  }
}

impl From<str::Utf8Error> for TiffError {
  fn from(_err: str::Utf8Error) -> TiffError {
    TiffError::FormatError(TiffFormatError::InvalidTag)
  }
}

impl From<string::FromUtf8Error> for TiffError {
  fn from(_err: string::FromUtf8Error) -> TiffError {
    TiffError::FormatError(TiffFormatError::InvalidTag)
  }
}

impl From<TiffFormatError> for TiffError {
  fn from(err: TiffFormatError) -> TiffError {
    TiffError::FormatError(err)
  }
}

impl From<TiffUnsupportedError> for TiffError {
  fn from(err: TiffUnsupportedError) -> TiffError {
    TiffError::UnsupportedError(err)
  }
}

impl From<std::num::TryFromIntError> for TiffError {
  fn from(_err: std::num::TryFromIntError) -> TiffError {
    TiffError::IntSizeError
  }
}

impl From<LzwError> for TiffError {
  fn from(err: LzwError) -> TiffError {
    match err {
      LzwError::InvalidCode => TiffError::FormatError(TiffFormatError::Format(String::from(
        "LZW compressed data corrupted",
      ))),
    }
  }
}

/// Result of an image decoding/encoding process
pub type TiffResult<T> = Result<T, TiffError>;
