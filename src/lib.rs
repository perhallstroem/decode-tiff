//! Decoding and Encoding of TIFF Images
//!
//! TIFF (Tagged Image File Format) is a versatile image format that supports
//! lossless and lossy compression.
//!
//! # Related Links
//! * <https://web.archive.org/web/20210108073850/https://www.adobe.io/open/standards/TIFF.html> - The TIFF specification

extern crate weezl;

mod bytecast;
pub mod decoder;
mod error;
pub mod tags;

pub use self::error::{TiffError, TiffFormatError, TiffResult, TiffUnsupportedError, UsageError};
use crate::decoder::ifd::Value;
use crate::decoder::{ChunkType, Decoder, DecodingResult};
use crate::tags::Tag;
use std::fs::File;
use std::path::PathBuf;

/// An enumeration over supported color types and their bit depths
#[derive(Copy, PartialEq, Eq, Debug, Clone, Hash)]
pub enum ColorType {
    /// Pixel is grayscale
    Gray(u8),

    /// Pixel contains R, G and B channels
    RGB(u8),

    /// Pixel is an index into a color palette
    Palette(u8),

    /// Pixel is grayscale with an alpha channel
    GrayA(u8),

    /// Pixel is RGB with an alpha channel
    RGBA(u8),

    /// Pixel is CMYK
    CMYK(u8),

    /// Pixel is YCbCr
    YCbCr(u8),
}
