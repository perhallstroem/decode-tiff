#![allow(clippy)]

use std::{
  collections::{HashMap, HashSet},
  convert::TryFrom,
  io::{self, Read, Seek},
  ops::Range,
};

use self::{
  ifd::Directory,
  image::Image,
  stream::{ByteOrder, EndianReader, SmartReader},
};
use crate::{
  bytecast,
  decoder::{ifd::Value, image::TileAttributes},
  tags::{CompressionMethod, Predictor, SampleFormat, Tag, Type},
  TiffError, TiffFormatError, TiffResult, TiffUnsupportedError,
};

pub mod ifd;
mod image;
mod stream;
mod tag_reader;

/// Result of a decoding process
#[derive(Debug)]
pub enum DecodingResult {
  /// A vector of unsigned bytes
  U8(Vec<u8>),
  /// A vector of unsigned words
  U16(Vec<u16>),
  /// A vector of 32 bit unsigned ints
  U32(Vec<u32>),
  /// A vector of 64 bit unsigned ints
  U64(Vec<u64>),
  /// A vector of 32 bit IEEE floats
  F32(Vec<f32>),
  /// A vector of 64 bit IEEE floats
  F64(Vec<f64>),
  /// A vector of 8 bit signed ints
  I8(Vec<i8>),
  /// A vector of 16 bit signed ints
  I16(Vec<i16>),
  /// A vector of 32 bit signed ints
  I32(Vec<i32>),
  /// A vector of 64 bit signed ints
  I64(Vec<i64>),
}

impl DecodingResult {
  fn new_u8(size: usize, limits: &Limits) -> TiffResult<DecodingResult> {
    if size > limits.decoding_buffer_size {
      Err(TiffError::LimitsExceeded)
    } else {
      Ok(DecodingResult::U8(vec![0; size]))
    }
  }

  fn new_u16(size: usize, limits: &Limits) -> TiffResult<DecodingResult> {
    if size > limits.decoding_buffer_size / 2 {
      Err(TiffError::LimitsExceeded)
    } else {
      Ok(DecodingResult::U16(vec![0; size]))
    }
  }

  fn new_u32(size: usize, limits: &Limits) -> TiffResult<DecodingResult> {
    if size > limits.decoding_buffer_size / 4 {
      Err(TiffError::LimitsExceeded)
    } else {
      Ok(DecodingResult::U32(vec![0; size]))
    }
  }

  fn new_u64(size: usize, limits: &Limits) -> TiffResult<DecodingResult> {
    if size > limits.decoding_buffer_size / 8 {
      Err(TiffError::LimitsExceeded)
    } else {
      Ok(DecodingResult::U64(vec![0; size]))
    }
  }

  fn new_f32(size: usize, limits: &Limits) -> TiffResult<DecodingResult> {
    if size > limits.decoding_buffer_size / std::mem::size_of::<f32>() {
      Err(TiffError::LimitsExceeded)
    } else {
      Ok(DecodingResult::F32(vec![0.0; size]))
    }
  }

  fn new_f64(size: usize, limits: &Limits) -> TiffResult<DecodingResult> {
    if size > limits.decoding_buffer_size / std::mem::size_of::<f64>() {
      Err(TiffError::LimitsExceeded)
    } else {
      Ok(DecodingResult::F64(vec![0.0; size]))
    }
  }

  fn new_i8(size: usize, limits: &Limits) -> TiffResult<DecodingResult> {
    if size > limits.decoding_buffer_size / std::mem::size_of::<i8>() {
      Err(TiffError::LimitsExceeded)
    } else {
      Ok(DecodingResult::I8(vec![0; size]))
    }
  }

  fn new_i16(size: usize, limits: &Limits) -> TiffResult<DecodingResult> {
    if size > limits.decoding_buffer_size / 2 {
      Err(TiffError::LimitsExceeded)
    } else {
      Ok(DecodingResult::I16(vec![0; size]))
    }
  }

  fn new_i32(size: usize, limits: &Limits) -> TiffResult<DecodingResult> {
    if size > limits.decoding_buffer_size / 4 {
      Err(TiffError::LimitsExceeded)
    } else {
      Ok(DecodingResult::I32(vec![0; size]))
    }
  }

  fn new_i64(size: usize, limits: &Limits) -> TiffResult<DecodingResult> {
    if size > limits.decoding_buffer_size / 8 {
      Err(TiffError::LimitsExceeded)
    } else {
      Ok(DecodingResult::I64(vec![0; size]))
    }
  }

  pub fn as_buffer(&mut self, start: usize) -> DecodingBuffer<'_> {
    match *self {
      DecodingResult::U8(ref mut buf) => DecodingBuffer::U8(&mut buf[start..]),
      DecodingResult::U16(ref mut buf) => DecodingBuffer::U16(&mut buf[start..]),
      DecodingResult::U32(ref mut buf) => DecodingBuffer::U32(&mut buf[start..]),
      DecodingResult::U64(ref mut buf) => DecodingBuffer::U64(&mut buf[start..]),
      DecodingResult::F32(ref mut buf) => DecodingBuffer::F32(&mut buf[start..]),
      DecodingResult::F64(ref mut buf) => DecodingBuffer::F64(&mut buf[start..]),
      DecodingResult::I8(ref mut buf) => DecodingBuffer::I8(&mut buf[start..]),
      DecodingResult::I16(ref mut buf) => DecodingBuffer::I16(&mut buf[start..]),
      DecodingResult::I32(ref mut buf) => DecodingBuffer::I32(&mut buf[start..]),
      DecodingResult::I64(ref mut buf) => DecodingBuffer::I64(&mut buf[start..]),
    }
  }

  pub fn as_bytes(&self) -> &[u8] {
    match self {
      Self::U8(buf) => buf,
      Self::I8(buf) => bytecast::i8_as_ne_bytes(buf),
      Self::U16(buf) => bytecast::u16_as_ne_bytes(buf),
      Self::I16(buf) => bytecast::i16_as_ne_bytes(buf),
      Self::U32(buf) => bytecast::u32_as_ne_bytes(buf),
      Self::I32(buf) => bytecast::i32_as_ne_bytes(buf),
      Self::U64(buf) => bytecast::u64_as_ne_bytes(buf),
      Self::I64(buf) => bytecast::i64_as_ne_bytes(buf),
      Self::F32(buf) => bytecast::f32_as_ne_bytes(buf),
      Self::F64(buf) => bytecast::f64_as_ne_bytes(buf),
    }
  }
}

// A buffer for image decoding
pub enum DecodingBuffer<'a> {
  /// A slice of unsigned bytes
  U8(&'a mut [u8]),
  /// A slice of unsigned words
  U16(&'a mut [u16]),
  /// A slice of 32 bit unsigned ints
  U32(&'a mut [u32]),
  /// A slice of 64 bit unsigned ints
  U64(&'a mut [u64]),
  /// A slice of 32 bit IEEE floats
  F32(&'a mut [f32]),
  /// A slice of 64 bit IEEE floats
  F64(&'a mut [f64]),
  /// A slice of 8 bits signed ints
  I8(&'a mut [i8]),
  /// A slice of 16 bits signed ints
  I16(&'a mut [i16]),
  /// A slice of 32 bits signed ints
  I32(&'a mut [i32]),
  /// A slice of 64 bits signed ints
  I64(&'a mut [i64]),
}

impl<'a> DecodingBuffer<'a> {
  fn byte_len(&self) -> usize {
    match *self {
      DecodingBuffer::U8(_) => 1,
      DecodingBuffer::U16(_) => 2,
      DecodingBuffer::U32(_) => 4,
      DecodingBuffer::U64(_) => 8,
      DecodingBuffer::F32(_) => 4,
      DecodingBuffer::F64(_) => 8,
      DecodingBuffer::I8(_) => 1,
      DecodingBuffer::I16(_) => 2,
      DecodingBuffer::I32(_) => 4,
      DecodingBuffer::I64(_) => 8,
    }
  }

  fn copy<'b>(&'b mut self) -> DecodingBuffer<'b>
  where
    'a: 'b,
  {
    match *self {
      DecodingBuffer::U8(ref mut buf) => DecodingBuffer::U8(buf),
      DecodingBuffer::U16(ref mut buf) => DecodingBuffer::U16(buf),
      DecodingBuffer::U32(ref mut buf) => DecodingBuffer::U32(buf),
      DecodingBuffer::U64(ref mut buf) => DecodingBuffer::U64(buf),
      DecodingBuffer::F32(ref mut buf) => DecodingBuffer::F32(buf),
      DecodingBuffer::F64(ref mut buf) => DecodingBuffer::F64(buf),
      DecodingBuffer::I8(ref mut buf) => DecodingBuffer::I8(buf),
      DecodingBuffer::I16(ref mut buf) => DecodingBuffer::I16(buf),
      DecodingBuffer::I32(ref mut buf) => DecodingBuffer::I32(buf),
      DecodingBuffer::I64(ref mut buf) => DecodingBuffer::I64(buf),
    }
  }

  fn subrange<'b>(&'b mut self, range: Range<usize>) -> DecodingBuffer<'b>
  where
    'a: 'b,
  {
    match *self {
      DecodingBuffer::U8(ref mut buf) => DecodingBuffer::U8(&mut buf[range]),
      DecodingBuffer::U16(ref mut buf) => DecodingBuffer::U16(&mut buf[range]),
      DecodingBuffer::U32(ref mut buf) => DecodingBuffer::U32(&mut buf[range]),
      DecodingBuffer::U64(ref mut buf) => DecodingBuffer::U64(&mut buf[range]),
      DecodingBuffer::F32(ref mut buf) => DecodingBuffer::F32(&mut buf[range]),
      DecodingBuffer::F64(ref mut buf) => DecodingBuffer::F64(&mut buf[range]),
      DecodingBuffer::I8(ref mut buf) => DecodingBuffer::I8(&mut buf[range]),
      DecodingBuffer::I16(ref mut buf) => DecodingBuffer::I16(&mut buf[range]),
      DecodingBuffer::I32(ref mut buf) => DecodingBuffer::I32(&mut buf[range]),
      DecodingBuffer::I64(ref mut buf) => DecodingBuffer::I64(&mut buf[range]),
    }
  }

  fn as_bytes_mut(&mut self) -> &mut [u8] {
    match self {
      DecodingBuffer::U8(ref mut buf) => buf,
      DecodingBuffer::I8(buf) => bytecast::i8_as_ne_mut_bytes(buf),
      DecodingBuffer::U16(buf) => bytecast::u16_as_ne_mut_bytes(buf),
      DecodingBuffer::I16(buf) => bytecast::i16_as_ne_mut_bytes(buf),
      DecodingBuffer::U32(buf) => bytecast::u32_as_ne_mut_bytes(buf),
      DecodingBuffer::I32(buf) => bytecast::i32_as_ne_mut_bytes(buf),
      DecodingBuffer::U64(buf) => bytecast::u64_as_ne_mut_bytes(buf),
      DecodingBuffer::I64(buf) => bytecast::i64_as_ne_mut_bytes(buf),
      DecodingBuffer::F32(buf) => bytecast::f32_as_ne_mut_bytes(buf),
      DecodingBuffer::F64(buf) => bytecast::f64_as_ne_mut_bytes(buf),
    }
  }
}

#[derive(Debug, Copy, Clone, PartialEq)]
/// Chunk type of the internal representation
pub enum ChunkType {
  Tile,
}

/// Decoding limits
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Limits {
  /// The maximum size of any `DecodingResult` in bytes, the default is
  /// 256MiB. If the entire image is decoded at once, then this will
  /// be the maximum size of the image. If it is decoded one strip at a
  /// time, this will be the maximum size of a strip.
  pub decoding_buffer_size: usize,
  /// The maximum size of any ifd value in bytes, the default is
  /// 1MiB.
  pub ifd_value_size: usize,
  /// Maximum size for intermediate buffer which may be used to limit the amount of data read per
  /// segment even if the entire image is decoded at once.
  pub intermediate_buffer_size: usize,
}

impl Limits {
  /// A configuration that does not impose any limits.
  ///
  /// This is a good start if the caller only wants to impose selective limits, contrary to the
  /// default limits which allows selectively disabling limits.
  ///
  /// Note that this configuration is likely to crash on excessively large images since,
  /// naturally, the machine running the program does not have infinite memory.
  pub fn unlimited() -> Limits {
    Limits {
      decoding_buffer_size: usize::max_value(),
      ifd_value_size: usize::max_value(),
      intermediate_buffer_size: usize::max_value(),
    }
  }
}

impl Default for Limits {
  fn default() -> Limits {
    Limits {
      decoding_buffer_size: 256 * 1024 * 1024,
      intermediate_buffer_size: 128 * 1024 * 1024,
      ifd_value_size: 1024 * 1024,
    }
  }
}

/// The representation of a TIFF decoder
///
/// Currently does not support decoding of interlaced images
#[derive(Debug)]
pub struct Decoder<R> {
  reader: SmartReader<R>,
  bigtiff: bool,
  limits: Limits,
  next_ifd: Option<u64>,
  ifd_offsets: Vec<u64>,
  seen_ifds: HashSet<u64>,
  image: Image,
}

trait Wrapping {
  fn wrapping_add(&self, other: Self) -> Self;
}

impl Wrapping for u8 {
  fn wrapping_add(&self, other: Self) -> Self {
    u8::wrapping_add(*self, other)
  }
}

impl Wrapping for u16 {
  fn wrapping_add(&self, other: Self) -> Self {
    u16::wrapping_add(*self, other)
  }
}

impl Wrapping for u32 {
  fn wrapping_add(&self, other: Self) -> Self {
    u32::wrapping_add(*self, other)
  }
}

impl Wrapping for u64 {
  fn wrapping_add(&self, other: Self) -> Self {
    u64::wrapping_add(*self, other)
  }
}

impl Wrapping for i8 {
  fn wrapping_add(&self, other: Self) -> Self {
    i8::wrapping_add(*self, other)
  }
}

impl Wrapping for i16 {
  fn wrapping_add(&self, other: Self) -> Self {
    i16::wrapping_add(*self, other)
  }
}

impl Wrapping for i32 {
  fn wrapping_add(&self, other: Self) -> Self {
    i32::wrapping_add(*self, other)
  }
}

impl Wrapping for i64 {
  fn wrapping_add(&self, other: Self) -> Self {
    i64::wrapping_add(*self, other)
  }
}

fn rev_hpredict_nsamp<T: Copy + Wrapping>(image: &mut [T], samples: usize) {
  for col in samples..image.len() {
    image[col] = image[col].wrapping_add(image[col - samples]);
  }
}

pub fn fp_predict_f32(input: &mut [u8], output: &mut [f32], samples: usize) {
  rev_hpredict_nsamp(input, samples);
  for i in 0..output.len() {
    output[i] = f32::from_be_bytes([
      input[i],
      input[input.len() / 4 + i],
      input[input.len() / 4 * 2 + i],
      input[input.len() / 4 * 3 + i],
    ]);
  }
}

pub fn fp_predict_f64(input: &mut [u8], output: &mut [f64], samples: usize) {
  rev_hpredict_nsamp(input, samples);
  for i in 0..output.len() {
    output[i] = f64::from_be_bytes([
      input[i],
      input[input.len() / 8 + i],
      input[input.len() / 8 * 2 + i],
      input[input.len() / 8 * 3 + i],
      input[input.len() / 8 * 4 + i],
      input[input.len() / 8 * 5 + i],
      input[input.len() / 8 * 6 + i],
      input[input.len() / 8 * 7 + i],
    ])
  }
}

fn fix_endianness_and_predict(
  mut image: DecodingBuffer<'_>, samples: usize, byte_order: ByteOrder, predictor: Predictor,
) {
  match predictor {
    Predictor::None => {
      fix_endianness(&mut image, byte_order);
    }
    Predictor::Horizontal => {
      fix_endianness(&mut image, byte_order);
      match image {
        DecodingBuffer::U8(buf) => rev_hpredict_nsamp(buf, samples),
        DecodingBuffer::U16(buf) => rev_hpredict_nsamp(buf, samples),
        DecodingBuffer::U32(buf) => rev_hpredict_nsamp(buf, samples),
        DecodingBuffer::U64(buf) => rev_hpredict_nsamp(buf, samples),
        DecodingBuffer::I8(buf) => rev_hpredict_nsamp(buf, samples),
        DecodingBuffer::I16(buf) => rev_hpredict_nsamp(buf, samples),
        DecodingBuffer::I32(buf) => rev_hpredict_nsamp(buf, samples),
        DecodingBuffer::I64(buf) => rev_hpredict_nsamp(buf, samples),
        DecodingBuffer::F32(_) | DecodingBuffer::F64(_) => {
          unreachable!("Caller should have validated arguments. Please file a bug.")
        }
      }
    }
    Predictor::FloatingPoint => {
      let mut buffer_copy = image.as_bytes_mut().to_vec();
      match image {
        DecodingBuffer::F32(buf) => fp_predict_f32(&mut buffer_copy, buf, samples),
        DecodingBuffer::F64(buf) => fp_predict_f64(&mut buffer_copy, buf, samples),
        _ => unreachable!("Caller should have validated arguments. Please file a bug."),
      }
    }
  }
}

/// Fix endianness. If `byte_order` matches the host, then conversion is a no-op.
fn fix_endianness(buf: &mut DecodingBuffer<'_>, byte_order: ByteOrder) {
  match byte_order {
    ByteOrder::LittleEndian => match buf {
      DecodingBuffer::U8(_) | DecodingBuffer::I8(_) => {}
      DecodingBuffer::U16(b) => b.iter_mut().for_each(|v| *v = u16::from_le(*v)),
      DecodingBuffer::I16(b) => b.iter_mut().for_each(|v| *v = i16::from_le(*v)),
      DecodingBuffer::U32(b) => b.iter_mut().for_each(|v| *v = u32::from_le(*v)),
      DecodingBuffer::I32(b) => b.iter_mut().for_each(|v| *v = i32::from_le(*v)),
      DecodingBuffer::U64(b) => b.iter_mut().for_each(|v| *v = u64::from_le(*v)),
      DecodingBuffer::I64(b) => b.iter_mut().for_each(|v| *v = i64::from_le(*v)),
      DecodingBuffer::F32(b) => {
        b.iter_mut().for_each(|v| *v = f32::from_bits(u32::from_le(v.to_bits())))
      }
      DecodingBuffer::F64(b) => {
        b.iter_mut().for_each(|v| *v = f64::from_bits(u64::from_le(v.to_bits())))
      }
    },
    ByteOrder::BigEndian => match buf {
      DecodingBuffer::U8(_) | DecodingBuffer::I8(_) => {}
      DecodingBuffer::U16(b) => b.iter_mut().for_each(|v| *v = u16::from_be(*v)),
      DecodingBuffer::I16(b) => b.iter_mut().for_each(|v| *v = i16::from_be(*v)),
      DecodingBuffer::U32(b) => b.iter_mut().for_each(|v| *v = u32::from_be(*v)),
      DecodingBuffer::I32(b) => b.iter_mut().for_each(|v| *v = i32::from_be(*v)),
      DecodingBuffer::U64(b) => b.iter_mut().for_each(|v| *v = u64::from_be(*v)),
      DecodingBuffer::I64(b) => b.iter_mut().for_each(|v| *v = i64::from_be(*v)),
      DecodingBuffer::F32(b) => {
        b.iter_mut().for_each(|v| *v = f32::from_bits(u32::from_be(v.to_bits())))
      }
      DecodingBuffer::F64(b) => {
        b.iter_mut().for_each(|v| *v = f64::from_bits(u64::from_be(v.to_bits())))
      }
    },
  };
}

impl<R> Decoder<R> {
  pub fn dimensions(&self) -> (usize, usize) {
    (self.image().width(), self.image().height())
  }

  pub(crate) fn image(&self) -> &Image {
    &self.image
  }
}

impl<R: Read + Seek> Decoder<R> {
  /// Create a new decoder that decodes from the stream ```r```
  pub fn new(mut r: R) -> TiffResult<Decoder<R>> {
    let mut endianess = Vec::with_capacity(2);
    (&mut r).take(2).read_to_end(&mut endianess)?;
    let byte_order = match &*endianess {
      b"II" => ByteOrder::LittleEndian,
      b"MM" => ByteOrder::BigEndian,
      _ => return Err(TiffError::FormatError(TiffFormatError::TiffSignatureNotFound)),
    };
    let mut reader = SmartReader::wrap(r, byte_order);

    let bigtiff = match reader.read_u16()? {
      42 => false,
      43 => {
        // Read bytesize of offsets (in bigtiff it's alway 8 but provide a way to move to 16 some
        // day)
        if reader.read_u16()? != 8 {
          return Err(TiffError::FormatError(TiffFormatError::TiffSignatureNotFound));
        }
        // This constant should always be 0
        if reader.read_u16()? != 0 {
          return Err(TiffError::FormatError(TiffFormatError::TiffSignatureNotFound));
        }
        true
      }
      _ => return Err(TiffError::FormatError(TiffFormatError::TiffSignatureInvalid)),
    };
    let next_ifd = if bigtiff {
      Some(reader.read_u64()?)
    } else {
      Some(u64::from(reader.read_u32()?))
    };

    let mut seen_ifds = HashSet::new();
    seen_ifds.insert(*next_ifd.as_ref().unwrap());
    let ifd_offsets = vec![*next_ifd.as_ref().unwrap()];

    let mut decoder = Decoder {
      reader,
      bigtiff,
      limits: Default::default(),
      next_ifd,
      ifd_offsets,
      seen_ifds,
      image: Image {
        ifd: None,
        bits_per_sample: 1,
        samples: 1,
        sample_format: vec![SampleFormat::Uint],
        compression_method: CompressionMethod::None,
        predictor: Predictor::None,
        tile_attributes: TileAttributes::default(),
        chunk_offsets: Vec::new(),
        chunk_bytes: Vec::new(),
      },
    };
    decoder.next_image()?;
    Ok(decoder)
  }

  pub fn with_limits(mut self, limits: Limits) -> Decoder<R> {
    self.limits = limits;
    self
  }

  /// Loads the IFD at the specified index in the list, if one exists
  pub fn seek_to_image(&mut self, ifd_index: usize) -> TiffResult<()> {
    // Check whether we have seen this IFD before, if so then the index will be less than the length
    // of the list of ifd offsets
    if ifd_index >= self.ifd_offsets.len() {
      // We possibly need to load in the next IFD
      if self.next_ifd.is_none() {
        return Err(TiffError::FormatError(TiffFormatError::ImageFileDirectoryNotFound));
      }

      loop {
        // Follow the list until we find the one we want, or we reach the end, whichever happens
        // first
        let (_ifd, next_ifd) = self.next_ifd()?;

        if next_ifd.is_none() {
          break;
        }

        if ifd_index < self.ifd_offsets.len() {
          break;
        }
      }
    }

    // If the index is within the list of ifds then we can load the selected image/IFD
    if let Some(ifd_offset) = self.ifd_offsets.get(ifd_index) {
      let (ifd, _next_ifd) = Self::read_ifd(&mut self.reader, self.bigtiff, *ifd_offset)?;

      self.image = Image::from_reader(&mut self.reader, ifd, &self.limits, self.bigtiff)?;

      Ok(())
    } else {
      Err(TiffError::FormatError(TiffFormatError::ImageFileDirectoryNotFound))
    }
  }

  fn next_ifd(&mut self) -> TiffResult<(Directory, Option<u64>)> {
    if self.next_ifd.is_none() {
      return Err(TiffError::FormatError(TiffFormatError::ImageFileDirectoryNotFound));
    }

    let (ifd, next_ifd) =
      Self::read_ifd(&mut self.reader, self.bigtiff, self.next_ifd.take().unwrap())?;

    if let Some(next) = next_ifd {
      if !self.seen_ifds.insert(next) {
        return Err(TiffError::FormatError(TiffFormatError::CycleInOffsets));
      }
      self.next_ifd = Some(next);
      self.ifd_offsets.push(next);
    }

    Ok((ifd, next_ifd))
  }

  /// Reads in the next image.
  /// If there is no further image in the TIFF file a format error is returned.
  /// To determine whether there are more images call `TIFFDecoder::more_images` instead.
  pub fn next_image(&mut self) -> TiffResult<()> {
    let (ifd, _next_ifd) = self.next_ifd()?;

    self.image = Image::from_reader(&mut self.reader, ifd, &self.limits, self.bigtiff)?;
    Ok(())
  }

  /// Returns `true` if there is at least one more image available.
  pub fn more_images(&self) -> bool {
    self.next_ifd.is_some()
  }

  /// Returns the byte_order
  pub fn byte_order(&self) -> ByteOrder {
    self.reader.byte_order
  }

  #[inline]
  pub fn read_ifd_offset(&mut self) -> Result<u64, io::Error> {
    if self.bigtiff {
      self.read_long8()
    } else {
      self.read_long().map(u64::from)
    }
  }

  /// Reads a TIFF long value
  #[inline]
  pub fn read_long(&mut self) -> Result<u32, io::Error> {
    self.reader.read_u32()
  }

  #[inline]
  pub fn read_long8(&mut self) -> Result<u64, io::Error> {
    self.reader.read_u64()
  }

  /// Reads a string
  #[inline]
  pub fn read_string(&mut self, length: usize) -> TiffResult<String> {
    let mut out = vec![0; length];
    self.reader.read_exact(&mut out)?;
    // Strings may be null-terminated, so we trim anything downstream of the null byte
    if let Some(first) = out.iter().position(|&b| b == 0) {
      out.truncate(first);
    }
    Ok(String::from_utf8(out)?)
  }

  #[inline]
  pub fn goto_offset_u64(&mut self, offset: u64) -> io::Result<()> {
    self.reader.seek(io::SeekFrom::Start(offset)).map(|_| ())
  }

  /// Reads a IFD entry.
  // An IFD entry has four fields:
  //
  // Tag   2 bytes
  // Type  2 bytes
  // Count 4 bytes
  // Value 4 bytes either a pointer the value itself
  fn read_entry(
    reader: &mut SmartReader<R>, bigtiff: bool,
  ) -> TiffResult<Option<(Tag, ifd::Entry)>> {
    let tag = Tag::from_u16_exhaustive(reader.read_u16()?);
    let type_ = if let Some(t) = Type::from_u16(reader.read_u16()?) {
      t
    } else {
      // Unknown type. Skip this entry according to spec.
      reader.read_u32()?;
      reader.read_u32()?;
      return Ok(None);
    };

    let entry = if bigtiff {
      let mut offset = [0; 8];

      let count = reader.read_u64()?;
      reader.read_exact(&mut offset)?;
      ifd::Entry::new_u64(type_, count, offset)
    } else {
      let mut offset = [0; 4];

      let count = reader.read_u32()?;
      reader.read_exact(&mut offset)?;
      ifd::Entry::new(type_, count, offset)
    };
    Ok(Some((tag, entry)))
  }

  /// Reads the IFD starting at the indicated location.
  fn read_ifd(
    reader: &mut SmartReader<R>, bigtiff: bool, ifd_location: u64,
  ) -> TiffResult<(Directory, Option<u64>)> {
    reader.goto_offset(ifd_location)?;

    let mut dir: Directory = HashMap::new();

    let num_tags = if bigtiff { reader.read_u64()? } else { reader.read_u16()?.into() };
    for _ in 0..num_tags {
      let (tag, entry) = match Self::read_entry(reader, bigtiff)? {
        Some(val) => val,
        None => {
          continue;
        } // Unknown data type in tag, skip
      };
      dir.insert(tag, entry);
    }

    let next_ifd = if bigtiff { reader.read_u64()? } else { reader.read_u32()?.into() };

    let next_ifd = match next_ifd {
      0 => None,
      _ => Some(next_ifd),
    };

    Ok((dir, next_ifd))
  }

  /// Tries to retrieve a tag.
  /// Return `Ok(None)` if the tag is not present.
  pub fn find_tag(&mut self, tag: Tag) -> TiffResult<Option<ifd::Value>> {
    let entry = match self.image().ifd.as_ref().unwrap().get(&tag) {
      None => return Ok(None),
      Some(entry) => entry.clone(),
    };

    Ok(Some(entry.val(&self.limits, self.bigtiff, &mut self.reader)?))
  }

  /// Tries to retrieve a tag and convert it to the desired unsigned type.
  pub fn find_tag_unsigned<T: TryFrom<u64>>(&mut self, tag: Tag) -> TiffResult<Option<T>> {
    self
      .find_tag(tag)?
      .map(Value::into_u64)
      .transpose()?
      .map(|value| T::try_from(value).map_err(|_| TiffFormatError::InvalidTagValueType(tag).into()))
      .transpose()
  }

  /// Tries to retrieve a vector of all a tag's values and convert them to
  /// the desired unsigned type.
  pub fn find_tag_unsigned_vec<T: TryFrom<u64>>(&mut self, tag: Tag) -> TiffResult<Option<Vec<T>>> {
    self
      .find_tag(tag)?
      .map(Value::into_u64_vec)
      .transpose()?
      .map(|v| {
        v.into_iter()
          .map(|u| T::try_from(u).map_err(|_| TiffFormatError::InvalidTagValueType(tag).into()))
          .collect()
      })
      .transpose()
  }

  /// Tries to retrieve a tag and convert it to the desired unsigned type.
  /// Returns an error if the tag is not present.
  pub fn get_tag_unsigned<T: TryFrom<u64>>(&mut self, tag: Tag) -> TiffResult<T> {
    self
      .find_tag_unsigned(tag)?
      .ok_or_else(|| TiffFormatError::RequiredTagNotFound(tag).into())
  }

  /// Tries to retrieve a tag.
  /// Returns an error if the tag is not present
  pub fn get_tag(&mut self, tag: Tag) -> TiffResult<ifd::Value> {
    match self.find_tag(tag)? {
      Some(val) => Ok(val),
      None => Err(TiffError::FormatError(TiffFormatError::RequiredTagNotFound(tag))),
    }
  }

  /// Number of tiles in image
  pub fn tile_count(&mut self) -> usize {
    self.image().chunk_offsets.len()
  }

  pub fn read_chunk_to_buffer(
    &mut self, mut buffer: DecodingBuffer<'_>, chunk_index: usize, output_width: usize,
  ) -> TiffResult<()> {
    let offset = self.image.chunk_file_range(chunk_index)?.0;
    self.goto_offset_u64(offset)?;

    let byte_order = self.reader.byte_order;

    self.image.expand_chunk(
      &mut self.reader,
      buffer.copy(),
      output_width,
      byte_order,
      chunk_index,
      &self.limits,
    )?;

    Ok(())
  }

  fn result_buffer(&self, width: usize, height: usize) -> TiffResult<DecodingResult> {
    let buffer_size =
      match width.checked_mul(height).and_then(|x| x.checked_mul(self.image().samples_per_pixel()))
      {
        Some(s) => s,
        None => return Err(TiffError::LimitsExceeded),
      };

    let max_sample_bits = self.image().bits_per_sample;
    match self.image().sample_format.first().unwrap_or(&SampleFormat::Uint) {
      SampleFormat::Uint => match max_sample_bits {
        n if n <= 8 => DecodingResult::new_u8(buffer_size, &self.limits),
        n if n <= 16 => DecodingResult::new_u16(buffer_size, &self.limits),
        n if n <= 32 => DecodingResult::new_u32(buffer_size, &self.limits),
        n if n <= 64 => DecodingResult::new_u64(buffer_size, &self.limits),
        n => Err(TiffError::UnsupportedError(TiffUnsupportedError::UnsupportedBitsPerChannel(n))),
      },
      SampleFormat::IEEEFP => match max_sample_bits {
        32 => DecodingResult::new_f32(buffer_size, &self.limits),
        64 => DecodingResult::new_f64(buffer_size, &self.limits),
        n => Err(TiffError::UnsupportedError(TiffUnsupportedError::UnsupportedBitsPerChannel(n))),
      },
      SampleFormat::Int => match max_sample_bits {
        n if n <= 8 => DecodingResult::new_i8(buffer_size, &self.limits),
        n if n <= 16 => DecodingResult::new_i16(buffer_size, &self.limits),
        n if n <= 32 => DecodingResult::new_i32(buffer_size, &self.limits),
        n if n <= 64 => DecodingResult::new_i64(buffer_size, &self.limits),
        n => Err(TiffError::UnsupportedError(TiffUnsupportedError::UnsupportedBitsPerChannel(n))),
      },
      format => Err(TiffUnsupportedError::UnsupportedSampleFormat(vec![*format]).into()),
    }
  }

  /// Read the specified chunk (at index `chunk_index`) and return the binary data as a Vector.
  pub fn read_chunk(&mut self, chunk_index: usize) -> TiffResult<DecodingResult> {
    let data_dims = self.image().chunk_data_dimensions(chunk_index);

    let mut result = self.result_buffer(data_dims.0, data_dims.1)?;

    self.read_chunk_to_buffer(result.as_buffer(0), chunk_index, data_dims.0)?;

    Ok(result)
  }

  /// Returns the default chunk size for the current image. Any given chunk in the image is at most
  /// as large as the value returned here. For the size of the data (chunk minus padding), use
  /// `chunk_data_dimensions`.
  pub fn chunk_dimensions(&self) -> (usize, usize) {
    self.image().chunk_dimensions()
  }

  /// Returns the size of the data in the chunk with the specified index. This is the default size
  /// of the chunk, minus any padding.
  pub fn chunk_data_dimensions(&self, chunk_index: usize) -> (usize, usize) {
    self.image().chunk_data_dimensions(chunk_index)
  }

  /// Decodes the entire image and return it as a Vector
  pub fn read_image(&mut self) -> TiffResult<DecodingResult> {
    let width = self.image().width();
    let height = self.image().height();
    let mut result = self.result_buffer(width, height)?;
    if width == 0 || height == 0 {
      return Ok(result);
    }

    let chunk_dimensions = self.image().chunk_dimensions();
    let chunk_dimensions = (chunk_dimensions.0.min(width), chunk_dimensions.1.min(height));
    if chunk_dimensions.0 == 0 || chunk_dimensions.1 == 0 {
      return Err(TiffError::FormatError(TiffFormatError::InconsistentSizesEncountered));
    }

    let samples = self.image().samples_per_pixel();
    if samples == 0 {
      return Err(TiffError::FormatError(TiffFormatError::InconsistentSizesEncountered));
    }

    let chunks_across = (width - 1) / chunk_dimensions.0 + 1;
    let strip_samples = width * chunk_dimensions.1 * samples;

    let image_chunks = self.image().chunk_offsets.len();
    // For multi-band images, only the first band is read.
    // Possible improvements:
    // * pass requested band as parameter
    // * collect bands to a RGB encoding result in case of RGB bands
    for chunk in 0..image_chunks {
      self.goto_offset_u64(self.image().chunk_offsets[chunk])?;

      let x = chunk % chunks_across;
      let y = chunk / chunks_across;
      let buffer_offset = y * strip_samples + x * chunk_dimensions.0 * samples;
      let byte_order = self.reader.byte_order;
      self.image.expand_chunk(
        &mut self.reader,
        result.as_buffer(buffer_offset).copy(),
        width,
        byte_order,
        chunk,
        &self.limits,
      )?;
    }

    Ok(result)
  }
}
