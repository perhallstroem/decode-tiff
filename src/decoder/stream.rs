//! All IO functionality needed for TIFF decoding

use std::{
  convert::TryFrom,
  io::{self, BufRead, BufReader, Read, Seek, Take},
};

/// Byte order of the TIFF file.
#[derive(Clone, Copy, Debug)]
pub enum ByteOrder {
  /// little endian byte order
  LittleEndian,
  /// big endian byte order
  BigEndian,
}

/// Reader that is aware of the byte order.
pub trait EndianReader: Read {
  /// Byte order that should be adhered to
  fn byte_order(&self) -> ByteOrder;

  /// Reads an u16
  #[inline(always)]
  fn read_u16(&mut self) -> Result<u16, io::Error> {
    let mut n = [0u8; 2];
    self.read_exact(&mut n)?;
    Ok(match self.byte_order() {
      ByteOrder::LittleEndian => u16::from_le_bytes(n),
      ByteOrder::BigEndian => u16::from_be_bytes(n),
    })
  }

  /// Reads an i8
  #[inline(always)]
  fn read_i8(&mut self) -> Result<i8, io::Error> {
    let mut n = [0u8; 1];
    self.read_exact(&mut n)?;
    Ok(match self.byte_order() {
      ByteOrder::LittleEndian => i8::from_le_bytes(n),
      ByteOrder::BigEndian => i8::from_be_bytes(n),
    })
  }

  /// Reads an i16
  #[inline(always)]
  fn read_i16(&mut self) -> Result<i16, io::Error> {
    let mut n = [0u8; 2];
    self.read_exact(&mut n)?;
    Ok(match self.byte_order() {
      ByteOrder::LittleEndian => i16::from_le_bytes(n),
      ByteOrder::BigEndian => i16::from_be_bytes(n),
    })
  }

  /// Reads an u32
  #[inline(always)]
  fn read_u32(&mut self) -> Result<u32, io::Error> {
    let mut n = [0u8; 4];
    self.read_exact(&mut n)?;
    Ok(match self.byte_order() {
      ByteOrder::LittleEndian => u32::from_le_bytes(n),
      ByteOrder::BigEndian => u32::from_be_bytes(n),
    })
  }

  /// Reads an i32
  #[inline(always)]
  fn read_i32(&mut self) -> Result<i32, io::Error> {
    let mut n = [0u8; 4];
    self.read_exact(&mut n)?;
    Ok(match self.byte_order() {
      ByteOrder::LittleEndian => i32::from_le_bytes(n),
      ByteOrder::BigEndian => i32::from_be_bytes(n),
    })
  }

  /// Reads an u64
  #[inline(always)]
  fn read_u64(&mut self) -> Result<u64, io::Error> {
    let mut n = [0u8; 8];
    self.read_exact(&mut n)?;
    Ok(match self.byte_order() {
      ByteOrder::LittleEndian => u64::from_le_bytes(n),
      ByteOrder::BigEndian => u64::from_be_bytes(n),
    })
  }

  /// Reads an i64
  #[inline(always)]
  fn read_i64(&mut self) -> Result<i64, io::Error> {
    let mut n = [0u8; 8];
    self.read_exact(&mut n)?;
    Ok(match self.byte_order() {
      ByteOrder::LittleEndian => i64::from_le_bytes(n),
      ByteOrder::BigEndian => i64::from_be_bytes(n),
    })
  }

  /// Reads an f32
  #[inline(always)]
  fn read_f32(&mut self) -> Result<f32, io::Error> {
    let mut n = [0u8; 4];
    self.read_exact(&mut n)?;
    Ok(f32::from_bits(match self.byte_order() {
      ByteOrder::LittleEndian => u32::from_le_bytes(n),
      ByteOrder::BigEndian => u32::from_be_bytes(n),
    }))
  }

  /// Reads an f64
  #[inline(always)]
  fn read_f64(&mut self) -> Result<f64, io::Error> {
    let mut n = [0u8; 8];
    self.read_exact(&mut n)?;
    Ok(f64::from_bits(match self.byte_order() {
      ByteOrder::LittleEndian => u64::from_le_bytes(n),
      ByteOrder::BigEndian => u64::from_be_bytes(n),
    }))
  }
}

///
/// # READERS

///
/// ## LZW Reader

/// Reader that decompresses LZW streams
pub struct LZWReader<R: Read> {
  reader: BufReader<Take<R>>,
  decoder: weezl::decode::Decoder,
}

impl<R: Read> LZWReader<R> {
  /// Wraps a reader
  pub fn new(reader: R, compressed_length: usize) -> LZWReader<R> {
    Self {
      reader: BufReader::with_capacity(
        (32 * 1024).min(compressed_length),
        reader.take(u64::try_from(compressed_length).unwrap()),
      ),
      decoder: weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8),
    }
  }
}

impl<R: Read> Read for LZWReader<R> {
  fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
    loop {
      let result = self.decoder.decode_bytes(self.reader.fill_buf()?, buf);
      self.reader.consume(result.consumed_in);

      match result.status {
        Ok(weezl::LzwStatus::Ok) => {
          if result.consumed_out == 0 {
            continue;
          } else {
            return Ok(result.consumed_out);
          }
        }
        Ok(weezl::LzwStatus::NoProgress) => {
          assert_eq!(result.consumed_in, 0);
          assert_eq!(result.consumed_out, 0);
          assert!(self.reader.buffer().is_empty());
          return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "no lzw end code found"));
        }
        Ok(weezl::LzwStatus::Done) => {
          return Ok(result.consumed_out);
        }
        Err(err) => return Err(io::Error::new(io::ErrorKind::InvalidData, err)),
      }
    }
  }
}

///
/// ## SmartReader Reader

/// Reader that is aware of the byte order.
#[derive(Debug)]
pub struct SmartReader<R> {
  reader: R,
  pub byte_order: ByteOrder,
}

impl<R> SmartReader<R>
where
  R: Read,
{
  /// Wraps a reader
  pub fn wrap(reader: R, byte_order: ByteOrder) -> SmartReader<R> {
    SmartReader { reader, byte_order }
  }

  pub fn into_inner(self) -> R {
    self.reader
  }
}
impl<R: Read + Seek> SmartReader<R> {
  pub fn goto_offset(&mut self, offset: u64) -> io::Result<()> {
    self.seek(io::SeekFrom::Start(offset)).map(|_| ())
  }
}

impl<R> EndianReader for SmartReader<R>
where
  R: Read,
{
  #[inline(always)]
  fn byte_order(&self) -> ByteOrder {
    self.byte_order
  }
}

impl<R: Read> Read for SmartReader<R> {
  #[inline]
  fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
    self.reader.read(buf)
  }
}

impl<R: Read + Seek> Seek for SmartReader<R> {
  #[inline]
  fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
    self.reader.seek(pos)
  }
}
