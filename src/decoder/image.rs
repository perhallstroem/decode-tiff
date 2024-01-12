use std::{
  convert::TryFrom,
  io::{self, Read, Seek},
};

use log::trace;

use super::{
  fp_predict_f32, fp_predict_f64,
  ifd::{Directory, Value},
  stream::{ByteOrder, LZWReader, SmartReader},
  tag_reader::TagReader,
  DecodingBuffer, Limits,
};
use crate::{
  tags::{
    CompressionMethod, PhotometricInterpretation, PlanarConfiguration, Predictor, SampleFormat, Tag,
  },
  ColorType, TiffError, TiffFormatError, TiffResult, TiffUnsupportedError,
};

#[derive(Debug)]
/// Computed values useful for tile decoding
pub(crate) struct TileAttributes {
  pub image_width: usize,
  pub image_height: usize,

  pub tile_width: usize,
  pub tile_length: usize,
}

impl TileAttributes {
  pub fn tiles_across(&self) -> usize {
    (self.image_width + self.tile_width - 1) / self.tile_width
  }

  pub fn tiles_down(&self) -> usize {
    (self.image_height + self.tile_length - 1) / self.tile_length
  }

  fn padding_right(&self) -> usize {
    (self.tile_width - self.image_width % self.tile_width) % self.tile_width
  }

  fn padding_down(&self) -> usize {
    (self.tile_length - self.image_height % self.tile_length) % self.tile_length
  }

  pub fn get_padding(&self, tile: usize) -> (usize, usize) {
    let row = tile / self.tiles_across();
    let column = tile % self.tiles_across();

    let padding_right = if column == self.tiles_across() - 1 { self.padding_right() } else { 0 };

    let padding_down = if row == self.tiles_down() - 1 { self.padding_down() } else { 0 };

    (padding_right, padding_down)
  }
}

#[derive(Debug)]
pub(crate) struct Image {
  pub ifd: Option<Directory>,
  pub width: u32,
  pub height: u32,
  pub bits_per_sample: u8,
  #[allow(unused)]
  pub samples: u16,
  pub sample_format: Vec<SampleFormat>,
  pub photometric_interpretation: PhotometricInterpretation,
  pub compression_method: CompressionMethod,
  pub predictor: Predictor,
  pub tile_attributes: Option<TileAttributes>,
  pub chunk_offsets: Vec<u64>,
  pub chunk_bytes: Vec<u64>,
}

impl Image {
  pub fn from_reader<R: Read + Seek>(
    reader: &mut SmartReader<R>, ifd: Directory, limits: &Limits, bigtiff: bool,
  ) -> TiffResult<Image> {
    let mut tag_reader = TagReader { reader, limits, ifd: &ifd, bigtiff };

    let width = tag_reader.require_tag(Tag::ImageWidth)?.into_u32()?;
    let height = tag_reader.require_tag(Tag::ImageLength)?.into_u32()?;
    if width == 0 || height == 0 {
      return Err(TiffError::FormatError(TiffFormatError::InvalidDimensions(width, height)));
    }

    let photometric_interpretation = tag_reader
      .find_tag(Tag::PhotometricInterpretation)?
      .map(Value::into_u16)
      .transpose()?
      .and_then(PhotometricInterpretation::from_u16)
      .ok_or(TiffUnsupportedError::UnknownInterpretation)?;

    // Try to parse both the compression method and the number, format, and bits of the included
    // samples. If they are not explicitly specified, those tags are reset to their default
    // values and not carried from previous images.
    let compression_method = match tag_reader.find_tag(Tag::Compression)? {
      Some(val) => CompressionMethod::from_u16_exhaustive(val.into_u16()?),
      None => CompressionMethod::None,
    };

    let samples: u16 =
      tag_reader.find_tag(Tag::SamplesPerPixel)?.map(Value::into_u16).transpose()?.unwrap_or(1);
    if samples == 0 {
      return Err(TiffFormatError::SamplesPerPixelIsZero.into());
    }

    let sample_format = match tag_reader.find_tag_uint_vec(Tag::SampleFormat)? {
      Some(vals) => {
        let sample_format: Vec<_> =
          vals.into_iter().map(SampleFormat::from_u16_exhaustive).collect();

        eprintln!("SAMPLE FORMATS: {:?}", sample_format);

        // TODO: for now, only homogenous formats across samples are supported.
        if !sample_format.windows(2).all(|s| s[0] == s[1]) {
          return Err(TiffUnsupportedError::UnsupportedSampleFormat(sample_format).into());
        }

        sample_format
      }
      None => vec![SampleFormat::Uint],
    };

    let bits_per_sample: Vec<u8> =
      tag_reader.find_tag_uint_vec(Tag::BitsPerSample)?.unwrap_or_else(|| vec![1]);

    eprintln!("BITS PER SAMPLE: {bits_per_sample:?}");

    if bits_per_sample.len() != samples.into() {
      return Err(TiffError::FormatError(TiffFormatError::InconsistentSizesEncountered));
    }

    // This library (and libtiff) do not support mixed sample formats.
    if bits_per_sample.iter().any(|&b| b != bits_per_sample[0]) {
      return Err(TiffUnsupportedError::InconsistentBitsPerSample(bits_per_sample).into());
    }

    let predictor = tag_reader
      .find_tag(Tag::Predictor)?
      .map(Value::into_u16)
      .transpose()?
      .map(|p| {
        Predictor::from_u16(p).ok_or(TiffError::FormatError(TiffFormatError::UnknownPredictor(p)))
      })
      .transpose()?
      .unwrap_or(Predictor::None);

    let planar_config = tag_reader
      .find_tag(Tag::PlanarConfiguration)?
      .map(Value::into_u16)
      .transpose()?
      .map(|p| {
        PlanarConfiguration::from_u16(p)
          .ok_or(TiffError::FormatError(TiffFormatError::UnknownPlanarConfiguration(p)))
      })
      .transpose()?
      .unwrap_or(PlanarConfiguration::Chunky);

    let planes = match planar_config {
      PlanarConfiguration::Chunky => 1,
    };

    let chunk_offsets;
    let chunk_bytes;
    let tile_attributes;
    match (ifd.contains_key(&Tag::TileByteCounts), ifd.contains_key(&Tag::TileOffsets)) {
      (true, true) => {
        let tile_width = usize::try_from(tag_reader.require_tag(Tag::TileWidth)?.into_u32()?)?;
        let tile_length = usize::try_from(tag_reader.require_tag(Tag::TileLength)?.into_u32()?)?;

        if tile_width == 0 {
          return Err(TiffFormatError::InvalidTagValueType(Tag::TileWidth).into());
        } else if tile_length == 0 {
          return Err(TiffFormatError::InvalidTagValueType(Tag::TileLength).into());
        }

        tile_attributes = Some(TileAttributes {
          image_width: usize::try_from(width)?,
          image_height: usize::try_from(height)?,
          tile_width,
          tile_length,
        });
        chunk_offsets = tag_reader.find_tag(Tag::TileOffsets)?.unwrap().into_u64_vec()?;
        chunk_bytes = tag_reader.find_tag(Tag::TileByteCounts)?.unwrap().into_u64_vec()?;

        let tile = tile_attributes.as_ref().unwrap();
        if chunk_offsets.len() != chunk_bytes.len()
          || chunk_offsets.len() != tile.tiles_down() * tile.tiles_across() * planes as usize
        {
          return Err(TiffError::FormatError(TiffFormatError::InconsistentSizesEncountered));
        }
      }
      (_, _) => {
        return Err(TiffError::FormatError(TiffFormatError::RequiredTileInformationNotFound))
      }
    };

    Ok(Image {
      ifd: Some(ifd),
      width,
      height,
      bits_per_sample: bits_per_sample[0],
      samples,
      sample_format,
      photometric_interpretation,
      compression_method,
      predictor,
      tile_attributes,
      chunk_offsets,
      chunk_bytes,
    })
  }

  pub(crate) fn colortype(&self) -> TiffResult<ColorType> {
    match self.photometric_interpretation {
      PhotometricInterpretation::RGB => match self.samples {
        3 => Ok(ColorType::RGB(self.bits_per_sample)),
        4 => Ok(ColorType::RGBA(self.bits_per_sample)),
        // FIXME: We should _ignore_ other components. In particular:
        // > Beware of extra components. Some TIFF files may have more components per pixel
        // than you think. A Baseline TIFF reader must skip over them gracefully,using the
        // values of the SamplesPerPixel and BitsPerSample fields.
        // > -- TIFF 6.0 Specification, Section 7, Additional Baseline requirements.
        _ => Err(TiffError::UnsupportedError(TiffUnsupportedError::InterpretationWithBits(
          self.photometric_interpretation,
          vec![self.bits_per_sample; self.samples as usize],
        ))),
      },
      PhotometricInterpretation::CMYK => match self.samples {
        4 => Ok(ColorType::CMYK(self.bits_per_sample)),
        _ => Err(TiffError::UnsupportedError(TiffUnsupportedError::InterpretationWithBits(
          self.photometric_interpretation,
          vec![self.bits_per_sample; self.samples as usize],
        ))),
      },
      PhotometricInterpretation::YCbCr => match self.samples {
        3 => Ok(ColorType::YCbCr(self.bits_per_sample)),
        _ => Err(TiffError::UnsupportedError(TiffUnsupportedError::InterpretationWithBits(
          self.photometric_interpretation,
          vec![self.bits_per_sample; self.samples as usize],
        ))),
      },
      PhotometricInterpretation::BlackIsZero | PhotometricInterpretation::WhiteIsZero
        if self.samples == 1 =>
      {
        Ok(ColorType::Gray(self.bits_per_sample))
      }

      // TODO: this is bad we should not fail at this point
      _ => Err(TiffError::UnsupportedError(TiffUnsupportedError::InterpretationWithBits(
        self.photometric_interpretation,
        vec![self.bits_per_sample; self.samples as usize],
      ))),
    }
  }

  fn create_reader<'r, R: 'r + Read>(
    reader: R, compression_method: CompressionMethod, compressed_length: u64,
  ) -> TiffResult<Box<dyn Read + 'r>> {
    Ok(match compression_method {
      CompressionMethod::None => Box::new(reader),
      CompressionMethod::LZW => {
        Box::new(LZWReader::new(reader, usize::try_from(compressed_length)?))
      }
      method => {
        return Err(TiffError::UnsupportedError(
          TiffUnsupportedError::UnsupportedCompressionMethod(method),
        ))
      }
    })
  }

  /// Samples per pixel within chunk.
  ///
  /// In planar config, samples are stored in separate strips/chunks, also called bands.
  ///
  /// Example with `bits_per_sample = [8, 8, 8]` and `PhotometricInterpretation::RGB`:
  /// * `PlanarConfiguration::Chunky` -> 3 (RGBRGBRGB...)
  /// * `PlanarConfiguration::Planar` -> 1 (RRR...) (GGG...) (BBB...)
  pub(crate) fn samples_per_pixel(&self) -> usize {
    self.samples.into()
  }

  pub(crate) fn chunk_file_range(&self, chunk: u32) -> TiffResult<(u64, u64)> {
    let file_offset = self
      .chunk_offsets
      .get(chunk as usize)
      .ok_or(TiffError::FormatError(TiffFormatError::InconsistentSizesEncountered))?;

    let compressed_bytes = self
      .chunk_bytes
      .get(chunk as usize)
      .ok_or(TiffError::FormatError(TiffFormatError::InconsistentSizesEncountered))?;

    Ok((*file_offset, *compressed_bytes))
  }

  pub(crate) fn chunk_dimensions(&self) -> TiffResult<(u32, u32)> {
    let tile_attrs = self.tile_attributes.as_ref().unwrap();
    Ok((u32::try_from(tile_attrs.tile_width)?, u32::try_from(tile_attrs.tile_length)?))
  }

  pub(crate) fn chunk_data_dimensions(&self, chunk_index: u32) -> TiffResult<(u32, u32)> {
    let tile_attrs = self.tile_attributes.as_ref().unwrap();
    let (padding_right, padding_down) = tile_attrs.get_padding(chunk_index as usize);

    let tile_width = tile_attrs.tile_width - padding_right;
    let tile_length = tile_attrs.tile_length - padding_down;

    Ok((u32::try_from(tile_width)?, u32::try_from(tile_length)?))
  }

  pub(crate) fn expand_chunk(
    &self, reader: impl Read, mut buffer: DecodingBuffer, output_width: usize,
    byte_order: ByteOrder, chunk_index: u32, limits: &Limits,
  ) -> TiffResult<()> {
    // Validate that the provided buffer is of the expected type.
    let color_type = self.colortype()?;
    match (color_type, &buffer) {
      (ColorType::RGB(n), _)
      | (ColorType::RGBA(n), _)
      | (ColorType::CMYK(n), _)
      | (ColorType::YCbCr(n), _)
      | (ColorType::Gray(n), _)
        if usize::from(n) == buffer.byte_len() * 8 => {}
      (ColorType::Gray(n), DecodingBuffer::U8(_)) if n < 8 => match self.predictor {
        Predictor::None => {}
        Predictor::Horizontal => {
          return Err(TiffError::UnsupportedError(TiffUnsupportedError::HorizontalPredictor(
            color_type,
          )))
        }
        Predictor::FloatingPoint => {
          return Err(TiffError::UnsupportedError(TiffUnsupportedError::FloatingPointPredictor(
            color_type,
          )));
        }
      },
      (type_, _) => {
        return Err(TiffError::UnsupportedError(TiffUnsupportedError::UnsupportedColorType(type_)))
      }
    }

    // Validate that the predictor is supported for the sample type.
    match (self.predictor, &buffer) {
      (Predictor::Horizontal, DecodingBuffer::F32(_))
      | (Predictor::Horizontal, DecodingBuffer::F64(_)) => {
        return Err(TiffError::UnsupportedError(TiffUnsupportedError::HorizontalPredictor(
          color_type,
        )));
      }
      (Predictor::FloatingPoint, DecodingBuffer::F32(_))
      | (Predictor::FloatingPoint, DecodingBuffer::F64(_)) => {}
      (Predictor::FloatingPoint, _) => {
        return Err(TiffError::UnsupportedError(TiffUnsupportedError::FloatingPointPredictor(
          color_type,
        )));
      }
      _ => {}
    }

    let compressed_bytes = self
      .chunk_bytes
      .get(chunk_index as usize)
      .ok_or(TiffError::FormatError(TiffFormatError::InconsistentSizesEncountered))?;
    if *compressed_bytes > limits.intermediate_buffer_size as u64 {
      return Err(TiffError::LimitsExceeded);
    }

    let byte_len = buffer.byte_len();
    let compression_method = self.compression_method;
    let photometric_interpretation = self.photometric_interpretation;
    let predictor = self.predictor;
    let samples = self.samples_per_pixel();

    let chunk_dims = self.chunk_dimensions()?;
    let data_dims = self.chunk_data_dimensions(chunk_index)?;

    let padding_right = chunk_dims.0 - data_dims.0;

    let mut reader = Self::create_reader(reader, compression_method, *compressed_bytes)?;

    if output_width == data_dims.0 as usize && padding_right == 0 {
      let total_samples = data_dims.0 as usize * data_dims.1 as usize * samples;
      let tile = &mut buffer.as_bytes_mut()[..total_samples * byte_len];
      reader.read_exact(tile)?;

      for row in 0..data_dims.1 as usize {
        let row_start = row * output_width * samples;
        let row_end = (row + 1) * output_width * samples;
        let row = buffer.subrange(row_start..row_end);
        super::fix_endianness_and_predict(row, samples, byte_order, predictor);
      }
      if photometric_interpretation == PhotometricInterpretation::WhiteIsZero {
        super::invert_colors(&mut buffer.subrange(0..total_samples), color_type);
      }
    } else if padding_right > 0 && self.predictor == Predictor::FloatingPoint {
      // The floating point predictor shuffles the padding bytes into the encoded output, so
      // this case is handled specially when needed.
      let mut encoded = vec![0u8; chunk_dims.0 as usize * samples * byte_len];

      for row in 0..data_dims.1 as usize {
        let row_start = row * output_width * samples;
        let row_end = row_start + data_dims.0 as usize * samples;

        reader.read_exact(&mut encoded)?;
        match buffer.subrange(row_start..row_end) {
          DecodingBuffer::F32(buf) => fp_predict_f32(&mut encoded, buf, samples),
          DecodingBuffer::F64(buf) => fp_predict_f64(&mut encoded, buf, samples),
          _ => unreachable!(),
        }
        if photometric_interpretation == PhotometricInterpretation::WhiteIsZero {
          super::invert_colors(&mut buffer.subrange(row_start..row_end), color_type);
        }
      }
    } else {
      for row in 0..data_dims.1 as usize {
        let row_start = row * output_width * samples;
        let row_end = row_start + data_dims.0 as usize * samples;

        let row = &mut buffer.as_bytes_mut()[(row_start * byte_len)..(row_end * byte_len)];
        reader.read_exact(row)?;

        // Skip horizontal padding
        if padding_right > 0 {
          let len = u64::try_from(padding_right as usize * samples * byte_len)?;
          io::copy(&mut reader.by_ref().take(len), &mut io::sink())?;
        }

        let mut row = buffer.subrange(row_start..row_end);
        super::fix_endianness_and_predict(row.copy(), samples, byte_order, predictor);
        if photometric_interpretation == PhotometricInterpretation::WhiteIsZero {
          super::invert_colors(&mut row, color_type);
        }
      }
    }

    Ok(())
  }
}
