extern crate tiff;

use std::{fs::File, path::PathBuf};

use tiff::decoder::{Decoder, DecodingResult};

const TEST_IMAGE_DIR: &str = "./tests/images/";

macro_rules! test_image_sum {
  ($name:ident, $buffer:ident, $sum_ty:ty) => {
    fn $name(file: &str, expected_sum: $sum_ty) {
      let path = PathBuf::from(TEST_IMAGE_DIR).join(file);
      let img_file = File::open(path).expect("Cannot find test image!");
      let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");
      let img_res = decoder.read_image().unwrap();

      match img_res {
        DecodingResult::$buffer(res) => {
          let sum: $sum_ty = res.into_iter().map(<$sum_ty>::from).sum();
          assert_eq!(sum, expected_sum);
        }
        _ => panic!("Wrong bit depth"),
      }
    }
  };
}

test_image_sum!(test_image_sum_u8, U8, u64);
test_image_sum!(test_image_sum_i8, I8, i64);
test_image_sum!(test_image_sum_u16, U16, u64);
test_image_sum!(test_image_sum_i16, I16, i64);
test_image_sum!(test_image_sum_u32, U32, u64);
test_image_sum!(test_image_sum_u64, U64, u64);
test_image_sum!(test_image_sum_f32, F32, f32);
test_image_sum!(test_image_sum_f64, F64, f64);

#[test]
fn test_tiled_rgb_u8() {
  test_image_sum_u8("tiled-rgb-u8.tif", 39528948);
}

#[test]
fn test_tiled_rect_rgb_u8() {
  test_image_sum_u8("tiled-rect-rgb-u8.tif", 62081032);
}

#[test]
fn test_tiled_oversize_gray_i8() {
  test_image_sum_i8("tiled-oversize-gray-i8.tif", 1214996);
}

#[test]
fn test_tiled_cmyk_i8() {
  test_image_sum_i8("tiled-cmyk-i8.tif", 1759101);
}

#[test]
fn test_tiled_incremental() {
  let file = "tiled-rgb-u8.tif";
  let sums = [
    188760, 195639, 108148, 81986, 665088, 366140, 705317, 423366, 172033, 324455, 244102, 81853,
    181258, 247971, 129486, 55600, 565625, 422102, 730888, 379271, 232142, 292549, 244045, 86866,
    188141, 115036, 150785, 84389, 353170, 459325, 719619, 329594, 278663, 220474, 243048, 113563,
    189152, 109684, 179391, 122188, 279651, 622093, 724682, 302459, 268428, 204499, 224255, 124674,
    170668, 121868, 192768, 183367, 378029, 585651, 657712, 296790, 241444, 197083, 198429, 134869,
    182318, 86034, 203655, 182338, 297255, 601284, 633813, 242531, 228578, 206441, 193552, 125412,
    181527, 165439, 202531, 159538, 268388, 565790, 611382, 272967, 236497, 215154, 158881, 90806,
    106114, 182342, 191824, 186138, 215174, 393193, 701228, 198866, 227944, 193830, 166330, 49008,
    55719, 122820, 197316, 161969, 203152, 170986, 624427, 188605, 186187, 111064, 115192, 39538,
    48626, 163929, 144682, 135796, 194141, 154198, 584125, 180255, 153524, 121433, 132641, 35743,
    47798, 152343, 162874, 167664, 160175, 133038, 659882, 138339, 166470, 124173, 118929, 51317,
    45267, 155776, 161331, 161006, 130052, 137618, 337291, 106481, 161999, 127343, 87724, 59540,
    63907, 155677, 140668, 141523, 108061, 168657, 186482, 98599, 147614, 139963, 90444, 56602,
    92547, 125644, 134212, 126569, 144153, 179800, 174516, 133969, 129399, 117681, 83305, 55075,
    110737, 115108, 128572, 128911, 130922, 179986, 143288, 145884, 155856, 96683, 94057, 56238,
    79649, 71651, 70182, 75010, 77009, 98855, 78979, 74341, 83482, 53403, 59842, 30305,
  ];

  let path = PathBuf::from(TEST_IMAGE_DIR).join(file);
  let img_file = File::open(path).expect("Cannot find test image!");
  let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");

  let tiles = decoder.tile_count();
  assert_eq!(tiles as usize, sums.len());

  for tile in 0..tiles {
    match decoder.read_chunk(tile).unwrap() {
      DecodingResult::U8(res) => {
        let sum: u64 = res.into_iter().map(<u64>::from).sum();
        assert_eq!(sum, sums[tile as usize]);
      }
      _ => panic!("Wrong bit depth"),
    }
  }
}

#[test]
fn test_div_zero() {
  use tiff::{TiffError, TiffFormatError};

  let image = [
    73, 73, 42, 0, 8, 0, 0, 0, 8, 0, 0, 1, 4, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 40, 1, 0, 0, 0,
    158, 0, 0, 251, 3, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 39, 6, 1, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 17, 1,
    4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 0, 1, 0, 0, 0, 158, 0, 0, 251, 67, 1, 3, 0, 1, 0, 0, 0,
    40, 0, 0, 0, 66, 1, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 178, 178, 178, 178, 178, 178, 178,
  ];

  let err = tiff::decoder::Decoder::new(std::io::Cursor::new(&image)).unwrap_err();

  match err {
    TiffError::FormatError(TiffFormatError::RequiredTileInformationNotFound) => {}
    unexpected => panic!("Unexpected error {}", unexpected),
  }
}

#[test]
fn test_too_many_value_bytes() {
  let image = [
    73, 73, 43, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 8, 0, 0, 0, 23,
    0, 12, 0, 0, 65, 4, 0, 1, 6, 0, 0, 1, 16, 0, 1, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 3,
    0, 1, 0, 0, 0, 1, 0, 0, 0, 59, 73, 84, 186, 202, 83, 240, 66, 1, 53, 22, 56, 47, 0, 0, 0, 0, 0,
    0, 1, 222, 4, 0, 58, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 4, 0, 0, 100, 0, 0, 89, 89, 89, 89,
    89, 89, 89, 89, 96, 1, 20, 89, 89, 89, 89, 18,
  ];

  let error = tiff::decoder::Decoder::new(std::io::Cursor::new(&image)).unwrap_err();

  match error {
    tiff::TiffError::LimitsExceeded => {}
    unexpected => panic!("Unexpected error {}", unexpected),
  }
}

#[test]
fn fuzzer_testcase5() {
  let image = [
    73, 73, 42, 0, 8, 0, 0, 0, 8, 0, 0, 1, 4, 0, 1, 0, 0, 0, 100, 0, 0, 0, 1, 1, 4, 0, 1, 0, 0, 0,
    158, 0, 0, 251, 3, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 6, 1, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 17, 1,
    4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 0, 0, 0, 0, 0, 246, 16, 0, 0, 22, 1, 4, 0, 1, 0, 0, 0,
    40, 0, 251, 255, 23, 1, 4, 0, 1, 0, 0, 0, 48, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178,
  ];

  let _ = tiff::decoder::Decoder::new(std::io::Cursor::new(&image)).unwrap_err();
}

#[test]
fn fuzzer_testcase1() {
  let image = [
    73, 73, 42, 0, 8, 0, 0, 0, 8, 0, 0, 1, 4, 0, 1, 0, 0, 0, 99, 255, 255, 254, 1, 1, 4, 0, 1, 0,
    0, 0, 158, 0, 0, 251, 3, 1, 3, 255, 254, 255, 255, 0, 1, 0, 0, 0, 6, 1, 3, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 17, 1, 4, 0, 9, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 0, 2, 0, 0, 0, 63, 0, 0, 0, 22, 1, 4, 0, 1,
    0, 0, 0, 44, 0, 0, 0, 23, 1, 4, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 1, 0, 178, 178,
  ];

  let _ = tiff::decoder::Decoder::new(std::io::Cursor::new(&image)).unwrap_err();
}

#[test]
fn fuzzer_testcase6() {
  let image = [
    73, 73, 42, 0, 8, 0, 0, 0, 8, 0, 0, 1, 4, 0, 1, 0, 0, 0, 100, 0, 0, 148, 1, 1, 4, 0, 1, 0, 0,
    0, 158, 0, 0, 251, 3, 1, 3, 255, 254, 255, 255, 0, 1, 0, 0, 0, 6, 1, 3, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 17, 1, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 0, 2, 0, 0, 0, 63, 0, 0, 0, 22, 1, 4, 0, 1, 0,
    0, 0, 44, 0, 248, 255, 23, 1, 4, 0, 1, 0, 0, 0, 178, 178, 178, 0, 1, 178, 178, 178,
  ];

  let _ = tiff::decoder::Decoder::new(std::io::Cursor::new(&image)).unwrap_err();
}

#[test]
fn oom() {
  let image = [
    73, 73, 42, 0, 8, 0, 0, 0, 8, 0, 0, 1, 4, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 40, 1, 0, 0, 0,
    158, 0, 0, 251, 3, 1, 3, 0, 1, 0, 0, 0, 7, 0, 0, 0, 6, 1, 3, 0, 1, 0, 0, 0, 2, 0, 0, 0, 17, 1,
    4, 0, 1, 0, 0, 0, 3, 77, 0, 0, 1, 1, 3, 0, 1, 0, 0, 0, 3, 128, 0, 0, 22, 1, 4, 0, 1, 0, 0, 0,
    40, 0, 0, 0, 23, 1, 4, 0, 1, 0, 0, 0, 178, 48, 178, 178, 178, 178, 162, 178,
  ];

  let _ = tiff::decoder::Decoder::new(std::io::Cursor::new(&image)).unwrap_err();
}

#[test]
fn fuzzer_testcase4() {
  let image = [
    73, 73, 42, 0, 8, 0, 0, 0, 8, 0, 0, 1, 4, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 40, 1, 0, 0, 0,
    158, 0, 0, 251, 3, 1, 3, 0, 1, 0, 0, 0, 5, 0, 0, 0, 6, 1, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 17, 1,
    4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 0, 1, 0, 0, 0, 3, 128, 0, 0, 22, 1, 4, 0, 1, 0, 0, 0,
    40, 0, 0, 0, 23, 1, 4, 0, 1, 0, 0, 0, 48, 178, 178, 178, 0, 1, 0, 13, 13,
  ];

  let _ = tiff::decoder::Decoder::new(std::io::Cursor::new(&image)).unwrap_err();
}

#[test]
fn fuzzer_testcase2() {
  let image = [
    73, 73, 42, 0, 8, 0, 0, 0, 15, 0, 0, 254, 44, 1, 0, 0, 0, 0, 0, 32, 0, 0, 0, 1, 4, 0, 1, 0, 0,
    0, 0, 1, 0, 0, 91, 1, 1, 0, 0, 0, 0, 0, 242, 4, 0, 0, 0, 22, 0, 56, 77, 0, 77, 1, 0, 0, 73, 42,
    0, 1, 4, 0, 1, 0, 0, 0, 4, 0, 8, 0, 0, 1, 4, 0, 1, 0, 0, 0, 158, 0, 0, 251, 3, 1, 3, 0, 1, 0,
    0, 0, 7, 0, 0, 0, 6, 1, 3, 0, 1, 0, 0, 0, 2, 0, 0, 0, 17, 1, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 3, 0, 1, 0, 0, 0, 0, 0, 0, 4, 61, 1, 18, 0, 1, 0, 0, 0, 202, 0, 0, 0, 17, 1, 100, 0, 129, 0,
    0, 0, 0, 0, 0, 0, 232, 254, 252, 255, 254, 255, 255, 255, 1, 29, 0, 0, 22, 1, 3, 0, 1, 0, 0, 0,
    16, 0, 0, 0, 23, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 123, 73, 254, 0, 73,
  ];

  let _ = tiff::decoder::Decoder::new(std::io::Cursor::new(&image)).unwrap_err();
}

#[test]
fn fuzzer_testcase3() {
  let image = [
    73, 73, 42, 0, 8, 0, 0, 0, 8, 0, 0, 1, 4, 0, 1, 0, 0, 0, 2, 0, 0, 0, 61, 1, 9, 0, 46, 22, 128,
    0, 0, 0, 0, 1, 6, 1, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 17, 1, 4, 0, 27, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    3, 0, 1, 0, 0, 0, 17, 1, 0, 231, 22, 1, 1, 0, 1, 0, 0, 0, 130, 0, 0, 0, 23, 1, 4, 0, 14, 0, 0,
    0, 0, 0, 0, 0, 133, 133, 133, 77, 77, 77, 0, 0, 22, 128, 0, 255, 255, 255, 255, 255,
  ];

  let _ = tiff::decoder::Decoder::new(std::io::Cursor::new(&image)).unwrap_err();
}

#[test]
fn timeout() {
  use tiff::{TiffError, TiffFormatError};

  let image = [
    73, 73, 42, 0, 8, 0, 0, 0, 16, 0, 254, 0, 4, 0, 1, 68, 0, 0, 0, 2, 0, 32, 254, 252, 0, 109, 0,
    129, 0, 0, 0, 32, 0, 58, 0, 1, 4, 0, 1, 0, 6, 0, 0, 0, 8, 0, 0, 1, 73, 73, 42, 0, 8, 0, 0, 0,
    8, 0, 0, 1, 4, 0, 1, 0, 0, 0, 21, 0, 0, 0, 61, 1, 255, 128, 9, 0, 0, 8, 0, 1, 113, 2, 3, 1, 3,
    0, 1, 0, 0, 0, 5, 0, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 112, 0, 0, 36, 0, 0, 0, 112, 56,
    200, 0, 5, 0, 0, 64, 0, 0, 1, 0, 4, 0, 0, 0, 2, 0, 6, 1, 3, 0, 1, 0, 0, 0, 0, 0, 0, 4, 17, 1,
    1, 0, 93, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 3, 6, 0, 231, 22, 1, 1, 0, 1, 0, 0, 0,
    2, 64, 118, 36, 23, 1, 1, 0, 43, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 4, 0, 8, 0, 0, 73, 73, 42, 0, 8,
    0, 0, 0, 0, 0, 32,
  ];

  let error = tiff::decoder::Decoder::new(std::io::Cursor::new(&image)).unwrap_err();

  match error {
    TiffError::FormatError(TiffFormatError::CycleInOffsets) => {}
    e => panic!("Unexpected error {:?}", e),
  }
}
