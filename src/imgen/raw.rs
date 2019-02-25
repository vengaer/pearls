extern crate libc;

use crate::imgen::error::Error;
use cv::core::CvType;
use cv::mat::Mat;
use libc::{c_char, c_int, c_uchar};
use std::ffi::CString;
use std::os::raw::c_double;

#[repr(C)]
pub struct SSIM {
    pub b: c_double,
    pub g: c_double,
    pub r: c_double,
}

extern "C" {
    fn cv_write(path: *const c_char, data: *const c_uchar, rows: c_int, cols: c_int, im_type: c_int) -> c_int;
    fn cv_ssim(im1_data: *const c_uchar, im1_rows: c_int, im1_cols: c_int, im1_type: c_int, 
               im2_data: *const c_uchar, im2_rows: c_int, im2_cols: c_int, im2_type: c_int) -> SSIM;
}

pub fn write(path: &str, img: &Mat) -> Result<(), Error> {
    let outcome: c_int;
    let path = CString::new(path).expect("Invalid string conversion");
    let rows = img.rows;
    let cols = img.cols;
    let im_type = CvType::Cv8UC3 as c_int;

    unsafe {
        outcome = cv_write(path.as_ptr(), img.data().as_ptr(), rows, cols, im_type);
    }

    match outcome {
        0 => Ok(()),
        _ => Err(Error::new("Failed to write file")),
    }
}

#[allow(dead_code)]
pub fn ssim(im1: &Mat, im2: &Mat) -> Result<SSIM, Error> {
    let result: SSIM;
    let im_type = CvType::Cv8UC3 as c_int;
    
    unsafe {
        result = cv_ssim(im1.data().as_ptr(), im1.rows, im1.cols, im_type, 
                         im2.data().as_ptr(), im2.rows, im2.cols, im_type);
    }

    if result.r < 0.0 {
        return Err(Error::new("Error computing SSIM"));
    }

    Ok(result)
}
