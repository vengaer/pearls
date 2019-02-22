extern crate libc;

use crate::imgen::error::Error;
use cv::core::CvType;
use cv::mat::Mat;
use libc::{c_char, c_int, c_uchar};
use std::ffi::CString;

extern "C" {
    fn cv_write(path: *const c_char, data: *const c_uchar, rows: c_int, cols: c_int, im_type: c_int) -> c_int;
}

pub fn write(path: &str, data: &Mat) -> Result<(), Error> {
    let outcome: c_int;
    let path = CString::new(path).expect("Invalid string conversion");
    let rows = data.rows;
    let cols = data.cols;
    let im_type = CvType::Cv8UC3 as c_int;

    unsafe {
        outcome = cv_write(path.as_ptr(), data.data().as_ptr(), rows, cols, im_type);
    }

    match outcome {
        0 => Ok(()),
        _ => Err(Error::new("Failed to write file")),
    }
}

