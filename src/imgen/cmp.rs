use crate::imgen::{math, raw};
use crate::imgen::core::{Image, PearlImage};
use crate::imgen::error::Error;
use crate::imgen::math::Point3f;
use crate::imgen::raw::SSIM;
use cv::imgproc::ColorConversion;
use libc::c_int;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Weights {
    pub eab: f32,
    pub ssim: f32,
}

impl Weights {
    #[allow(dead_code)]
    pub fn new(eab: f32, ssim: f32) -> Result<Weights, Error> {
        if !math::approx_eq(eab + ssim , 1.0) {
            return Err(Error::new("Weights must sum to 1"));
        }
        Ok(Weights{ eab, ssim })
    }
}

pub fn mean_distance(orig: &Image, repl: &PearlImage) -> f32 {
    let orig_lab = orig.data.cvt_color(ColorConversion::RGB2Lab);
    let lm_orig = lab_means(&Image::from_mat(orig_lab));

    lm_orig.euclid_dist(&repl.lab_means)
}

pub fn lab_means(lab_img: &Image) -> Point3f {
    let mut result: f64 = 0.0;
    let mut means: [f64; 3] = [0.0; 3];
    let pixels = lab_img.rows() * lab_img.cols();

    for c in 0usize..3 {
        for y in 0..lab_img.data.rows {
            for x in 0..lab_img.data.cols {
                result += lab_img.at(x, y, c as c_int) as f64;
            }
        }
        means[c] = result / pixels as f64;
        result = 0.0;
    }
    Point3f::from_arr(&means).unwrap()
}

#[allow(dead_code)]
pub fn ssim(im1: &Image, im2: &Image) -> Result<SSIM, Error> {
    if im1.data.rows == im2.data.rows &&
       im1.data.cols == im2.data.cols {
        return raw::ssim(&im1.data, &im2.data);
    }

    let mut to_scale: Image;
    let reference: &Image;
    match im1.data.rows < im2.data.rows {
        true => {
            if im1.data.cols > im2.data.cols {
                return Err(Error::new("Unsuitable dimensions for SSIM"));
            }
            to_scale = im1.clone();
            reference = im2;
        },
        false => {
            if im1.data.cols < im2.data.cols {
                return Err(Error::new("Unsuitable dimensions for SSIM"));
            }
            to_scale = im2.clone();
            reference = im1;
        },
    };

    to_scale.resize(reference.data.rows as u32, reference.data.cols as u32);
    
    raw::ssim(&to_scale.data, &reference.data)
}

#[allow(dead_code)]
pub fn ssim_mean(im1: &Image, im2: &Image) -> Result<f32, Error> {
    let ssim = ssim(&im1, &im2)?;

    Ok((ssim.r + ssim.g + ssim.b) as f32 / 3.0)
}
