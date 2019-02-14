pub use cv::core::Scalar;
pub use cv::highgui::{Show, WindowFlag, highgui_named_window};

use crate::imgen::error::Error;
use crate::imgen::color::Lab;
use cv::core::{CvType, FromBytes, LineType, Point2i, Rect, Size2i};
use cv::imgcodecs::ImageReadMode;
use cv::imgproc::{ColorConversion, InterpolationFlag};
use cv::mat::Mat;
use std::f64::consts::PI;
use std::ops::Range;

#[derive(Debug)]
pub struct Image {
    pub data: Mat,
}

impl Image {
    pub fn new(rows: i32, cols: i32) -> Image {
        let data = Mat::with_size(rows, cols, CvType::Cv8UC3 as i32);
        Image{ data }
    }

    pub fn from_file(path: &str) -> Result<Image, Error> {
        let data = Mat::from_path(path, ImageReadMode::Color);

        let data = match data {
            Ok(data) => data,
            Err(_error) => {
                return Err(Error::new("Could not find file"));
            },
        };

        if !data.is_valid() {
            Err(Error::new("Invalid image"))
        }
        else {
            Ok(Image{data})
        }
    }

    pub fn from_mat(data: Mat) -> Image {
        Image{ data }
    }

    fn rectangle(&self, height: i32, width: i32, color: Scalar) {
        let rect = Rect::new(0, 0, width, height);

        self.data.rectangle_custom(rect, color, height, LineType::Filled);
    }

    fn circle(&self, radius: i32, color: Scalar) {
        let center = Point2i::new(self.data.cols / 2, self.data.rows / 2);

        self.data.ellipse_custom(center, 
                                 Size2i::new(0,0),
                                 0.0,
                                 0.0,
                                 2.0*PI,
                                 color,
                                 2*radius,
                                 LineType::Filled,
                                 0);
    }

    pub fn pearl(&self, fg_color: Scalar, bg_color: Scalar) {
        let outer_radius = self.data.rows / 2;
        let inner_radius = self.data.rows / 5;
        self.rectangle(self.data.rows, self.data.cols, bg_color);
        self.circle(outer_radius, fg_color);
        self.circle(inner_radius, bg_color);
    }

    pub fn rows(&self) -> i32 {
        self.data.rows
    }

    pub fn cols(&self) -> i32 {
        self.data.cols
    }

    fn at(&self, x_idx: i32, y_idx: i32, z_idx: i32) -> u8 {
        self.data.at3::<u8>(x_idx, y_idx, z_idx)
    }

    pub fn show(&self, name: &str, delay: i32) -> Result<(), Error> {
        match self.data.show(name, delay) {
            Ok(()) => Ok(()),
            Err(_error) => {
                Err(Error::new("Could not show image"))
            },
        }
    }

    fn to_lab(&self) -> Image {
        let data = self.data.cvt_color(ColorConversion::RGB2Lab);
        Image{ data }
    }

    pub fn lab_means(&self) -> Lab {
        let lab_img = self.to_lab();
        let mut result: f64 = 0.0;
        let mut means: [f64; 3] = [0.0; 3];
        let pixels = lab_img.rows() * lab_img.cols();

        for c in 0usize..3 {
            for y in 0..lab_img.data.rows {
                for x in 0..lab_img.data.cols {
                    result += lab_img.at(y, x, c as i32) as f64;
                }
            }
            means[c] = result / pixels as f64;
            result = 0.0;
        }
        Lab::from_arr(&means).unwrap()
    }

    pub fn subsection(&self, x_range: Range<i32>, y_range: Range<i32>) -> Result<Image, Error> {
        if y_range.end <= y_range.start || x_range.end <= x_range.start {
            Err(Error::new("Range must be of length >= 1 and increasing"))
        }
        else {
            let rows = y_range.end - y_range.start;
            let cols = x_range.end - x_range.start;

            let rect = Rect::new(x_range.start, y_range.start, rows, cols);
            let data = self.data.roi(rect);

            Ok(Image::from_mat(data))
        }
    }

    pub fn resize(&self, cols: i32, rows: i32) -> Image {
        let data = self.data.resize_to(Size2i::new(cols, rows), 
                                       InterpolationFlag::InterCubic);
        Image{ data }
    }
}
