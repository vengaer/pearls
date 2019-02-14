pub use cv::core::Scalar;
pub use cv::highgui::{Show, WindowFlag, highgui_named_window};

use cv::core::{CvType,FromBytes,LineType,Point2i,Rect,Size2i};
use cv::imgcodecs::ImageReadMode;
use cv::imgproc::ColorConversion;
use cv::mat::Mat;
use failure::Error;
use num::Float;
use std::f64::consts::PI;
use std::io::Error as IOError;
use std::io::ErrorKind;
use std::vec::{Vec, IntoIter};

#[derive(Debug)]
pub struct Vec3f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3f {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3f {
        Vec3f{ x, y, z }
    }

    pub fn from_vec<U: Float>(vec: Vec<U>) -> Vec3f {
        if vec.len() != 3 {
            panic!("Cannot convert Vec of size {} to Vec3f", vec.len());
        }
        Vec3f{x: num::cast(vec[0]).unwrap(), 
                y: num::cast(vec[1]).unwrap(), 
                z: num::cast(vec[2]).unwrap()}
    }
}

#[derive(Debug)]
pub struct Image {
    pub data: Mat,
}

impl Image {
    pub fn new(rows: i32, cols: i32) -> Image {
        let data = Mat::with_size(rows, cols, CvType::Cv8UC3 as i32);
        Image{ data }
    }

    pub fn from_file(path: &str) -> Image {
        let data = Mat::from_path(path, ImageReadMode::Color).expect("Could not find file");
        

        if !data.is_valid() {
            panic!("Could not read file");
        }

        Image{data}
    }

    pub fn rectangle(&self, height: i32, width: i32, color: Scalar) {
        let rect = Rect::new(0, 0, width, height);

        self.data.rectangle_custom(rect, color, height, LineType::Filled);
    }

    pub fn circle(&self, radius: i32, color: Scalar) {
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

    pub fn at<T: FromBytes>(&self, x_idx: i32, y_idx: i32, z_idx: i32) -> T {
        self.data.at3::<T>(x_idx, y_idx, z_idx)
    }


    pub fn show(&self, name: &str, delay: i32) -> Result<(), Error> {
        self.data.show(name, delay)
    }

    pub fn to_lab(&self) -> Image {
        let data = self.data.cvt_color(ColorConversion::RGB2Lab);
        Image{ data }
    }

    pub fn lab_means(&self) -> Vec3f {
        let lab_img = self.to_lab();
        let mut result: f64 = 0.0;
        let mut means: Vec<f64> = Vec::new();
        let pixels = lab_img.rows() * lab_img.cols();

        for c in 0usize..3 {
            for y in 0..lab_img.data.rows {
                for x in 0..lab_img.data.cols {
                    result += lab_img.at::<u8>(x,y, c as i32) as f64;
                }
            }
            means[c] = result / pixels as f64;
            result = 0.0;
        }
        Vec3f::from_vec(means)
    }
}
