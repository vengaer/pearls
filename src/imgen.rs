use cv::core::{CvType,FromBytes,LineType,Point2i,Rect,Scalar,Size2i};
use cv::highgui::Show;
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
pub struct Point3f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point3f {
    pub fn new(x: f32, y: f32, z: f32) -> Point3f {
        Point3f{ x, y, z }
    }

    pub fn from_vec<U: Float>(vec: Vec<U>) -> Point3f {
        if vec.len() != 3 {
            panic!("Cannot convert Vec of size {} to Point3f", vec.len());
        }
        Point3f{x: num::cast(vec[0]).unwrap(), 
                y: num::cast(vec[1]).unwrap(), 
                z: num::cast(vec[2]).unwrap()}
    }
}

#[derive(Debug)]
pub struct Point3i {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Point3i {
    pub fn new(x: i32, y: i32, z: i32) -> Point3i {
        Point3i{ x, y, z }
    }

    pub fn from_vec(vec: Vec<i32>) -> Point3i {
        if vec.len() != 3 {
            panic!("Cannot convert Vec of size {} to Point3i", vec.len());
        }
        Point3i{x: vec[0], 
                y: vec[1], 
                z: vec[2]}
    }

}

#[derive(Debug)]
enum ColorCombination {
    R,
    G,
    B,
    RG,
    RB,
    GB,
    RGB,
}

impl ColorCombination {
    fn value(&self) -> i32 {
        match *self {
            ColorCombination::R   => 0,
            ColorCombination::G   => 1,
            ColorCombination::B   => 2,
            ColorCombination::RG  => 3,
            ColorCombination::RB  => 4,
            ColorCombination::GB  => 5,
            ColorCombination::RGB => 6,
        }
    }

    fn from_i32(value: i32) -> ColorCombination {
        match value {
             0 => ColorCombination::R,
             1 => ColorCombination::G,
             2 => ColorCombination::B,
             3 => ColorCombination::RG,
             4 => ColorCombination::RB,
             5 => ColorCombination::GB,
             6 => ColorCombination::RGB,
             _ => panic!("Value out of range"),
        }
    }
}

fn assign_color(intensity: i32, channels: ColorCombination) -> Scalar {
    match channels {
        ColorCombination::R   => {
            Scalar::new(0, 0, intensity, 255)
        },
        ColorCombination::G   => { 
            Scalar::new(0, intensity, 0, 255)
        },
        ColorCombination::B   => {
            Scalar::new(intensity, 0, 0, 255)
        },
        ColorCombination::RG  => {
             Scalar::new(0, intensity, intensity, 255)
        },
        ColorCombination::RB  => {
             Scalar::new(intensity, 0, intensity, 255)
        },
        ColorCombination::GB  => {
             Scalar::new(intensity, intensity, 0, 255)
        },
        ColorCombination::RGB => {
             Scalar::new(intensity, intensity, intensity, 255)
        },
    }
}

pub struct ColorSpace {
    pub colors: Vec<Scalar>,
}

impl ColorSpace {
    pub fn new(samples: i32) -> ColorSpace {
        if samples < 7 {
            panic!("Too few color samples requested");
        }

        let size = samples + (7 - (samples % 7)) + 1; /* Round up to multiple of 7 
                                                         + 1 for black */

        let mut space = ColorSpace{ colors: Vec::with_capacity(size as usize) };
        space.colors.push(Scalar::new(0,0,0,255));
        let per_color = size / 7;

        let intensity_step = 255 / per_color; 

        let mut intensity = intensity_step;

        let mut channels = -1;

        for i in 0..size-1 {
            if i % per_color == 0 { /* Next channel(s) */
                channels += 1;
                intensity = intensity_step;
            }
            else {
                intensity += intensity_step;
            }
            space.colors.push(assign_color(intensity, ColorCombination::from_i32(channels)));
        }

        space
    }

    pub fn debug(&self) {
        for col in &self.colors {
            println!("{:?}", col);
        }
    }
}

impl IntoIterator for ColorSpace {
    type Item = Scalar;
    type IntoIter = ::std::vec::IntoIter<Scalar>;

    fn into_iter(self) -> Self::IntoIter {
        self.colors.into_iter()
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

    pub fn from_file(path: &str, mode: ImageReadMode) -> Image {
        let data = Mat::from_path(path, mode).expect("Could not find file");
        

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

    pub fn lab_means(&self) -> Point3f {
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
        Point3f::from_vec(means)
    }
}
