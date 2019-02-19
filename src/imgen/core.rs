pub use cv::core::Scalar;
pub use cv::highgui::{Show, WindowFlag, highgui_named_window};

use crate::imgen::color::SampleSpace;
use crate::imgen::error::Error;
use crate::imgen::math::{Point2u, Point3f};
use cv::core::{CvType, FromBytes, LineType, Point2i, Rect, Size2i};
use cv::imgcodecs::ImageReadMode;
use cv::imgproc::{ColorConversion, InterpolationFlag};
use cv::mat::Mat;
use std::f64::consts::PI;
use std::iter::Iterator;
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

#[derive(Debug)]
struct ImageDistance<'a> {
    pub image: &'a Image,
    pub dist: f64,
}

impl<'a> ImageDistance<'a> {
    pub fn mean<'b>(original: &Image, replacement: &'b Image) -> ImageDistance<'b> {
        let repl_lab = replacement.data.cvt_color(ColorConversion::RGB2Lab);
        let orig_lab = original.data.cvt_color(ColorConversion::RGB2Lab);
        let lm_repl = ImageDistance::lab_means(Image::from_mat(repl_lab));
        let lm_orig = ImageDistance::lab_means(Image::from_mat(orig_lab));
        let dist = lm_repl.euclid_dist(lm_orig) as f64;

        ImageDistance { image: replacement, dist }
    }

    fn lab_means(lab_img: Image) -> Point3f {

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
        Point3f::from_arr(&means).unwrap()
    }

    pub fn force_dist(&mut self, dist: f64) {
        self.dist = dist;
    }
}


impl Image {
    pub fn resize_self(&mut self, rows: u32, cols: u32) {
        self.data = self.data.resize_to(Size2i::new(cols as i32, rows as i32), 
                                        InterpolationFlag::InterCubic);

    }

    /* TODO: replace section */
    pub fn replace_section(&mut self, x_low: u32, y_low: u32, new_section: Image) {
        for c in 0..3 {
            for y in 0..new_section.rows() {
                for x in 0..new_section.cols() {
                }
            }
        }
    }


    pub fn reproduce(&mut self, section_size: Point2u, n_images: u32) -> Result<Image, Error> {
        if section_size.x > self.data.cols as u32 || 
           section_size.y > self.data.rows as u32 {
            return Err(Error::new("Invalid sub section dims"));
        }
        else if n_images == 0 {
            return Err(Error::new("Must request at least 1 image"));
        }
        else if n_images > 500 {
            return Err(Error::new("Too many images requested"));
        }

        /* Generate sub images */
        let mut pearls: Vec<Image> = Vec::with_capacity(n_images as usize);
        let sample_space= SampleSpace::new(n_images as i32);

        for (i, color) in sample_space.colors.iter().enumerate() {
            pearls.push(Image::new(section_size.y as i32, 
                                   section_size.x as i32));
            pearls[i].pearl(color.clone(), Scalar::new(255, 255, 255, 255));
        }

            
        let rem_x = self.data.rows as u32 % section_size.x;
        let rem_y = self.data.cols as u32% section_size.y;

        self.resize_self(self.data.rows as u32 - rem_y, 
                         self.data.cols as u32 - rem_x);

        let mut result = Image{ data: self.data.clone() };
        let mut opt_img = ImageDistance::mean(self, self);
        opt_img.force_dist(99999999999999999999f64);

        for y in (0..result.rows() as usize).step_by(section_size.y as usize) {
            for x in (0..result.cols() as usize).step_by(section_size.x as usize) {

                let sub_img = self.subsection(x as i32..x as i32 + section_size.x as i32, 
                                              y as i32..y as i32 + section_size.y as i32)
                    .expect("Subsection boundaries out of range");
                
                for pearl in &pearls {
                    let im_dist = ImageDistance::mean(&sub_img, &pearl);
                    if im_dist.dist < opt_img.dist {
                        opt_img = im_dist;
                    }
                }
                // TODO: insert into result image (from opt_img.image)
            }
        }


        // TODO: change to actual image
        Ok(Image::from_mat(self.data.clone()))
    }

}


