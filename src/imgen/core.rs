pub use crate::imgen::raw::SSIM;
pub use cv::core::Scalar;

use crate::imgen::{cmp, math};
use crate::imgen::cmp::Weights;
use crate::imgen::color::SampleSpace;
use crate::imgen::error::Error;
use crate::imgen::math::{DensityEstimate, Point2u, Point3f, Size2u};
use crate::imgen::raw;
use cv::core::{CvType, LineType, Point2i, Rect, Size2i};
use cv::highgui::{Show, WindowFlag, highgui_named_window};
use cv::imgcodecs::ImageReadMode;
use cv::imgproc::{ColorConversion, InterpolationFlag};
use cv::mat::Mat;
use libc::c_int;
use std::cmp as scmp;
use std::f64::consts::PI;
use std::iter::Iterator;
use std::ops::Range;
use std::sync::{Arc, Mutex};


#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Image {
    pub data: Mat,
}

unsafe impl Sync for Image { }

impl Image {
    pub fn new(rows: c_int, cols: c_int) -> Image {
        let data = Mat::with_size(rows, cols, CvType::Cv8UC3 as c_int);
        Image{ data }
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn to_file(&self, path: &str) -> Result<(), Error> {
        raw::write(&path, &self.data)
    }

    pub fn from_mat(data: Mat) -> Image {
        Image{ data }
    }

    pub fn rectangle(&self, height: c_int, width: c_int, color: Scalar) {
        let rect = Rect::new(0, 0, width, height);

        self.data.rectangle_custom(rect, color, 2*height, LineType::Filled);
    }

    pub fn rectangle_custom(&self, height: c_int, width: c_int, first_row_nr: c_int, first_col_nr: c_int, color: Scalar) {
        let rect = Rect::new(first_col_nr, first_row_nr, width, height);

        self.data.rectangle_custom(rect, color, -20, LineType::Line8);
    }

    pub fn circle(&self, radius: c_int, color: Scalar) {
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

    pub fn circle_custom(&self, radius: c_int, color: Scalar, center: &Point2u) {
        let center = Point2i::new(center.x as c_int, center.y as c_int);

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

    pub fn rows(&self) -> c_int {
        self.data.rows
    }

    pub fn cols(&self) -> c_int {
        self.data.cols
    }

    pub fn at(&self, x_idx: c_int, y_idx: c_int, z_idx: c_int) -> u8 {
        self.data.at3::<u8>(y_idx, x_idx, z_idx)
    }

    pub fn subsection(&self, x_range: &Range<c_int>, y_range: &Range<c_int>) -> Result<Image, Error> {

        if y_range.end <= y_range.start || x_range.end <= x_range.start {
            return Err(Error::new("Range must be of length >= 1 and increasing"))
        }
        else if y_range.end - 1 > self.data.rows || x_range.end - 1 > self.data.cols {
            println!("x_range: {}-{}, y_range: {}-{}", x_range.start, x_range.end, y_range.start, y_range.end);
            println!("Dims: {}x{}", self.data.cols, self.data.rows);
            return Err(Error::new("Range out of bounds"));
        }

        let rows = y_range.end - y_range.start;
        let cols = x_range.end - x_range.start;

        let rect = Rect::new(x_range.start, y_range.start, cols, rows);
        let data = self.data.roi(rect);

        Ok(Image::from_mat(data))
    }

}

#[derive(Debug, Clone)]
pub struct PearlImage {
    pub image: Image,
    outer_radius: u32,
    inner_radius: u32,
    color: Scalar,
    pub lab_means: Point3f,
}

impl PearlImage {
    #[allow(dead_code)]
    pub fn new(fg_color: Scalar, image_dims: &Size2u) -> PearlImage {
        let inner_radius = image_dims.y as u32 / 4;

        PearlImage::with_inner_radius(fg_color, &image_dims, inner_radius).unwrap()
    }

    pub fn with_inner_radius(fg_color: Scalar, image_dims: &Size2u, inner_radius: u32) -> Result<PearlImage, Error> {
        let image = Image::new(image_dims.y as c_int, image_dims.x as c_int);

        let outer_radius = image.data.rows as u32 / 2;

        if inner_radius > outer_radius - 1 {
            return Err(Error::new("Inner radius larger than outer"));
        }

        image.rectangle(image.data.rows, image.data.cols, Scalar::all(255));
        image.circle(outer_radius as c_int, fg_color);
        image.circle(inner_radius as c_int, Scalar::all(255));

        let lab_img = image.data.cvt_color(ColorConversion::RGB2Lab);
        let lab_means = cmp::lab_means(&Image::from_mat(lab_img));

        Ok(PearlImage{ image, outer_radius, inner_radius, color: fg_color, lab_means })
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum ExecutionPolicy {
    Sequential,
    Parallellx4,
    Parallellx8,
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum Filter {
    None,
    Sdev,
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum PostProc {
    None,
    OrigSize,
}

impl Image {
    pub fn resize(&mut self, rows: u32, cols: u32) {
        self.data = self.data.resize_to(Size2i::new(cols as c_int, rows as c_int), 
                                        InterpolationFlag::InterLanczos4);
    }

    #[allow(dead_code)]
    pub fn resize_by(&mut self, row_fact: f32, col_fact: f32) {
        let nrows = self.data.rows as f32 * row_fact;
        let ncols = self.data.cols as f32 * col_fact;

        self.data = self.data.resize_to(Size2i::new(ncols as c_int, nrows as c_int),
                                        InterpolationFlag::InterLanczos4);

    }

    #[allow(dead_code)]
    pub fn clamp_size(&mut self, max: u32, min: u32) {
        let max = max as c_int;
        let min = min as c_int;

        let mut factor = 1.0;

        if self.data.cols > max ||
           self.data.rows > max {
            factor = max as f32 / scmp::max(self.data.cols, self.data.rows) as f32;
        }
        else if self.data.cols < min ||
                self.data.rows < min {
            factor = min as f32 / scmp::min(self.data.cols, self.data.rows) as f32;
        }

        println!("Resizing image by factor {}", factor);

        self.resize_by(factor, factor);
    }

    fn replace_section(&self, lower_bound: Point2u, new_section: &PearlImage) {
        self.rectangle_custom(new_section.image.data.rows,
                              new_section.image.data.cols,
                              lower_bound.y as c_int,
                              lower_bound.x as c_int,
                              Scalar::new(255, 255, 255, 255));

        let center = Point2u::new(lower_bound.x + new_section.image.data.cols as u32 / 2,
                                  lower_bound.y + new_section.image.data.rows as u32 / 2);

        self.circle_custom(new_section.outer_radius as c_int, 
                           new_section.color,
                           &center);
        self.circle_custom(new_section.inner_radius as c_int,
                           Scalar::new(255, 255, 255, 255),
                           &center);
    }

    /* Compute Lab densities for self */
    fn lab_densities(&self) -> [DensityEstimate; 3] {
        let size = self.data.rows * self.data.cols;
        let mut l: Vec<f32> = Vec::with_capacity(size as usize);
        let mut a: Vec<f32> = Vec::with_capacity(size as usize);
        let mut b: Vec<f32> = Vec::with_capacity(size as usize);

        for y in 0..self.data.rows as usize {
            for x in 0..self.data.cols as usize {
                l.push(self.data.at3::<u8>(y as c_int, x as c_int, 0) as f32);
                a.push(self.data.at3::<u8>(y as c_int, x as c_int, 1) as f32);
                b.push(self.data.at3::<u8>(y as c_int, x as c_int, 2) as f32);
            }
        }

        let l_density = DensityEstimate::new(&l);
        let a_density = DensityEstimate::new(&a);
        let b_density = DensityEstimate::new(&b);

        [l_density, a_density, b_density]
    }

    #[allow(dead_code)]
    fn filter(&self, pearls: &mut Vec<PearlImage>) {
        if pearls.len() == 0 {
            panic!("Vector is empty");
        }
        /* All pearls assumed to have same size */
        let rows = pearls[0].image.data.rows as usize;
        let cols = pearls[0].image.data.cols as usize;

        let densities = self.lab_densities();

        let sdev_factor = 1.0;

        let l_max = densities[0].mean + sdev_factor * densities[0].sdev;
        let l_min = densities[0].mean - sdev_factor * densities[0].sdev;
        let a_max = densities[1].mean + sdev_factor * densities[1].sdev;
        let a_min = densities[1].mean - sdev_factor * densities[1].sdev;
        let b_max = densities[1].mean + sdev_factor * densities[1].sdev;
        let b_min = densities[1].mean - sdev_factor * densities[1].sdev;


        let size = rows * cols;
        let mut l: Vec<f32> = Vec::with_capacity(size);
        let mut a: Vec<f32> = Vec::with_capacity(size);
        let mut b: Vec<f32> = Vec::with_capacity(size);

        /* Remove less relevant pearls */
        pearls.retain(|p| {
            l.clear();
            a.clear();
            b.clear();
            for y in 0..p.image.data.rows as usize {
                for x in 0..p.image.data.cols as usize {
                    l.push(p.image.data.at3::<u8>(y as c_int, x as c_int, 0) as f32);
                    a.push(p.image.data.at3::<u8>(y as c_int, x as c_int, 1) as f32);
                    b.push(p.image.data.at3::<u8>(y as c_int, x as c_int, 2) as f32);
                }
            }

            let l_mean = DensityEstimate::mean(&l);
            let a_mean = DensityEstimate::mean(&a);
            let b_mean = DensityEstimate::mean(&b);

            l_mean > l_min && l_mean < l_max &&
            a_mean > a_min && a_mean < a_max &&
            b_mean > b_min && b_mean < b_max
        });
    }

    fn compute_diff(eab: f32, ssim: f32, weights: &Weights) -> f32 {
        weights.eab * eab + weights.ssim * (1.0 - ssim)
    }

    fn reproduce_impl(result: Arc<Mutex<&Image>>, 
                      orig: Image, 
                      image_size: Size2u, 
                      x_range: Range<usize>, 
                      y_range: Range<usize>, 
                      pearls: Vec<PearlImage>, 
                      upscale: Size2u, 
                      weights: Weights) {

        for y in (y_range.start..y_range.end).step_by(image_size.y as usize) {

            let y_orig = y - y_range.start;
            let y_orig_range = y_orig as c_int / upscale.y as c_int..(y_orig as c_int + image_size.y as c_int) / upscale.y as c_int;
            
            for x in (x_range.start..x_range.end).step_by(image_size.x as usize) {

                let x_orig_range = x as c_int / upscale.x as c_int..(x as c_int + image_size.x as c_int) / upscale.x as c_int;

                let sub_img: Image;
                sub_img = orig.subsection(&x_orig_range, &y_orig_range)
                                .expect("Subsection boundaries out of range");

                let mut opt: f32 = 1000000.0;
                let mut opt_img: &PearlImage = &pearls[0];

                for pearl in &pearls {
                    let eab = match math::approx_eq(weights.eab, 0.0) {
                        true  => 0.0,
                        false => cmp::mean_distance(&sub_img, &pearl),
                    };
                    let mssim = match math::approx_eq(weights.ssim, 0.0) {
                        true  => 0.0,
                        false => cmp::ssim_mean(&sub_img, &pearl.image).unwrap(),
                    };

                    let diff = Image::compute_diff(eab, mssim, &weights);

                    if diff < opt {
                        opt_img = pearl;
                        opt = diff;
                    }
                }

                {
                    let img = result.lock().unwrap();
                    img.replace_section(Point2u::new(x as u32, y as u32), opt_img);
                }
            }
        }
    }



    #[allow(dead_code)]
    pub fn reproduce(&mut self, 
                     section_size: Size2u, 
                     n_images: u32, 
                     mut image_size: Size2u, 
                     weights: Weights, 
                     filter: Filter, 
                     policy: ExecutionPolicy, 
                     proc: PostProc) 
                     -> Result<Image, Error> {

        if section_size.x > self.data.cols as u32 || 
           section_size.y > self.data.rows as u32 {
            return Err(Error::new("Invalid sub section dims"));
        }
        else if section_size.x != section_size.y {
            return Err(Error::new("Section must be nxn"));
        }
        else if image_size.x < section_size.x {
            return Err(Error::new("Image size must be larger than section size"));
        }
        else if n_images < 7 {
            return Err(Error::new("Must request at least 7 images"));
        }
        else if n_images > 500 {
            return Err(Error::new("Too many images requested"));
        }
        else if image_size.x != image_size.y {
            return Err(Error::new("Replacement images must be nxn"))
        }

        if image_size.x > 48 {
            eprintln!("Warning: Size of replacement images will be clamped to 48x48");
            image_size.x = 48;
            image_size.y = 48;
        }
        else if image_size.x < 6 {
            eprintln!("Warning: Size of replacement images will be clamped to 6x6");
            image_size.x = 6;
            image_size.y = 6;
        }

        if image_size.x % section_size.x != 0 {
            eprintln!("Warning: Image size is not an even multiple of section size");
            let factor = image_size.x / section_size.x + 1;
            eprintln!("Adjusting image size to {} times the section size", factor);
            image_size.x = factor * section_size.x;
            image_size.y = factor * section_size.y;
        }
        println!("Reproducing...");

        /* Generate sub images */
        let mut pearls: Vec<PearlImage> = Vec::with_capacity(n_images as usize);
        let sample_space = SampleSpace::new(n_images as c_int);

        let step = 3usize;
        let nradii = image_size.x as usize / 2 / step + 1;

        let mut radii: Vec<u32> = Vec::with_capacity(nradii);
        for i in 0..nradii - 1 {
            radii.push((i*step) as u32);
        }
        radii.push(image_size.x / 2 - 1);

        for color in sample_space {
            for radius in &radii {
                pearls.push(PearlImage::with_inner_radius(color, &image_size, *radius).unwrap());
            }
        }

        match filter {
            Filter::Sdev => self.filter(&mut pearls),
            _ => (),
        };

        /* Make even multiple of section_size */
        let rem_x = self.data.cols as u32 % section_size.x;
        let rem_y = self.data.rows as u32 % section_size.y;

        self.resize(self.data.rows as u32 - rem_y, 
                    self.data.cols as u32 - rem_x);

        /* Factors the image will be upscaled with */
        let x_upscale = image_size.x / section_size.x;
        let y_upscale = image_size.y / section_size.y;

        let mut result = Image::new(self.data.rows * y_upscale as c_int,
                                self.data.cols * x_upscale as c_int);

        let res = Arc::new(Mutex::new(&result));

        match policy {
            ExecutionPolicy::Sequential => {
                Image::reproduce_impl(res, 
                                      self.clone(), 
                                      image_size, 
                                      0..result.cols() as usize, 
                                      0..result.rows() as usize, 
                                      pearls, 
                                      Size2u::new(x_upscale, y_upscale),
                                      weights);
            },

            _ => {
                let nthreads = match policy {
                    ExecutionPolicy::Parallellx4  => 4,
                    ExecutionPolicy::Parallellx8  => {
                        eprintln!("Note: the benefits of running more than 4 threads are generally next to none.");
                        8
                    },
                    _ => return Err(Error::new("Unknown execution policy")),
                };

                crossbeam::scope( |scope| {
                    let mut handles = vec![];

                    let mut row_start = 0usize;
                    let mut rem_rows = self.data.rows as usize;

                    for i in 0..nthreads {
                        let ares = Arc::clone(&res);
                        let image_size = image_size.clone();
                        let pearls = pearls.clone();
                        let cols = result.cols() as usize;
                        let weights = weights.clone();

                        let per_thread = rem_rows as f32 / section_size.y as f32 / (nthreads - i) as f32;
                        let thread_rem = per_thread % 1.0;

                        let per_thread = per_thread as usize;
                        let mut row_end = row_start + per_thread * section_size.y as usize * y_upscale as usize;

                        if !math::approx_eq(thread_rem, 0.0) {
                            /* Thread should handle an extra section of rows */
                            row_end += section_size.y as usize * y_upscale as usize;
                        }

                        let soi_x_range = 0..self.data.cols;
                        let soi_y_range = row_start as c_int / y_upscale as c_int..row_end as c_int / y_upscale as c_int;

                        let soi = self.subsection(&soi_x_range, &soi_y_range).unwrap();

                        let handle = scope.spawn(move |_| {
                            Image::reproduce_impl(ares, 
                                                  soi, 
                                                  image_size, 
                                                  0..cols, 
                                                  row_start..row_end, 
                                                  pearls, 
                                                  Size2u::new(x_upscale, y_upscale),
                                                  weights);
                        });

                        handles.push(handle);
                        rem_rows -= (row_end - row_start) / y_upscale as usize;
                        row_start = row_end;
                    }
                    
                    for thread in handles {
                        thread.join().unwrap();
                    }

                }).unwrap();

            },
        };

        match proc {
            PostProc::OrigSize => result.resize(self.data.rows as u32, self.data.cols as u32),
            _ => (),
        };

        Ok(result)
    }
}



#[derive(Debug)]
#[allow(dead_code)]
pub struct Window {
    title: String,
    active: bool,
}

impl Window {
    #[allow(dead_code)]
    pub fn new(title: &str) -> Window {
        let title = title.to_string();
        let active = false;
        Window{ title, active }
    }

    #[allow(dead_code)]
    pub fn show(&mut self, image: &Image) -> Result<(), Error> {
        if !self.active {
            self.active = match highgui_named_window(&self.title, WindowFlag::Normal) {
                Ok(()) => true,
                Err(_) => return Err(Error::new("Could not create window")),
            };
        }
        match image.data.show(&self.title, 0) {
            Ok(()) => Ok(()),
            Err(_) => {
                Err(Error::new("Could not show image"))
            },
        }
    }
}
