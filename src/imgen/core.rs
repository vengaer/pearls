pub use crate::imgen::raw::SSIM;
pub use cv::core::Scalar;
pub use cv::highgui::{Show, WindowFlag, highgui_named_window};

use crate::imgen::color::SampleSpace;
use crate::imgen::error::Error;
use crate::imgen::math::{DensityEstimate, Point2u, Point3f, Size2u};
use crate::imgen::raw;
use cv::core::{CvType, LineType, Point2i, Rect, Size2i};
use cv::imgcodecs::ImageReadMode;
use cv::imgproc::{ColorConversion, InterpolationFlag};
use cv::mat::Mat;
use libc::c_int;
use std::f64::consts::PI;
use std::iter::Iterator;
use std::ops::Range;


#[derive(Debug, Clone)]
pub struct Image {
    pub data: Mat,
}

impl Image {
    pub fn new(rows: c_int, cols: c_int) -> Image {
        let data = Mat::with_size(rows, cols, CvType::Cv8UC3 as c_int);
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

    pub fn subsection(&self, x_range: Range<c_int>, y_range: Range<c_int>) -> Result<Image, Error> {
        if y_range.end <= y_range.start || x_range.end <= x_range.start {
            return Err(Error::new("Range must be of length >= 1 and increasing"))
        }
        else if y_range.end > self.data.rows || x_range.end > self.data.cols {
            println!("x_range: {}-{}, y_range: {}-{}", x_range.start, x_range.end, y_range.start, y_range.end);
            println!("Dims: {}x{}", self.data.cols, self.data.rows);
            return Err(Error::new("Range out of bounds"));
        }

        let rows = y_range.end - 1 - y_range.start;
        let cols = x_range.end - 1 - x_range.start;

        let rect = Rect::new(x_range.start, y_range.start, rows, cols);
        let data = self.data.roi(rect);

        Ok(Image::from_mat(data))
    }

}

#[derive(Debug, Clone)]
struct PearlImage {
    pub image: Image,
    outer_radius: u32,
    inner_radius: u32,
    color: Scalar,
}

impl PearlImage {

    pub fn new(fg_color: Scalar, image_dims: &Size2u) -> PearlImage {
        let image = Image::new(image_dims.y as c_int, image_dims.x as c_int);

        let outer_radius = image.data.rows as u32 / 2;
        let inner_radius = image.data.rows as u32 / 4;
        image.rectangle(image.data.rows, image.data.cols, Scalar::all(255));
        image.circle(outer_radius as c_int, fg_color);
        image.circle(inner_radius as c_int, Scalar::all(255));

        PearlImage{ image, outer_radius, inner_radius, color: fg_color }
    }

    pub fn customize(&self, new_radius: u32) {
        if new_radius >= self.outer_radius {
            return ();
        }

        if new_radius < self.inner_radius {
            self.image.circle(self.outer_radius as c_int, self.color);
        }
        if new_radius != 0 {
            self.image.circle(new_radius as c_int, Scalar::all(255));
        }
    }
}

#[derive(Debug)]
enum SizeMod {
    NoOpt,
    Decrease,
    Increase
}

#[derive(Debug)]
struct ImageDistance<'a> {
    pub image: &'a PearlImage,
    pub dist: f64,
}

impl<'a> ImageDistance<'a> {
    pub fn mean<'b>(original: &Image, replacement: &'b PearlImage) -> ImageDistance<'b> {
        let repl_lab = replacement.image.data.cvt_color(ColorConversion::RGB2Lab);
        let orig_lab = original.data.cvt_color(ColorConversion::RGB2Lab);
        let lm_repl = ImageDistance::lab_means(&Image::from_mat(repl_lab));
        let lm_orig = ImageDistance::lab_means(&Image::from_mat(orig_lab));
        let dist = lm_repl.euclid_dist(&lm_orig) as f64;

        ImageDistance { image: replacement, dist }
    }

    fn lab_means(lab_img: &Image) -> Point3f {

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

    pub fn force_dist(&mut self, dist: f64) {
        self.dist = dist;
    }

    /* Implementation of greedy algorithm to minimize Eab. Heavily inspired by gradient descent */
    fn optimize_inner_radius_impl(mut img: &mut PearlImage, cmp_means: &Point3f, radius: u32, mut step: u32, dist: f32, prev: SizeMod) -> u32 {
        println!("Radius: {}, Step: {} ", radius, step);
        println!("-> Adjusted Eab: {}", dist);

        /* Base case */
        if step == 0 {
            return radius;
        }

        /* If larger radius would be >= as outer radius, make step smaller and push radius inward */
        let larger_radius = match radius {
            r if r + step > img.outer_radius => {
                step /= 2;
                img.outer_radius - step
            },
            _ => radius + step, /* Larger radius in allowed interval */
        };
        /* If smaller radius would be <= 0, make step smaller and push radius outward */
        let smaller_radius = match radius {
            r if r < step  => {
                step /= 2;
                0 + step
            },
            _ => radius - step, /* Smaller radius in allows interval */
        };

        /* Get Eab for pearl with larger inner radius */
        img.customize(larger_radius);
        let larger_lab = img.image.data.cvt_color(ColorConversion::RGB2Lab);
        let lm_large = ImageDistance::lab_means(&Image::from_mat(larger_lab));

        let lrad_dist = lm_large.euclid_dist(&cmp_means);

        /* Get Eab for pearl with smaller inner radius */
        img.customize(smaller_radius);
        let smaller_lab = img.image.data.cvt_color(ColorConversion::RGB2Lab);
        let lm_small = ImageDistance::lab_means(&Image::from_mat(smaller_lab));

        let srad_dist = lm_small.euclid_dist(&cmp_means);

        /* If step is 1 and current Eab is better than the alternatives, stop early */
        if step == 1 && dist < srad_dist && dist < lrad_dist {
            println!("Radius: {}, Step: {} ", radius, step);
            println!("Early stop");
            return radius;
        }

        let mut nstep = step;
        let nmod: SizeMod;
        let nrad: u32;
        let ndist: f32;
        
        /* Larger radius gives smaller Eab, try even larger */
        if lrad_dist < srad_dist {

            nmod = SizeMod::Increase;
            nrad = larger_radius;
            ndist = lrad_dist;

            /* If the previous size mod shrunk the radius, halve the step */
            match prev {
                SizeMod::Decrease => nstep = step / 2,
                _ => (),
            };
        }
        /* Smaller radius gives smaller Eab, try even smaller */
        else {
            nmod = SizeMod::Decrease;
            nrad = smaller_radius;
            ndist = srad_dist;

            /* If the previous size mod enlarged the radius, halve the step */
            match prev {
                SizeMod::Increase => nstep = step / 2,
                _ => (),
            };
        }

        /* Recursive call */
        ImageDistance::optimize_inner_radius_impl(&mut img, cmp_means, nrad, nstep, ndist, nmod)
    }

    pub fn optimize_inner_radius(&self, cmp: &Image) -> PearlImage {
        let step = self.image.inner_radius / 2;

        /* Precompute Lab means for image to compare with */
        let lab_cmp = cmp.data.cvt_color(ColorConversion::RGB2Lab);
        let cmp_means = ImageDistance::lab_means(&Image::from_mat(lab_cmp));

        /* Image to return (modifying self would ruin future comparisons) */
        let mut img = self.image.clone();

        /* Get optimal radius and modify image to return */
        let new_radius = ImageDistance::optimize_inner_radius_impl(&mut img, &cmp_means, self.image.inner_radius, step, 1000f32, SizeMod::NoOpt);
        img.inner_radius = new_radius;
        img.customize(new_radius);

        img
    }
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

        /* TODO: sdev_factor should depend on the image */
        let sdev_factor = 2.5;

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

        pearls.retain(|p| {
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



    pub fn reproduce(&mut self, section_size: Size2u, n_images: u32, image_size: Size2u) -> Result<Image, Error> {
        if section_size.x > self.data.cols as u32 || 
           section_size.y > self.data.rows as u32 {
            return Err(Error::new("Invalid sub section dims"));
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

        /* Generate sub images */
        let mut pearls: Vec<PearlImage> = Vec::with_capacity(n_images as usize);
        let sample_space = SampleSpace::new(n_images as c_int);

        for color in sample_space {
            pearls.push(PearlImage::new(color, &image_size));
        }

        self.filter(&mut pearls);

        /* Make even multiple of section_size */
        let rem_x = self.data.cols as u32 % section_size.x;
        let rem_y = self.data.rows as u32 % section_size.y;

        self.resize(self.data.rows as u32 - rem_y, 
                    self.data.cols as u32 - rem_x);

        /* Factors the image will be upscaled with */
        let x_upscale = image_size.x / section_size.x;
        let y_upscale = image_size.y / section_size.y;

        let result = Image::new(self.data.rows * y_upscale as c_int,
                                self.data.cols * x_upscale as c_int);

        let mut opt_img = ImageDistance::mean(&Image::new(1,1), &pearls[0]);
        opt_img.force_dist(999999999f64);

        /* Keep track of progress */
        let iterations = (result.rows() as u32 / image_size.y) as f32 *
                         (result.cols() as u32 / image_size.x) as f32;
        let mut progress = 0f32;

        for y in (0..result.rows() as usize).step_by(image_size.y as usize) {
            for x in (0..result.cols() as usize).step_by(image_size.x as usize) {
                progress += 1.0;
                println!("Processing: {}/{}", progress, iterations);

                /* Select sub image */
                let sub_img = self.subsection(x as c_int / x_upscale as c_int
                                              ..(x as c_int + image_size.x as c_int) / x_upscale as c_int, 
                                              y as c_int / x_upscale as c_int
                                              ..(y as c_int + image_size.y as c_int) / y_upscale as c_int)
                    .expect("Subsection boundaries out of range");
                
                /* Check all generated pearls agains sub image */
                for pearl in &pearls {
                    let c_img = ImageDistance::mean(&sub_img, &pearl);

                    if c_img.dist < opt_img.dist {
                        opt_img = c_img;
                    }
                }
                println!("Eab: {}", opt_img.dist);

                /* Pearl with inner radius that minimizes Eab */
                let piece = opt_img.optimize_inner_radius(&sub_img);

                /* Add to image */
                result.replace_section(Point2u::new(x as u32, y as u32), 
                                       &piece);

                opt_img.force_dist(999999999f64);
                print!("\n");
            }
        }

        Ok(result)
    }

    #[allow(dead_code)]
    pub fn ssim(&self, other: &Image) -> Result<SSIM, Error> {
        if self.data.rows == other.data.rows &&
           self.data.cols == other.data.cols {
            return raw::ssim(&self.data, &other.data);
        }

        let mut other = other.clone();
        other.resize(self.data.rows as u32, self.data.cols as u32);
        raw::ssim(&self.data, &other.data)
    }

}



#[derive(Debug)]
#[allow(dead_code)]
pub struct Window {
    title: String,
}

impl Window {
    pub fn new(title: &str) -> Window {
        highgui_named_window(&title, WindowFlag::Normal).unwrap();

        let title = title.to_string();
        Window{ title }
    }

    #[allow(dead_code)]
    pub fn show(&self, image: &Image) -> Result<(), Error> {
        match image.data.show(&self.title, 0) {
            Ok(()) => Ok(()),
            Err(_) => {
                Err(Error::new("Could not show image"))
            },
        }
    }
}
