use crate::imgen::error::Error;
use cv::core::Scalar;
use num::Float;

#[derive(Debug)]
pub struct Point2f{
    pub x: f32,
    pub y: f32,
}
impl Point2f {
    pub fn new(x: f32, y: f32) -> Point2f {
        Point2f{ x, y }
    }

    pub fn euclid_dist(&self, pt2: Point2f) -> f32 {
        ((self.x - pt2.x).powf(2.0) +
         (self.y - pt2.y).powf(2.0)).sqrt()
    }

    pub fn from_arr<T: Float>(arr: &[T]) -> Result<Point2f, Error> {
        if arr.len() != 2 {
            Err(Error::new("Length of array must be 3"))
        }
        else {
            Ok(Point2f{x: num::cast(arr[0]).unwrap(), 
                       y: num::cast(arr[1]).unwrap()})
        }
    }
}

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

    pub fn euclid_dist(&self, pt2: Point3f) -> f32 {
        ((self.x - pt2.x).powf(2.0) + 
         (self.y - pt2.y).powf(2.0) +
         (self.z - pt2.z).powf(2.0)).sqrt()
    }

    pub fn from_arr<T: Float>(arr: &[T]) -> Result<Point3f, Error> {
        if arr.len() != 3 {
            Err(Error::new("Length of array must be 3"))
        }
        else {
            Ok(Point3f{x: num::cast(arr[0]).unwrap(), 
                       y: num::cast(arr[1]).unwrap(), 
                       z: num::cast(arr[2]).unwrap()})
        }
    }
}

#[derive(Debug)]
pub struct Point2u{
    pub x: u32,
    pub y: u32,
}

impl Point2u {
    pub fn new(x: u32, y: u32) -> Point2u {
        Point2u{ x, y }
    }
}
