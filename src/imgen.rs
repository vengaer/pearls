mod color;
mod error;
mod raw;

pub mod cmp;
pub mod core;
pub mod math;

#[cfg(test)]
mod tests {
    use crate::imgen::math;
    use crate::imgen::core::Image;

    /* On the SSIM test: Matlab uses an 8x8 block window 
     * whereas I use an 11x11 Gaussian meaning the 
     * results won't be identical */

    #[test]
    fn ssim_self() {
        let im1 = Image::from_file("tests/test1.jpg").unwrap();
        let ssim = im1.ssim(&im1).unwrap();
        assert!(math::arbitrarily_close(ssim.r as f32, 1.0));
        assert!(math::arbitrarily_close(ssim.g as f32, 1.0));
        assert!(math::arbitrarily_close(ssim.b as f32, 1.0));
    }

    #[test]
    fn ssim_test1() {
        let orig = Image::from_file("tests/test1.jpg").unwrap();
        let repr = Image::from_file("tests/t1_out.jpg").unwrap();
        let ssim = orig.ssim(&repr).unwrap();

        /* SSIM per Matlab */
        assert!(math::arbitrarily_close(ssim.r as f32, 0.0980));
        assert!(math::arbitrarily_close(ssim.g as f32, 0.0991));
        assert!(math::arbitrarily_close(ssim.b as f32, 0.0811));
    }

    #[test]
    fn ssim_test2() {
        let orig = Image::from_file("tests/test2.jpg").unwrap();
        let repr = Image::from_file("tests/test2_out.jpg").unwrap();
        let ssim = orig.ssim(&repr).unwrap();

        /* SSIM per Matlab */
        assert!(math::arbitrarily_close(ssim.r as f32, 0.0273));
        assert!(math::arbitrarily_close(ssim.g as f32, 0.0289));
        assert!(math::arbitrarily_close(ssim.b as f32, 0.0279));
    }
}
