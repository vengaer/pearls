#include "cv.h"
#include <iostream>
#include <limits>



int cv_write(char const* path, unsigned char const* data, int rows, int cols, int type) {
	static_assert(std::numeric_limits<unsigned char>::digits == 8, "Unsuitable byte size");
	
	using cv::Mat;
	using cv::Size;

	if(type != CV_8UC3) {
		std::cerr << "Only 3-channel images are supported\n";
		return -1;
	}

	auto* buf = const_cast<unsigned char*>(data);
	Mat img{Size{cols, rows}, type, static_cast<void*>(buf)};
	
	try {
		if(cv::imwrite(path, img))
			return 0;
	}
	catch(...) { }

	return -1;
}

/* Taken from the OpenCV documentation */
SSIM cv_ssim(unsigned char const* im1_data, int im1_rows, int im1_cols, int im1_type, 
		     unsigned char const* im2_data, int im2_rows, int im2_cols, int im2_type) {
	static_assert(std::numeric_limits<unsigned char>::digits == 8, "Unsuitable byte size");

	using cv::Mat;
	using cv::Scalar;
	using cv::Size;

	if(im1_type != CV_8UC3 || im2_type != CV_8UC3) {
		std::cerr << "Only 3-channel images are supported\n";
		return {20.f, 20.f, 20.f};
	}
	

	auto* buf_im1 = const_cast<unsigned char*>(im1_data);
	auto* buf_im2 = const_cast<unsigned char*>(im2_data);

	Mat img1{Size{im1_cols, im1_rows}, im1_type, static_cast<void*>(buf_im1)};
	Mat img2{Size{im2_cols, im2_rows}, im2_type, static_cast<void*>(buf_im2)};

	double const c1 = 6.5025, c2 = 58.5225;
	int d = CV_32F;

	Mat i1, i2;

	img1.convertTo(i1, d);
	img2.convertTo(i2, d);

	Mat i1_2 = i1.mul(i1);
	Mat i2_2 = i2.mul(i2);
	Mat i1_i2 = i1.mul(i2);

	Mat mu1, mu2;
	
	GaussianBlur(i1, mu1, Size(11, 11), 1.5);
	GaussianBlur(i2, mu2, Size(11, 11), 1.5);

	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);

	Mat sigma1_2, sigma2_2, sigma12;

	GaussianBlur(i1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	GaussianBlur(i2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;

	GaussianBlur(i1_i2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	/* Formula */
	Mat t1, t2, t3;

	t1 = 2 * mu1_mu2 + c1;
	t2 = 2 * sigma12 + c2;
	t3 = t1.mul(t2);

	t1 = mu1_2 + mu2_2 + c1;
	t2 = sigma1_2 + sigma2_2 + c2;
	t1 = t1.mul(t2);

	Mat ssim_map;
	divide(t3, t1, ssim_map);

	Scalar mssim = mean(ssim_map);

	return {mssim.val[0], mssim.val[1], mssim.val[2]};
}
