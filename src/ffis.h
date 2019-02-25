#ifndef FFIS_H
#define FFIS_H

#pragma once
#include <opencv2/opencv.hpp>


struct SSIM {
	double b;
	double g;
	double r;
};

extern "C" {
	/* Write image to disk */
	int cv_write(char const* path, unsigned char const* data, int rows, int cols, int type);

	/* Compute MSSIM per channel */
	SSIM cv_ssim(unsigned char const* im1_data, int im1_rows, int im1_cols, int im1_type, 
				 unsigned char const* im2_data, int in2_rows, int im2_cols, int im2_type);
}

#endif
