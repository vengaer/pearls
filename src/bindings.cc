#include "bindings.h"
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>



int cv_write(char const* path, unsigned char const* data, int rows, int cols, int type) {
	static_assert(std::numeric_limits<unsigned char>::digits == 8, "Unsuitable byte size");
	
	using cv::Mat;
	using cv::Vec3b;
	using cv::Point;
	using cv::Size;

	if(type != CV_8UC3) {
		std::cerr << "Only 3-channel images are supported\n";
		return -1;
	}

	int const channels = 3;
	auto* buf = const_cast<unsigned char*>(data);

	Mat img{Size{cols, rows}, type, reinterpret_cast<void*>(buf)};
	
	try {
		if(cv::imwrite(path, img))
			return 0;
	}
	catch(...) { }

	return -1;
}
