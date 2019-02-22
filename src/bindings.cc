#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>
#include <iostream>
#include <limits>


extern "C"{

	/* Write image to disk */
	int cv_write(char const* path, unsigned char const* data, int rows, int cols, int type) {
		static_assert(std::numeric_limits<unsigned char>::digits == 8, "Unsuitable byte size");
		
		using cv::Mat;
		using cv::Vec3b;
		using cv::Point;

		if(type != CV_8UC3) {
			std::cerr << "Only 3-channel images are supported\n";
			return -1;
		}

		int const channels = 3;

		Mat img = Mat::zeros(rows, cols, type);
		
		for(int y = 0; y < rows; y++) {
			for(int x = 0; x < cols; x++) {
				Vec3b& col = img.at<Vec3b>(Point(x,y));
				for(int z = 0; z < channels; z++)
					col[z] = data[channels * cols * y + channels * x + z];
			}
		}
		
		try {
			if(cv::imwrite(path, img))
				return 0;
		}
		catch(...) { }
	
		return -1;
	}
}
