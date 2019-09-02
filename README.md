# Pearls
Image reproduction using smaller pearl-like images. The program reads an image from the disk, generates a number of colored, hollow circles and uses them to reprocue the original image. Written as part of the course TNM097 Image Reproduction and Image Quality at Linköping University.

[![Build Status](https://gitlab.com/vilhelmengstrom/pearls/badges/master/build.svg)](https://gitlab.com/vilhelmengstrom/pearls/commits/master)

## Dependencies
- [OpenCV](https://opencv.org)

## Configuration
Options may be specified in a .toml file. See the config.toml file in the project root for examples. If ran through Cargo, the file will be read from the project root. If the project is ran straight from the executable, it will attempt to read the config from the same directory. Custom locations to config files may be specified using the -c flag.

## Flags
- -i   Specify path to input image*
- -o   Specify path to where the output image is to be stored*
- -c   Specify path to where the config file is located

*Options specified with these flags override options in the config file

## Results
### Original
![](images/landscape.jpg)
### Non-filtered
![](images/landscape_non_filtered_out.jpg)
### Filtered
![](images/landscape_filtered_out.jpg)

## Common Issues
- OpenCV is installed but opencv2/opencv.hpp is not found.  
  Make sure that opencv2 is in your include path. As an example, Pacman installs OpenCV to /usr/include/opencv4/opencv2 rather than the conventional /usr/include/opencv2. The easiest way to make the compiler find the headers is to create a symlink using `sudo ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2`.
  

## Credits
Images courtesy of Wikimedia Commons and Pexels.  
Thanks to Github users nebgnahz, Pzixel, Restioson, yuvallanger, joelgallant, oefd, AgustinCB and peahonen for [cv-rs](https://github.com/nebgnahz/cv-rs)
