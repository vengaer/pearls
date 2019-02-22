#ifndef BINDINGS_H
#define BINDINGS_H

#pragma once


extern "C" {
	/* Write image to disk */
	int cv_write(char const* path, unsigned char const* data, int rows, int cols, int type);
}

#endif
