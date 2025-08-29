#define _POSIX_C_SOURCE 200809L
#include "../include/conv2d.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void conv2d_serial(float **f, int H, int W, float **g, int kH, int kW,
                   float **output) {
    // Compute valid output dimensions from padded input and kernel sizes
    const int out_H = H - kH + 1;
    const int out_W = W - kW + 1;

    // Perform convolution producing an output of size out_H x out_W
    for (int i = 0; i < out_H; i++) {
        for (int j = 0; j < out_W; j++) {
            float sum = 0.0f;

            // Apply kernel
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    sum += f[i + ki][j + kj] * g[ki][kj];
                }
            }

            output[i][j] = sum;
        }
    }
}

void conv2d_parallel(float **f, int H, int W, float **g, int kH, int kW,
                     float **output) {
    // Compute valid output dimensions from padded input and kernel sizes
    const int out_H = H - kH + 1;
    const int out_W = W - kW + 1;

    // Perform parallel convolution
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < out_H; i++) {
        for (int j = 0; j < out_W; j++) {
            float sum = 0.0f;
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    sum += f[i + ki][j + kj] * g[ki][kj];
                }
            }

            output[i][j] = sum;
        }
    }
}

#if defined(_ISOC11_SOURCE)
#define ALIGNED_ALLOC_SUPPORTED
#endif

#ifndef MATRIX_ALIGNMENT
#define MATRIX_ALIGNMENT 32
#endif

/**
 * @brief Allocate a 2D matrix
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return float** Pointer to the allocated matrix
 */
float **allocate_matrix(int rows, int cols) {
    float **matrix = NULL;
#ifdef ALIGNED_ALLOC_SUPPORTED
    matrix = (float **)aligned_alloc(MATRIX_ALIGNMENT, rows * sizeof(float *));
#else
    // fallback to malloc if aligned_alloc is not available
    matrix = (float **)malloc(rows * sizeof(float *));
#endif
    if (matrix == NULL) {
        perror("Error: Failed to allocate memory for matrix rows\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
#ifdef ALIGNED_ALLOC_SUPPORTED
        matrix[i] =
            (float *)aligned_alloc(MATRIX_ALIGNMENT, cols * sizeof(float));
#else
        matrix[i] = (float *)malloc(cols * sizeof(float));
#endif
        if (matrix[i] == NULL) {
            perror("Error: Failed to allocate memory for matrix columns\n");
            // Free previously allocated rows
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            exit(EXIT_FAILURE);
        }
    }

    return matrix;
}

// Free a 2D matrix
void free_matrix(float **matrix, int rows) {
    if (matrix == NULL) return;

    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Initialize matrix with a specific value
void initialize_matrix(float **matrix, int rows, int cols, float value) {
    for (int i = 0; i < rows; i++) {
        memset(matrix[i], value, (size_t)cols * sizeof(float));
    }
}

// // Copy matrix from source to destination
// void copy_matrix(float **src, float **dst, int rows, int cols) {
//     for (int i = 0; i < rows; i++) {
//         memcpy(dst[i], src[i], (size_t)cols * sizeof(float));
//     }
// }

// Compare two matrices with tolerance
int compare_matrices(float **matrix1, float **matrix2, int rows, int cols,
                     float tolerance) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fabs(matrix1[i][j] - matrix2[i][j]) > tolerance) {
                return 0;  // Matrices are different
            }
        }
    }
    return 1;  // Matrices are the same
}

void generate_padded_matrix(float **input, int height, int width,
                            int kernel_height, int kernel_width,
                            float ***padded, int *padded_height,
                            int *padded_width) {
    // Asymmetric "same" padding so that output has the same size as input
    // Works for both odd and even kernel sizes
    int pad_top = (kernel_height - 1) / 2;
    int pad_left = (kernel_width - 1) / 2;
    int pad_bottom = kernel_height - 1 - pad_top;
    int pad_right = kernel_width - 1 - pad_left;

    *padded_height = height + pad_top + pad_bottom;
    *padded_width = width + pad_left + pad_right;
    *padded = allocate_matrix(*padded_height, *padded_width);
    initialize_matrix(*padded, *padded_height, *padded_width, 0.0f);
    for (int i = 0; i < height; i++) {
        // Copy each row of the input matrix into the center of the padded
        // matrix
        memcpy((*padded)[i + pad_top] + pad_left, input[i],
               width * sizeof(float));
    }
}
