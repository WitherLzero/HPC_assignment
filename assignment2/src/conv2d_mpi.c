#include "conv2d_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void calculate_stride_output_dims(int input_H, int input_W, int stride_H, int stride_W,
                                  int *output_H, int *output_W) {
    *output_H = (int)ceil((double)input_H / stride_H);
    *output_W = (int)ceil((double)input_W / stride_W);
}

void conv2d_stride_serial(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                          int sH, int sW, float **restrict output) {
    // For "same" padding, the unpadded dimensions would be:
    int original_H = H - kH + 1;  // Remove padding to get original size
    int original_W = W - kW + 1;

    // Calculate strided output dimensions based on original (unpadded) size
    int output_H, output_W;
    calculate_stride_output_dims(original_H, original_W, sH, sW, &output_H, &output_W);

    // Perform strided convolution on the padded input
    // The padded input f has dimensions H x W
    // We sample at stride intervals starting from the valid convolution positions

    for (int i = 0; i < output_H; i++) {
        for (int j = 0; j < output_W; j++) {
            float sum = 0.0f;

            // Starting position in the padded input for this output element
            // Since input is padded, we can start convolution at position (i*sH, j*sW)
            int start_row = i * sH;
            int start_col = j * sW;

            // Ensure we don't go beyond the valid convolution area
            if (start_row + kH <= H && start_col + kW <= W) {
                // Apply kernel at this position
                for (int ki = 0; ki < kH; ki++) {
                    for (int kj = 0; kj < kW; kj++) {
                        sum += f[start_row + ki][start_col + kj] * g[ki][kj];
                    }
                }
            }

            output[i][j] = sum;
        }
    }
}

// Placeholder implementations for parallel functions (not implemented yet)
void conv2d_stride_openmp(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                          int sH, int sW, float **restrict output) {
    printf("OpenMP implementation not yet implemented. Using serial version.\n");
    conv2d_stride_serial(f, H, W, g, kH, kW, sH, sW, output);
}

void conv2d_stride_mpi(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                       int sH, int sW, float **restrict output, MPI_Comm comm) {
    printf("MPI implementation not yet implemented. Using serial version.\n");
    conv2d_stride_serial(f, H, W, g, kH, kW, sH, sW, output);
}

void conv2d_stride_hybrid(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                          int sH, int sW, float **restrict output, MPI_Comm comm) {
    printf("Hybrid implementation not yet implemented. Using serial version.\n");
    conv2d_stride_serial(f, H, W, g, kH, kW, sH, sW, output);
}


// TODO: cached-aligned allocator 

// Matrix utility functions (reused from assignment1)
float **allocate_matrix(int rows, int cols) {
    float **matrix = (float **)malloc(rows * sizeof(float *));
    if (matrix == NULL) {
        fprintf(stderr, "Error: Unable to allocate matrix rows\n");
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        matrix[i] = (float *)malloc(cols * sizeof(float));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Error: Unable to allocate matrix column %d\n", i);
            // Free previously allocated rows
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

void free_matrix(float **matrix, int rows) {
    if (matrix == NULL) return;

    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void initialize_matrix(float **matrix, int rows, int cols, float value) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = value;
        }
    }
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
        // Copy each row of the input matrix into the center of the padded matrix
        memcpy((*padded)[i + pad_top] + pad_left, input[i],
               width * sizeof(float));
    }
}

// Placeholder implementations for MPI functions that will be needed later
void mpi_distribute_matrix(float **global_matrix, int global_H, int global_W,
                           int kernel_H, int kernel_W,
                           float ***local_matrix, int *local_H, int *local_W,
                           int *local_start_row, MPI_Comm comm) {
    // Placeholder - not implemented yet
    printf("MPI distribute matrix not yet implemented.\n");
}

void mpi_gather_output(float **local_output, int local_output_H, int local_output_W,
                       int local_start_row, float ***global_output,
                       int global_output_H, int global_output_W, MPI_Comm comm) {
    // Placeholder - not implemented yet
    printf("MPI gather output not yet implemented.\n");
}

void mpi_broadcast_kernel(float ***kernel, int kernel_H, int kernel_W, MPI_Comm comm) {
    // Placeholder - not implemented yet
    printf("MPI broadcast kernel not yet implemented.\n");
}

// MPI timing utilities
void mpi_timer_start(mpi_timer_t *timer) {
    timer->start_time = MPI_Wtime();
}

void mpi_timer_end(mpi_timer_t *timer, MPI_Comm comm) {
    timer->end_time = MPI_Wtime();
    timer->elapsed_time = timer->end_time - timer->start_time;

    // For serial testing, just copy values
    timer->max_time = timer->elapsed_time;
    timer->min_time = timer->elapsed_time;
    timer->avg_time = timer->elapsed_time;
}

void mpi_timer_print(mpi_timer_t *timer, const char *description, MPI_Comm comm) {
    printf("Timing - %s: %.6f seconds\n", description, timer->elapsed_time);
    fflush(stdout);
}