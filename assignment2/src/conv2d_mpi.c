#include "conv2d_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// OpenMP performance tuning parameters
#define OMP_CHUNK_SIZE 8        // Rows per chunk for dynamic scheduling
#define PREFETCH_DISTANCE 2     // Rows to prefetch ahead
#define MIN_PARALLEL_SIZE 100   // Minimum matrix size for parallelization

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

// OpenMP parallel implementation with stride
void conv2d_stride_openmp(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                          int sH, int sW, float **restrict output) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("OpenMP General: Using %d threads\n", omp_get_num_threads());
        }
    }

    // Calculate actual input dimensions (remove padding)
    int original_H = H - kH + 1;
    int original_W = W - kW + 1;

    // Calculate strided output dimensions
    int output_H = (int)ceil((double)original_H / sH);
    int output_W = (int)ceil((double)original_W / sW);

    // Check for minimum size threshold - use serial for small matrices
    if (output_H * output_W < MIN_PARALLEL_SIZE) {
        conv2d_stride_serial(f, H, W, g, kH, kW, sH, sW, output);
        return;
    }

    // Select optimized implementation based on kernel size
    if (kH == 3 && kW == 3) {
        conv2d_3x3_stride_optimized_openmp(f, H, W, g, sH, sW, output);
        return;
    } else if (kH == 5 && kW == 5) {
        conv2d_5x5_stride_optimized_openmp(f, H, W, g, sH, sW, output);
        return;
    }

    // General OpenMP implementation
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, OMP_CHUNK_SIZE) nowait
        for (int i = 0; i < output_H; i++) {
            // Prefetch next input row for better cache performance
            int next_row = (i + PREFETCH_DISTANCE) * sH;
            if (next_row < H) {
                __builtin_prefetch(&f[next_row][0], 0, 3);
            }

            for (int j = 0; j < output_W; j++) {
                float sum = 0.0f;
                int start_row = i * sH;
                int start_col = j * sW;

                // Apply kernel with bounds checking
                if (start_row + kH <= H && start_col + kW <= W) {
                    for (int ki = 0; ki < kH; ki++) {
                        #pragma omp simd reduction(+:sum)
                        for (int kj = 0; kj < kW; kj++) {
                            sum += f[start_row + ki][start_col + kj] * g[ki][kj];
                        }
                    }
                }

                output[i][j] = sum;
            }
        }
    }
}

// Optimized 3x3 kernel OpenMP implementation with stride
void conv2d_3x3_stride_optimized_openmp(float **restrict f, int H, int W, float **restrict g,
                                         int sH, int sW, float **restrict output) {
    // Calculate actual input dimensions (remove padding)
    int original_H = H - 2;  // 3x3 kernel needs 2 pixels of padding removed
    int original_W = W - 2;

    // Calculate strided output dimensions
    int output_H = (int)ceil((double)original_H / sH);
    int output_W = (int)ceil((double)original_W / sW);

    // Cache kernel values for better performance
    const float g00 = g[0][0], g01 = g[0][1], g02 = g[0][2];
    const float g10 = g[1][0], g11 = g[1][1], g12 = g[1][2];
    const float g20 = g[2][0], g21 = g[2][1], g22 = g[2][2];

    // Parallel loop over output rows
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < output_H; i++) {
        int row = i * sH;
        for (int j = 0; j < output_W; j++) {
            int col = j * sW;

            // Fully unrolled 3x3 convolution
            float sum = f[row][col]     * g00 + f[row][col+1]     * g01 + f[row][col+2]     * g02 +
                        f[row+1][col]   * g10 + f[row+1][col+1]   * g11 + f[row+1][col+2]   * g12 +
                        f[row+2][col]   * g20 + f[row+2][col+1]   * g21 + f[row+2][col+2]   * g22;

            output[i][j] = sum;
        }
    }
}

// Optimized 5x5 kernel OpenMP implementation with stride
void conv2d_5x5_stride_optimized_openmp(float **restrict f, int H, int W, float **restrict g,
                                         int sH, int sW, float **restrict output) {
    // Calculate actual input dimensions (remove padding)
    int original_H = H - 4;  // 5x5 kernel needs 4 pixels of padding removed
    int original_W = W - 4;

    // Calculate strided output dimensions
    int output_H = (int)ceil((double)original_H / sH);
    int output_W = (int)ceil((double)original_W / sW);

    // Cache all 25 kernel values for maximum performance
    const float g00 = g[0][0], g01 = g[0][1], g02 = g[0][2], g03 = g[0][3], g04 = g[0][4];
    const float g10 = g[1][0], g11 = g[1][1], g12 = g[1][2], g13 = g[1][3], g14 = g[1][4];
    const float g20 = g[2][0], g21 = g[2][1], g22 = g[2][2], g23 = g[2][3], g24 = g[2][4];
    const float g30 = g[3][0], g31 = g[3][1], g32 = g[3][2], g33 = g[3][3], g34 = g[3][4];
    const float g40 = g[4][0], g41 = g[4][1], g42 = g[4][2], g43 = g[4][3], g44 = g[4][4];

    // Parallel loop over output rows
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < output_H; i++) {
        int row = i * sH;
        for (int j = 0; j < output_W; j++) {
            int col = j * sW;

            // Fully unrolled 5x5 convolution
            float sum =
                f[row][col]     * g00 + f[row][col+1]     * g01 + f[row][col+2]     * g02 + f[row][col+3]     * g03 + f[row][col+4]     * g04 +
                f[row+1][col]   * g10 + f[row+1][col+1]   * g11 + f[row+1][col+2]   * g12 + f[row+1][col+3]   * g13 + f[row+1][col+4]   * g14 +
                f[row+2][col]   * g20 + f[row+2][col+1]   * g21 + f[row+2][col+2]   * g22 + f[row+2][col+3]   * g23 + f[row+2][col+4]   * g24 +
                f[row+3][col]   * g30 + f[row+3][col+1]   * g31 + f[row+3][col+2]   * g32 + f[row+3][col+3]   * g33 + f[row+3][col+4]   * g34 +
                f[row+4][col]   * g40 + f[row+4][col+1]   * g41 + f[row+4][col+2]   * g42 + f[row+4][col+3]   * g43 + f[row+4][col+4]   * g44;

            output[i][j] = sum;
        }
    }
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



// Use aligned_alloc for cache-line optimization (C11 standard)
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define ALIGNED_ALLOC_SUPPORTED
#endif

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64  // Modern systems use 64-byte cache lines
#endif

// Debug utility to check cache alignment
static inline int is_cache_aligned(const void *ptr) {
    return ((uintptr_t)ptr % CACHE_LINE_SIZE) == 0;
}


// Matrix utility functions (reused from assignment1)
float **allocate_matrix(int rows, int cols) {
    float **matrix = NULL;

#ifdef ALIGNED_ALLOC_SUPPORTED
    // Allocate row pointers with cache-line alignment
    matrix = (float **)aligned_alloc(CACHE_LINE_SIZE, rows * sizeof(float *));
#else
    // Fallback to malloc if aligned_alloc is not available
    matrix = (float **)malloc(rows * sizeof(float *));
#endif

    if (matrix == NULL) {
        fprintf(stderr, "Error: Unable to allocate matrix rows\n");
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
#ifdef ALIGNED_ALLOC_SUPPORTED
        // Allocate each row with cache-line alignment
        matrix[i] = (float *)aligned_alloc(CACHE_LINE_SIZE, cols * sizeof(float));
#else
        matrix[i] = (float *)malloc(cols * sizeof(float));
#endif
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

#ifdef DEBUG
    // Verify alignment in debug mode
    printf("Matrix allocation: rows=%s, first_row=%s (%d x %d)\n",
           is_cache_aligned(matrix) ? "aligned" : "unaligned",
           is_cache_aligned(matrix[0]) ? "aligned" : "unaligned",
           rows, cols);
#endif

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