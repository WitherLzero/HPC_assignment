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
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Calculate global output dimensions
    int global_output_H = (H - kH + 1 + sH - 1) / sH;
    int global_output_W = (W - kW + 1 + sW - 1) / sW;

    // Broadcast kernel to all processes
    float **local_kernel;
    if (rank == 0) {
        local_kernel = g;
    } else {
        local_kernel = NULL;
    }
    mpi_broadcast_kernel(&local_kernel, kH, kW, comm);

    // Distribute input matrix across processes
    float **local_matrix;
    int local_H, local_W, local_start_row;

    if (sH > 1 || sW > 1) {
        mpi_distribute_matrix_stride_aware(f, H, W, kH, kW, sH, sW, &local_matrix, &local_H, &local_W, &local_start_row, comm);
    } else {
        mpi_distribute_matrix(f, H, W, kH, kW, &local_matrix, &local_H, &local_W, &local_start_row, comm);
    }

    // Exchange halo regions with neighboring processes
    if (size > 1 && local_matrix != NULL && local_H > 0) {
        mpi_exchange_halos(local_matrix, local_H, local_W, kH, comm);
    }

    // Calculate local output dimensions and allocate output matrix
    int local_output_H = 0, local_output_W = 0;
    float **local_output = NULL;

    if (local_matrix != NULL && local_H > 0) {
        if (sH > 1 || sW > 1) {
            int halo_size = (kH - 1) / 2;
            int halo_top = (rank > 0) ? halo_size : 0;
            int halo_bottom = (rank < size - 1) ? halo_size : 0;
            int data_rows = local_H - halo_top - halo_bottom;

            local_output_H = (data_rows - kH + 1 + sH - 1) / sH;
            local_output_W = (local_W - kW + 1 + sW - 1) / sW;
        } else {
            local_output_H = (local_H - kH + 1 + sH - 1) / sH;
            local_output_W = (local_W - kW + 1 + sW - 1) / sW;
        }

        local_output = allocate_matrix(local_output_H, local_output_W);
    }


    // Perform local computation
    if (local_matrix != NULL && local_output != NULL && local_H > 0) {
        if (sH > 1 || sW > 1) {
            int halo_size = (kH - 1) / 2;
            int halo_top = (rank > 0) ? halo_size : 0;

            float **data_region = local_matrix + halo_top;
            int data_region_H = local_H - halo_top - ((rank < size - 1) ? halo_size : 0);

            conv2d_stride_serial(data_region, data_region_H, local_W, local_kernel, kH, kW, sH, sW, local_output);
        } else {
            conv2d_stride_serial(local_matrix, local_H, local_W, local_kernel, kH, kW, sH, sW, local_output);
        }
    }

    // Calculate output start row for this process
    int output_start_row;

    if (sH > 1 || sW > 1) {
        int output_base_rows = global_output_H / size;
        int output_remainder = global_output_H % size;
        int my_output_rows = output_base_rows + (rank < output_remainder ? 1 : 0);

        if (my_output_rows > 0) {
            output_start_row = rank * output_base_rows + (rank < output_remainder ? rank : output_remainder);
        } else {
            output_start_row = 0;
        }
    } else {
        int base_rows = H / size;
        int remainder = H % size;
        int halo_size = (kH - 1) / 2;

        output_start_row = 0;
        for (int p = 0; p < rank; p++) {
            int p_base_rows = base_rows + (p < remainder ? 1 : 0);
            int p_halo_top = (p > 0) ? halo_size : 0;
            int p_halo_bottom = (p < size - 1) ? halo_size : 0;
            int p_local_H = p_base_rows + p_halo_top + p_halo_bottom;
            int p_local_output_H = (p_local_H - kH + 1 + sH - 1) / sH;
            output_start_row += p_local_output_H;
        }
    }

    // Gather results from all processes
    MPI_Barrier(comm);

    float **gathered_output;
    mpi_gather_output(local_output, local_output_H, local_output_W, output_start_row,
                      &gathered_output, global_output_H, global_output_W, comm);

    // Copy gathered output to the provided output matrix (root only)
    if (rank == 0) {
        for (int i = 0; i < global_output_H; i++) {
            for (int j = 0; j < global_output_W; j++) {
                output[i][j] = gathered_output[i][j];
            }
        }
        free_matrix(gathered_output, global_output_H);
    }

    // Cleanup
    free_matrix(local_matrix, local_H);
    free_matrix(local_output, local_output_H);
    if (rank != 0) {
        free_matrix(local_kernel, kH);
    }
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

// MPI halo exchange function
void mpi_exchange_halos(float **local_matrix, int local_H, int local_W,
                       int kernel_H, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int halo_size = (kernel_H - 1) / 2;

    // Exchange with previous process (if exists)
    if (rank > 0) {
        // Send top data rows to previous process, receive our top halo from previous process
        for (int i = 0; i < halo_size; i++) {
            MPI_Sendrecv(local_matrix[halo_size + i], local_W, MPI_FLOAT, rank - 1, 100 + i,
                         local_matrix[i], local_W, MPI_FLOAT, rank - 1, 200 + i,
                         comm, MPI_STATUS_IGNORE);
        }
    }

    // Exchange with next process (if exists)
    if (rank < size - 1) {
        // Send bottom data rows to next process, receive our bottom halo from next process
        int data_end = local_H - halo_size - halo_size;
        for (int i = 0; i < halo_size; i++) {
            MPI_Sendrecv(local_matrix[data_end + i], local_W, MPI_FLOAT, rank + 1, 200 + i,
                         local_matrix[local_H - halo_size + i], local_W, MPI_FLOAT, rank + 1, 100 + i,
                         comm, MPI_STATUS_IGNORE);
        }
    }
}

// Stride-aware matrix distribution function
void mpi_distribute_matrix_stride_aware(float **global_matrix, int global_H, int global_W,
                                        int kernel_H, int kernel_W, int stride_H, int stride_W,
                                        float ***local_matrix, int *local_H, int *local_W,
                                        int *local_start_row, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Calculate output distribution first, then work backwards to input requirements
    int global_output_H = (global_H - kernel_H + 1 + stride_H - 1) / stride_H;

    // Distribute output rows evenly
    int output_base_rows = global_output_H / size;
    int output_remainder = global_output_H % size;
    int my_output_rows = output_base_rows + (rank < output_remainder ? 1 : 0);
    int my_output_start = rank * output_base_rows + (rank < output_remainder ? rank : output_remainder);

    // Handle processes with zero output rows
    if (my_output_rows == 0) {
        // Process has no work to do, allocate minimal buffer
        *local_H = 0;
        *local_W = global_W;
        *local_start_row = 0;
        *local_matrix = NULL;  // No allocation needed
        return;
    }

    // Calculate required input rows for this output range
    int input_start = my_output_start * stride_H;
    int input_end = (my_output_start + my_output_rows - 1) * stride_H + kernel_H;

    // Ensure we don't exceed global bounds
    if (input_end > global_H) input_end = global_H;

    int my_input_rows = input_end - input_start;
    *local_start_row = input_start;

    // Add halo regions for convolution boundary conditions
    int halo_size = (kernel_H - 1) / 2;
    int halo_top = (rank > 0) ? halo_size : 0;
    int halo_bottom = (rank < size - 1) ? halo_size : 0;

    *local_H = my_input_rows + halo_top + halo_bottom;
    *local_W = global_W;

    // Allocate local matrix
    *local_matrix = allocate_matrix(*local_H, *local_W);

    if (rank == 0) {
        // Root process: copy its own data and send to others
        int my_copy_rows = my_input_rows + halo_bottom;
        for (int i = 0; i < my_copy_rows; i++) {
            for (int j = 0; j < global_W; j++) {
                (*local_matrix)[i][j] = global_matrix[i][j];
            }
        }

        // Send data to other processes
        for (int p = 1; p < size; p++) {
            int p_output_base_rows = output_base_rows + (p < output_remainder ? 1 : 0);
            int p_output_start = p * output_base_rows + (p < output_remainder ? p : output_remainder);
            int p_input_start = p_output_start * stride_H;
            int p_input_end = (p_output_start + p_output_base_rows - 1) * stride_H + kernel_H;
            if (p_input_end > global_H) p_input_end = global_H;
            int p_input_rows = p_input_end - p_input_start;

            int p_halo_top = halo_size;
            int p_halo_bottom = (p < size - 1) ? halo_size : 0;
            int p_local_H = p_input_rows + p_halo_top + p_halo_bottom;

            // Calculate the range of global matrix rows to send
            int global_start = p_input_start - p_halo_top;
            int global_end = p_input_start + p_input_rows + p_halo_bottom;

            // Ensure we stay within bounds
            if (global_start < 0) global_start = 0;
            if (global_end > global_H) global_end = global_H;

            // Send each row within the valid range
            int local_row = 0;
            for (int global_row = global_start; global_row < global_end; global_row++) {
                MPI_Send(global_matrix[global_row], global_W, MPI_FLOAT, p, local_row, comm);
                local_row++;
            }

            // Send zero rows for any remaining local rows
            float *zero_row = (float*)calloc(global_W, sizeof(float));
            while (local_row < p_local_H) {
                MPI_Send(zero_row, global_W, MPI_FLOAT, p, local_row, comm);
                local_row++;
            }
            free(zero_row);
        }
    } else {
        // Non-root processes: receive their data
        for (int i = 0; i < *local_H; i++) {
            MPI_Recv((*local_matrix)[i], global_W, MPI_FLOAT, 0, i, comm, MPI_STATUS_IGNORE);
        }
    }
}

// Original matrix distribution function (kept for backward compatibility)
void mpi_distribute_matrix(float **global_matrix, int global_H, int global_W,
                           int kernel_H, int kernel_W,
                           float ***local_matrix, int *local_H, int *local_W,
                           int *local_start_row, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Distribute input rows evenly among processes
    int base_rows = global_H / size;
    int remainder = global_H % size;

    // Each process gets base_rows, first 'remainder' processes get one extra
    int my_rows = base_rows + (rank < remainder ? 1 : 0);
    *local_start_row = rank * base_rows + (rank < remainder ? rank : remainder);

    // Add halo regions for convolution boundary conditions
    int halo_size = (kernel_H - 1) / 2;
    int halo_top = (rank > 0) ? halo_size : 0;
    int halo_bottom = (rank < size - 1) ? halo_size : 0;

    *local_H = my_rows + halo_top + halo_bottom;
    *local_W = global_W;

    // Allocate local matrix
    *local_matrix = allocate_matrix(*local_H, *local_W);

    if (rank == 0) {
        // Root process: copy its own data and send to others

        // Copy root's own data (including bottom halo)
        int my_copy_rows = my_rows + halo_bottom;
        for (int i = 0; i < my_copy_rows; i++) {
            for (int j = 0; j < global_W; j++) {
                (*local_matrix)[i][j] = global_matrix[i][j];
            }
        }

        // Send data to other processes
        for (int p = 1; p < size; p++) {
            int p_base_rows = base_rows + (p < remainder ? 1 : 0);
            int p_start_row = p * base_rows + (p < remainder ? p : remainder);
            int p_halo_top = (p > 0) ? halo_size : 0;
            int p_halo_bottom = (p < size - 1) ? halo_size : 0;
            int p_local_H = p_base_rows + p_halo_top + p_halo_bottom;

            // Calculate the range of global matrix rows to send
            int global_start = p_start_row - p_halo_top;
            int global_end = p_start_row + p_base_rows + p_halo_bottom;

            // Ensure we stay within bounds
            if (global_start < 0) global_start = 0;
            if (global_end > global_H) global_end = global_H;

            // Send each row within the valid range
            int local_row = 0;
            for (int global_row = global_start; global_row < global_end; global_row++) {
                MPI_Send(global_matrix[global_row], global_W, MPI_FLOAT, p, local_row, comm);
                local_row++;
            }

            // Send zero rows for any remaining local rows
            float *zero_row = (float*)calloc(global_W, sizeof(float));
            while (local_row < p_local_H) {
                MPI_Send(zero_row, global_W, MPI_FLOAT, p, local_row, comm);
                local_row++;
            }
            free(zero_row);
        }
    } else {
        // Non-root processes: receive their data
        for (int i = 0; i < *local_H; i++) {
            MPI_Recv((*local_matrix)[i], global_W, MPI_FLOAT, 0, i, comm, MPI_STATUS_IGNORE);
        }
    }
}

void mpi_gather_output(float **local_output, int local_output_H, int local_output_W,
                       int local_start_row, float ***global_output,
                       int global_output_H, int global_output_W, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        // Root process: allocate global output matrix and collect from all processes
        *global_output = allocate_matrix(global_output_H, global_output_W);

        // Copy root's own output (if it has any)
        if (local_output != NULL && local_output_H > 0) {
            for (int i = 0; i < local_output_H; i++) {
                for (int j = 0; j < local_output_W; j++) {
                    (*global_output)[i][j] = local_output[i][j];
                }
            }
        }

        // Receive output from other processes
        for (int p = 1; p < size; p++) {
            // First, receive the dimensions and start row for this process
            int p_output_H, p_output_W, p_start_row;
            MPI_Recv(&p_output_H, 1, MPI_INT, p, 100, comm, MPI_STATUS_IGNORE);
            MPI_Recv(&p_output_W, 1, MPI_INT, p, 101, comm, MPI_STATUS_IGNORE);
            MPI_Recv(&p_start_row, 1, MPI_INT, p, 102, comm, MPI_STATUS_IGNORE);

            // Receive output data row by row with bounds checking
            for (int i = 0; i < p_output_H; i++) {
                int global_row = p_start_row + i;
                if (global_row >= 0 && global_row < global_output_H) {
                    MPI_Recv((*global_output)[global_row], p_output_W, MPI_FLOAT,
                            p, 200 + i, comm, MPI_STATUS_IGNORE);
                } else {
                    // Receive into temporary buffer and discard
                    float *temp_buffer = (float*)malloc(p_output_W * sizeof(float));
                    MPI_Recv(temp_buffer, p_output_W, MPI_FLOAT, p, 200 + i, comm, MPI_STATUS_IGNORE);
                    free(temp_buffer);
                }
            }
        }
    } else {
        // Non-root processes: send their output to root

        // First, send dimensions and start row (even if 0)
        // For processes with no output, use 0 as a safe start row
        int output_start_row = (local_output_H > 0) ? local_start_row : 0;
        MPI_Send(&local_output_H, 1, MPI_INT, 0, 100, comm);
        MPI_Send(&local_output_W, 1, MPI_INT, 0, 101, comm);
        MPI_Send(&output_start_row, 1, MPI_INT, 0, 102, comm);

        // Send output data row by row (only if we have output)
        if (local_output != NULL && local_output_H > 0) {
            for (int i = 0; i < local_output_H; i++) {
                MPI_Send(local_output[i], local_output_W, MPI_FLOAT, 0, 200 + i, comm);
            }
        }
    }
}

void mpi_broadcast_kernel(float ***kernel, int kernel_H, int kernel_W, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Broadcast kernel dimensions first
    MPI_Bcast(&kernel_H, 1, MPI_INT, 0, comm);
    MPI_Bcast(&kernel_W, 1, MPI_INT, 0, comm);

    // Use contiguous buffer for broadcast since allocate_matrix creates non-contiguous memory
    float *buffer = malloc(kernel_H * kernel_W * sizeof(float));

    if (rank == 0) {
        // Root: copy kernel to contiguous buffer
        for (int i = 0; i < kernel_H; i++) {
            for (int j = 0; j < kernel_W; j++) {
                buffer[i * kernel_W + j] = (*kernel)[i][j];
            }
        }
    }

    // Broadcast the contiguous buffer
    MPI_Bcast(buffer, kernel_H * kernel_W, MPI_FLOAT, 0, comm);

    if (rank != 0) {
        // Non-root: allocate matrix and copy from buffer
        *kernel = allocate_matrix(kernel_H, kernel_W);
        for (int i = 0; i < kernel_H; i++) {
            for (int j = 0; j < kernel_W; j++) {
                (*kernel)[i][j] = buffer[i * kernel_W + j];
            }
        }
    }

    free(buffer);
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