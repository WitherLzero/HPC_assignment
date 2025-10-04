#include "conv2d_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define OMP_CHUNK_SIZE 8
#define PREFETCH_DISTANCE 2
#define MIN_PARALLEL_SIZE 100

void calculate_stride_output_dims(int input_H, int input_W, int stride_H, int stride_W,
                                  int *output_H, int *output_W) {
    *output_H = (int)ceil((double)input_H / stride_H);
    *output_W = (int)ceil((double)input_W / stride_W);
}

void calculate_padding_for_process(
    int rank, int size,
    int kH, int kW,
    int* pad_top, int* pad_bottom,
    int* pad_left, int* pad_right
) {
    int halo = (kH - 1) / 2;

    // Top padding
    if (rank == 0) {
        *pad_top = (kH - 1) / 2;        // "Same" padding for first process
    } else {
        *pad_top = halo;                // Halo (will be filled by exchange)
    }

    // Bottom padding
    if (rank == size - 1) {
        *pad_bottom = kH - 1 - (kH - 1) / 2;  // "Same" padding (asymmetric)
    } else {
        *pad_bottom = halo;             // Halo (will be filled by exchange)
    }

    // Horizontal padding (all processes same - "same" padding)
    *pad_left = (kW - 1) / 2;
    *pad_right = kW - 1 - (kW - 1) / 2;
}

void calculate_local_dimensions(
    int rank, int size,
    int H_global, int W_global,
    int kH, int kW,
    int* local_H, int* local_W,
    int* local_start_row,
    int* padded_local_H, int* padded_local_W
) {
    // OUTPUT-FIRST DISTRIBUTION: Distribute output rows evenly, then calculate input needs
    int output_H = H_global;
    int output_W = W_global;

    int base_output_rows = output_H / size;
    int output_remainder = output_H % size;
    int my_output_rows = base_output_rows + (rank < output_remainder ? 1 : 0);
    int output_start_row = rank * base_output_rows + (rank < output_remainder ? rank : output_remainder);

    // Calculate padding requirements for this process
    int pad_top, pad_bottom, pad_left, pad_right;
    calculate_padding_for_process(rank, size, kH, kW,
                                   &pad_top, &pad_bottom,
                                   &pad_left, &pad_right);

    // Padded input size: my_output_rows + (kH - 1)
    *padded_local_H = my_output_rows + (kH - 1);
    *padded_local_W = W_global + (kW - 1);

    *local_H = *padded_local_H - pad_top - pad_bottom;
    *local_start_row = output_start_row;
    *local_W = W_global;
}

void calculate_local_dimensions_stride_aware(
    int rank, int size,
    int H_global, int W_global,
    int kH, int kW,
    int sH, int sW,
    int* local_H, int* local_W,
    int* local_start_row,
    int* padded_local_H, int* padded_local_W
) {
    // STRIDE-AWARE DISTRIBUTION: Operates on padded matrix, maps back to file coordinates
    // H_global is ORIGINAL unpadded size from file
    int H_global_padded = H_global + (kH - 1);
    int W_global_padded = W_global + (kW - 1);

    int global_output_H = (H_global_padded - kH + 1 + sH - 1) / sH;

    // Distribute output rows evenly
    int output_base_rows = global_output_H / size;
    int output_remainder = global_output_H % size;
    int my_output_rows = output_base_rows + (rank < output_remainder ? 1 : 0);
    int my_output_start = rank * output_base_rows + (rank < output_remainder ? rank : output_remainder);

    if (my_output_rows == 0) {
        *local_H = 0;
        *local_W = W_global;
        *local_start_row = 0;
        *padded_local_H = 0;
        *padded_local_W = W_global + kW - 1;
        return;
    }

    // Calculate input rows needed for this output range
    int input_start = my_output_start * sH;
    int input_end = (my_output_start + my_output_rows - 1) * sH + kH;
    if (input_end > H_global_padded) input_end = H_global_padded;

    int my_input_rows = input_end - input_start;

    // Add halo regions for convolution overlap
    int halo_size = (kH - 1) / 2;
    int halo_top = (rank > 0) ? halo_size : 0;
    int halo_bottom = (rank < size - 1) ? halo_size : 0;

    // Map padded coordinates to file coordinates
    int pad_offset = (kH - 1) / 2;

    int padded_range_start = input_start - halo_top;
    int padded_range_end = input_end + halo_bottom;
    if (padded_range_start < 0) padded_range_start = 0;
    if (padded_range_end > H_global_padded) padded_range_end = H_global_padded;

    int file_read_start = padded_range_start - pad_offset;
    int file_read_end = padded_range_end - pad_offset;
    if (file_read_start < 0) file_read_start = 0;
    if (file_read_end > H_global) file_read_end = H_global;

    *local_start_row = file_read_start;
    *local_H = file_read_end - file_read_start;
    *local_W = W_global;
    *padded_local_H = my_input_rows + halo_top + halo_bottom;
    *padded_local_W = W_global_padded;
}


void conv2d_stride_serial(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                          int sH, int sW, float **restrict output) {
    int original_H = H - kH + 1;
    int original_W = W - kW + 1;

    int output_H, output_W;
    calculate_stride_output_dims(original_H, original_W, sH, sW, &output_H, &output_W);

    for (int i = 0; i < output_H; i++) {
        for (int j = 0; j < output_W; j++) {
            float sum = 0.0f;
            int start_row = i * sH;
            int start_col = j * sW;

            if (start_row + kH <= H && start_col + kW <= W) {
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

void conv2d_stride_openmp(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                          int sH, int sW, float **restrict output) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("OpenMP General: Using %d threads\n", omp_get_num_threads());
        }
    }

    int original_H = H - kH + 1;
    int original_W = W - kW + 1;
    int output_H = (int)ceil((double)original_H / sH);
    int output_W = (int)ceil((double)original_W / sW);

    // Use serial for small matrices
    if (output_H * output_W < MIN_PARALLEL_SIZE) {
        conv2d_stride_serial(f, H, W, g, kH, kW, sH, sW, output);
        return;
    }

    // Optimized kernels for common sizes
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
            int next_row = (i + PREFETCH_DISTANCE) * sH;
            if (next_row < H) {
                __builtin_prefetch(&f[next_row][0], 0, 3);
            }

            for (int j = 0; j < output_W; j++) {
                float sum = 0.0f;
                int start_row = i * sH;
                int start_col = j * sW;

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
    int original_H = H - 2;
    int original_W = W - 2;
    int output_H = (int)ceil((double)original_H / sH);
    int output_W = (int)ceil((double)original_W / sW);

    // Cache kernel values
    const float g00 = g[0][0], g01 = g[0][1], g02 = g[0][2];
    const float g10 = g[1][0], g11 = g[1][1], g12 = g[1][2];
    const float g20 = g[2][0], g21 = g[2][1], g22 = g[2][2];

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < output_H; i++) {
        int row = i * sH;
        for (int j = 0; j < output_W; j++) {
            int col = j * sW;
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

// MPI-only strided convolution 
void conv2d_stride_mpi(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                       int sH, int sW, float **restrict output, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Perform halo exchange only for stride=1 cases
    // For stride>1, data is already distributed with necessary overlaps via file reading
    if (size > 1 ) {
        mpi_exchange_halos(f, H, W, kH, comm);
    }


    // MPI-only: single-threaded computation on local data
    if (sH > 1 || sW > 1) {
        // For stride > 1: Skip halo region, compute only on data region
        int halo_size = (kH - 1) / 2;
        int halo_top = (rank > 0) ? halo_size : 0;
        int halo_bottom = (rank < size - 1) ? halo_size : 0;

        // Point to data region (skip halo_top rows)
        float **data_region = f + halo_top;
        int data_region_H = H - halo_top - halo_bottom;

        conv2d_stride_serial(data_region, data_region_H, W, g, kH, kW, sH, sW, output);
    } else {
        conv2d_stride_serial(f, H, W, g, kH, kW, sH, sW, output);
    }
}

// Hybrid MPI+OpenMP strided convolution 
void conv2d_stride_hybrid(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                          int sH, int sW, float **restrict output, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size > 1) {
        mpi_exchange_halos(f, H, W, kH, comm);
    }

    if (sH > 1 || sW > 1) {
        int halo_size = (kH - 1) / 2;
        int halo_top = (rank > 0) ? halo_size : 0;
        int halo_bottom = (rank < size - 1) ? halo_size : 0;

        float **data_region = f + halo_top;
        int data_region_H = H - halo_top - halo_bottom;

        conv2d_stride_openmp(data_region, data_region_H, W, g, kH, kW, sH, sW, output);
    } else {
        conv2d_stride_openmp(f, H, W, g, kH, kW, sH, sW, output);
    }
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

    // DEBUG: Print the above info in each process
    if(rank == 0)
        printf("Global output rows: %d (base %d + remainder %d)\n", global_output_H, output_base_rows, output_remainder);
    
    printf("Rank %d/%d: output rows %d starting at %d\n", rank, size, my_output_rows, my_output_start);

    MPI_Barrier(comm);

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

// Gather local output matrices back to global output matrix
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

// Broadcast kernel to all processes
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