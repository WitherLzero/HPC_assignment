#ifndef CONV2D_MPI_H
#define CONV2D_MPI_H

#include <mpi.h>
#include <omp.h>
#include <unistd.h>
#include <stdbool.h>

/**
 * @brief Serial implementation of 2D convolution with stride
 *
 * @param f Input matrix
 * @param H Number of rows in input matrix
 * @param W Number of columns in input matrix
 * @param g Kernel matrix
 * @param kH Number of rows in kernel matrix
 * @param kW Number of columns in kernel matrix
 * @param sH Stride in height direction
 * @param sW Stride in width direction
 * @param output Output matrix
 */
void conv2d_stride_serial(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                          int sH, int sW, float **restrict output);

/**
 * @brief OpenMP parallel implementation of 2D convolution with stride
 *
 * @param f Input matrix
 * @param H Number of rows in input matrix
 * @param W Number of columns in input matrix
 * @param g Kernel matrix
 * @param kH Number of rows in kernel matrix
 * @param kW Number of columns in kernel matrix
 * @param sH Stride in height direction
 * @param sW Stride in width direction
 * @param output Output matrix
 */
void conv2d_stride_openmp(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                          int sH, int sW, float **restrict output);

/**
 * @brief Optimized 3x3 kernel OpenMP implementation with stride
 */
void conv2d_3x3_stride_optimized_openmp(float **restrict f, int H, int W, float **restrict g,
                                         int sH, int sW, float **restrict output);

/**
 * @brief Optimized 5x5 kernel OpenMP implementation with stride
 */
void conv2d_5x5_stride_optimized_openmp(float **restrict f, int H, int W, float **restrict g,
                                         int sH, int sW, float **restrict output);

/**
 * @brief MPI distributed memory implementation of 2D convolution with stride
 *
 * @param f Input feature map (padded)
 * @param H Global input height
 * @param W Global input width
 * @param g Kernel matrix
 * @param kH Number of rows in kernel matrix
 * @param kW Number of columns in kernel matrix
 * @param sH Stride in height direction
 * @param sW Stride in width direction
 * @param output Local output matrix
 * @param comm MPI communicator
 */
void conv2d_stride_mpi(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                       int sH, int sW, float **restrict output, MPI_Comm comm);

/**
 * @brief Hybrid MPI+OpenMP implementation of 2D convolution with stride
 *
 * @param f Input feature map (padded)
 * @param H Global input height
 * @param W Global input width
 * @param g Kernel matrix
 * @param kH Number of rows in kernel matrix
 * @param kW Number of columns in kernel matrix
 * @param sH Stride in height direction
 * @param sW Stride in width direction
 * @param output Local output matrix
 * @param comm MPI communicator
 */
void conv2d_stride_hybrid(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                          int sH, int sW, float **restrict output, MPI_Comm comm);

/**
 * @brief Calculate output dimensions for strided convolution
 *
 * @param input_H Input height
 * @param input_W Input width
 * @param stride_H Height stride
 * @param stride_W Width stride
 * @param output_H Pointer to store output height
 * @param output_W Pointer to store output width
 */
void calculate_stride_output_dims(int input_H, int input_W, int stride_H, int stride_W,
                                  int *output_H, int *output_W);

/**
 * @brief Calculate padding requirements for a specific process in distributed computation
 *
 * Determines the padding needed for top/bottom/left/right based on process position:
 * - First process (rank 0): Top gets "same" padding, bottom gets halo
 * - Middle processes: Both top and bottom get halo padding
 * - Last process: Top gets halo, bottom gets "same" padding
 * - All processes: Left/right get "same" padding
 *
 * @param rank Process rank in communicator
 * @param size Total number of processes in communicator
 * @param kH Kernel height
 * @param kW Kernel width
 * @param pad_top Pointer to store top padding size
 * @param pad_bottom Pointer to store bottom padding size
 * @param pad_left Pointer to store left padding size
 * @param pad_right Pointer to store right padding size
 */
void calculate_padding_for_process(
    int rank, int size,
    int kH, int kW,
    int* pad_top, int* pad_bottom,
    int* pad_left, int* pad_right
);

/**
 * @brief Calculate local matrix dimensions for distributed computation
 *
 * Determines how many rows each process should handle and the dimensions
 * of the local padded matrix including halo regions.
 *
 * @param rank Process rank in communicator
 * @param size Total number of processes in communicator
 * @param H_global Global matrix height (without padding)
 * @param W_global Global matrix width (without padding)
 * @param kH Kernel height (for padding calculation)
 * @param kW Kernel width (for padding calculation)
 * @param local_H Pointer to store local matrix height (without padding)
 * @param local_W Pointer to store local matrix width (without padding)
 * @param local_start_row Pointer to store starting row in global coordinates
 * @param padded_local_H Pointer to store local matrix height (with padding)
 * @param padded_local_W Pointer to store local matrix width (with padding)
 */
void calculate_local_dimensions(
    int rank, int size,
    int H_global, int W_global,
    int kH, int kW,
    int* local_H, int* local_W,
    int* local_start_row,
    int* padded_local_H, int* padded_local_W
);

/**
 * @brief Calculate local matrix dimensions for distributed computation (stride-aware)
 *
 * Uses INPUT-FIRST distribution strategy: distributes input rows evenly across
 * processes, then each process computes output based on stride parameters.
 * This approach works correctly for both stride=1 and stride>1 cases.
 *
 * @param rank Process rank in communicator
 * @param size Total number of processes in communicator
 * @param H_global Global matrix height (without padding)
 * @param W_global Global matrix width (without padding)
 * @param kH Kernel height (for padding calculation)
 * @param kW Kernel width (for padding calculation)
 * @param sH Stride in height direction
 * @param sW Stride in width direction
 * @param local_H Pointer to store local matrix height (without padding)
 * @param local_W Pointer to store local matrix width (without padding)
 * @param local_start_row Pointer to store starting row in global coordinates
 * @param padded_local_H Pointer to store local matrix height (with padding)
 * @param padded_local_W Pointer to store local matrix width (with padding)
 */
void calculate_local_dimensions_stride_aware(
    int rank, int size,
    int H_global, int W_global,
    int kH, int kW,
    int sH, int sW,
    int* local_H, int* local_W,
    int* local_start_row,
    int* padded_local_H, int* padded_local_W
);

/**
 * @brief Distribute input matrix across MPI processes with necessary overlap for convolution
 *
 * @param global_matrix Global input matrix (only valid on root)
 * @param global_H Global matrix height
 * @param global_W Global matrix width
 * @param kernel_H Kernel height (for overlap calculation)
 * @param kernel_W Kernel width (for overlap calculation)
 * @param local_matrix Pointer to store local matrix portion
 * @param local_H Pointer to store local matrix height
 * @param local_W Pointer to store local matrix width
 * @param local_start_row Pointer to store starting row in global coordinates
 * @param comm MPI communicator
 */
void mpi_distribute_matrix(float **global_matrix, int global_H, int global_W,
                           int kernel_H, int kernel_W,
                           float ***local_matrix, int *local_H, int *local_W,
                           int *local_start_row, MPI_Comm comm);

/**
 * @brief Stride-aware distribution of input matrix across MPI processes
 *
 * @param global_matrix Global input matrix (only valid on root)
 * @param global_H Global matrix height
 * @param global_W Global matrix width
 * @param kernel_H Kernel height (for overlap calculation)
 * @param kernel_W Kernel width (for overlap calculation)
 * @param stride_H Stride in height direction
 * @param stride_W Stride in width direction
 * @param local_matrix Pointer to store local matrix portion
 * @param local_H Pointer to store local matrix height
 * @param local_W Pointer to store local matrix width
 * @param local_start_row Pointer to store starting row in global coordinates
 * @param comm MPI communicator
 */
void mpi_distribute_matrix_stride_aware(float **global_matrix, int global_H, int global_W,
                                        int kernel_H, int kernel_W, int stride_H, int stride_W,
                                        float ***local_matrix, int *local_H, int *local_W,
                                        int *local_start_row, MPI_Comm comm);

/**
 * @brief Gather output matrices from all MPI processes
 *
 * @param local_output Local output matrix
 * @param local_output_H Local output height
 * @param local_output_W Local output width
 * @param local_start_row Starting row in global output coordinates
 * @param global_output Pointer to store global output matrix (valid on root)
 * @param global_output_H Global output height
 * @param global_output_W Global output width
 * @param comm MPI communicator
 */
void mpi_gather_output(float **local_output, int local_output_H, int local_output_W,
                       int local_start_row, float ***global_output,
                       int global_output_H, int global_output_W, MPI_Comm comm);

/**
 * @brief Broadcast kernel matrix to all MPI processes
 *
 * @param kernel Kernel matrix
 * @param kernel_H Kernel height
 * @param kernel_W Kernel width
 * @param comm MPI communicator
 */
void mpi_broadcast_kernel(float ***kernel, int kernel_H, int kernel_W, MPI_Comm comm);

/**
 * @brief Exchange halo regions between neighboring MPI processes
 *
 * @param local_matrix Local matrix with halo regions
 * @param local_H Local matrix height (including halos)
 * @param local_W Local matrix width
 * @param kernel_H Kernel height (determines halo size)
 * @param comm MPI communicator
 */
void mpi_exchange_halos(float **local_matrix, int local_H, int local_W,
                       int kernel_H, MPI_Comm comm);

// Matrix utility functions (reused from assignment1)
float **allocate_matrix(int rows, int cols);
void free_matrix(float **matrix, int rows);
void initialize_matrix(float **matrix, int rows, int cols, float value);
void generate_padded_matrix(float **input, int height, int width,
                            int kernel_height, int kernel_width,
                            float ***padded, int *padded_height,
                            int *padded_width);

// Matrix I/O functions with MPI support
int mpi_read_matrix_from_file(const char *filename, float ***matrix, int *rows,
                              int *cols, MPI_Comm comm);
int mpi_write_matrix_to_file(const char *filename, float **matrix, int rows,
                             int cols, MPI_Comm comm);
void mpi_print_matrix(float **matrix, int rows, int cols, MPI_Comm comm);

// Serial versions for testing without MPI
int read_matrix_from_file(const char *filename, float ***matrix, int *rows, int *cols);
int write_matrix_to_file(const char *filename, float **matrix, int rows, int cols);
void print_matrix(float **matrix, int rows, int cols);
int compare_matrices(float **matrix1, float **matrix2, int rows, int cols, float tolerance);
// Matrix generation functions with MPI support
float **mpi_generate_random_matrix(int rows, int cols, float min_val,
                                   float max_val, MPI_Comm comm);

// Serial version for kernel generation
float **generate_random_matrix(int rows, int cols, float min_val, float max_val);

/**
 * @brief Generate local portion of padded matrix for distributed computation
 *
 * This function generates a local portion of a matrix with padding, suitable for
 * distributed 2D convolution. Each process generates only its assigned rows plus
 * necessary padding. Uses deterministic position-based seeding for reproducibility.
 *
 * @param H_global Global matrix height (without padding)
 * @param W_global Global matrix width (without padding)
 * @param kH Kernel height (for padding calculation)
 * @param kW Kernel width (for padding calculation)
 * @param padded_local_H Returned local matrix height (with padding)
 * @param padded_local_W Returned local matrix width (with padding)
 * @param local_start_row Starting row in global coordinates
 * @param min_val Minimum random value (inclusive)
 * @param max_val Maximum random value (exclusive)
 * @param comm MPI communicator (typically active_comm)
 *
 * @return Allocated padded matrix with generated values
 *
 * @note Padding regions are initialized to zero. Use mpi_exchange_halos()
 *       to fill with actual neighbor data.
 * @note Uses deterministic seeding: same global matrix regardless of process count
 */
float** mpi_generate_local_padded_matrix(
    int H_global, int W_global,
    int kH, int kW,
    int sH, int sW,
    int* padded_local_H,
    int* padded_local_W,
    int* local_start_row,
    float min_val, float max_val,
    MPI_Comm comm
);

/**
 * @brief Read local portion of padded matrix using MPI Parallel I/O
 *
 * Each process reads only its local portion from the file using MPI_File_read_at().
 * The matrix is allocated with padding and initialized with zeros for padding regions.
 * After reading, use mpi_exchange_halos() to fill halo regions with neighbor data.
 *
 * @param filename Input file path
 * @param H_global Global matrix height (output parameter, read from file)
 * @param W_global Global matrix width (output parameter, read from file)
 * @param kH Kernel height (for padding calculation)
 * @param kW Kernel width (for padding calculation)
 * @param padded_local_H Local matrix height with padding (output)
 * @param padded_local_W Local matrix width with padding (output)
 * @param local_start_row Starting row in global coordinates (output)
 * @param comm MPI communicator (typically active_comm)
 *
 * @return Allocated padded matrix with data read from file, NULL on error
 *
 * @note Padding regions are initialized to zero. Use mpi_exchange_halos()
 *       to fill halo regions with actual neighbor data.
 * @note Uses MPI Parallel I/O for efficient distributed reading
 */
float** mpi_read_local_padded_matrix(
    const char* filename,
    int* H_global, int* W_global,
    int kH, int kW,
    int sH, int sW,
    int* padded_local_H,
    int* padded_local_W,
    int* local_start_row,
    MPI_Comm comm
);

/**
 * @brief Write local padded input matrix to file using MPI Parallel I/O
 *
 * Each process writes only its data portion (excluding padding/halo) directly to the file.
 * This avoids the wasteful approach of extracting padding, gathering to root, then writing.
 *
 * @param filename Output file path
 * @param local_padded_matrix Local padded matrix
 * @param padded_local_H Local matrix height (with padding)
 * @param padded_local_W Local matrix width (with padding)
 * @param local_start_row Starting row in global coordinates
 * @param H_global Global matrix height (without padding)
 * @param W_global Global matrix width (without padding)
 * @param kH Kernel height (for padding calculation)
 * @param kW Kernel width (for padding calculation)
 * @param comm MPI communicator (typically active_comm)
 *
 * @return 0 on success, -1 on error
 *
 * @note Writes only the data portion, skipping padding/halo regions
 * @note Uses MPI Parallel I/O for efficient distributed writing
 */
int mpi_write_input_parallel(
    const char* filename,
    float **local_padded_matrix,
    int padded_local_H, int padded_local_W,
    int local_start_row,
    int H_global, int W_global,
    int kH, int kW,
    MPI_Comm comm
);

/**
 * @brief Compare two matrices element-wise within an absolute tolerance (MPI-aware)
 *
 * @param matrix1 Pointer to the first matrix (size rows x cols)
 * @param matrix2 Pointer to the second matrix (size rows x cols)
 * @param rows Number of rows in both matrices
 * @param cols Number of columns in both matrices
 * @param tolerance Maximum allowed absolute difference per element
 * @param comm MPI communicator
 * @return int 1 if matrices are equal within tolerance, 0 otherwise
 */
int mpi_compare_matrices(float **matrix1, float **matrix2, int rows, int cols,
                         float tolerance, MPI_Comm comm);

// Timing utilities for MPI
typedef struct {
    double start_time;
    double end_time;
    double elapsed_time;
    double max_time;    // Maximum time across all processes
    double min_time;    // Minimum time across all processes
    double avg_time;    // Average time across all processes
} mpi_timer_t;

void mpi_timer_start(mpi_timer_t *timer);
void mpi_timer_end(mpi_timer_t *timer, MPI_Comm comm);
void mpi_timer_print(mpi_timer_t *timer, const char *description, MPI_Comm comm);

#endif  // CONV2D_MPI_H