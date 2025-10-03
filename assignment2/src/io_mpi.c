#include "conv2d_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <math.h>

// CAN BE REMOVED ??
int mpi_read_matrix_from_file(const char *filename, float ***matrix, int *rows,
                              int *cols, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        // Only root process reads the file
        FILE *file = fopen(filename, "r");
        if (file == NULL) {
            fprintf(stderr, "Error: Cannot open file %s\n", filename);
            return -1;
        }

        if (fscanf(file, "%d %d", rows, cols) != 2) {
            fprintf(stderr, "Error: Invalid file format in %s\n", filename);
            fclose(file);
            return -1;
        }

        *matrix = allocate_matrix(*rows, *cols);
        if (*matrix == NULL) {
            fprintf(stderr, "Error: Cannot allocate matrix\n");
            fclose(file);
            return -1;
        }

        for (int i = 0; i < *rows; i++) {
            for (int j = 0; j < *cols; j++) {
                if (fscanf(file, "%f", &((*matrix)[i][j])) != 1) {
                    fprintf(stderr, "Error: Cannot read matrix element [%d][%d]\n", i, j);
                    free_matrix(*matrix, *rows);
                    fclose(file);
                    return -1;
                }
            }
        }
        fclose(file);
    }

    // Broadcast dimensions to all processes
    MPI_Bcast(rows, 1, MPI_INT, 0, comm);
    MPI_Bcast(cols, 1, MPI_INT, 0, comm);

    // Non-root processes allocate matrix
    if (rank != 0) {
        *matrix = allocate_matrix(*rows, *cols);
    }

    // Broadcast matrix data
    for (int i = 0; i < *rows; i++) {
        MPI_Bcast((*matrix)[i], *cols, MPI_FLOAT, 0, comm);
    }

    return 0;
}

// CAN BE REMOVED ??
int mpi_write_matrix_to_file(const char *filename, float **matrix, int rows,
                             int cols, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        // Only root process writes the file
        FILE *file = fopen(filename, "w");
        if (file == NULL) {
            fprintf(stderr, "Error: Cannot create file %s\n", filename);
            return -1;
        }

        fprintf(file, "%d %d\n", rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(file, "%.3f", matrix[i][j]);
                if (j < cols - 1) fprintf(file, " ");
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }

    return 0;
}

// CAN BE REMOVED ??
void mpi_print_matrix(float **matrix, int rows, int cols, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        // Only root process prints
        printf("Matrix (%dx%d):\n", rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%.3f", matrix[i][j]);
                if (j < cols - 1) printf(" ");
            }
            printf("\n");
        }
        printf("\n");
    }
}

// Fast random number generator using xorshift algorithm
static inline unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// CAN BE REMOVED ??? 
float **mpi_generate_random_matrix(int rows, int cols, float min_val,
                                   float max_val, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    float **matrix = allocate_matrix(rows, cols);
    if (matrix == NULL) {
        return NULL;
    }

    if (rank == 0) {
        // Only root process generates random values using optimized approach
        // Pre-calculate range for efficiency
        const float range = max_val - min_val;
        const float inv_max = 1.0f / (float)UINT_MAX;
        const float scale = range * inv_max;

        // Use OpenMP for parallel generation
        #pragma omp parallel
        {
            // Each thread gets its own random state
            unsigned int seed = (unsigned int)(time(NULL) + omp_get_thread_num() * 12345);

            #pragma omp for collapse(2) schedule(static)
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    // Generate random float between min_val and max_val using fast xorshift
                    unsigned int rand_int = xorshift32(&seed);
                    matrix[i][j] = ((float)rand_int * scale) + min_val;
                }
            }
        }
    }

    return matrix;
}

// ===================================================================
// Direct Local Padded Matrix Generation (Memory-Optimized)
// ===================================================================

float** mpi_generate_local_padded_matrix(
    int H_global, int W_global,
    int kH, int kW,
    int* padded_local_H,
    int* padded_local_W,
    int* local_start_row,
    float min_val, float max_val,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Calculate local dimensions and padding using Phase 2 utilities
    int local_H, local_W;
    calculate_local_dimensions(rank, size, H_global, W_global, kH, kW,
                               &local_H, &local_W, local_start_row,
                               padded_local_H, padded_local_W);

    int pad_top, pad_bottom, pad_left, pad_right;
    calculate_padding_for_process(rank, size, kH, kW,
                                   &pad_top, &pad_bottom,
                                   &pad_left, &pad_right);

    // Allocate padded matrix
    float **matrix = allocate_matrix(*padded_local_H, *padded_local_W);
    if (matrix == NULL) {
        fprintf(stderr, "Error: Rank %d failed to allocate local padded matrix\n", rank);
        return NULL;
    }

    // Initialize all to zero (padding regions)
    initialize_matrix(matrix, *padded_local_H, *padded_local_W, 0.0f);

    // Generate data region with random values using xorshift
    const float range = max_val - min_val;
    const float inv_max = 1.0f / (float)UINT_MAX;
    const float scale = range * inv_max;

    //#pragma omp parallel
    {
        // Each thread gets its own random seed
        unsigned int seed = (unsigned int)(time(NULL) + rank * 12345);
                                       // + omp_get_thread_num() * 67890);

        //#pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < local_H; i++) {
            for (int j = 0; j < local_W; j++) {
                // Generate random float using xorshift32
                unsigned int rand_int = xorshift32(&seed);
                float value = min_val + ((float)rand_int * scale);
                matrix[i + pad_top][j + pad_left] = value;
            }
        }
    }

    return matrix;
}


// ===================================================================
// MPI Parallel I/O - File Reading into Padded
// ===================================================================

float** mpi_read_local_padded_matrix(
    const char* filename,
    int* H_global, int* W_global,
    int kH, int kW,
    int* padded_local_H,
    int* padded_local_W,
    int* local_start_row,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPI_File fh;
    int result = MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (result != MPI_SUCCESS) {
        if (rank == 0) {
            fprintf(stderr, "Error: Cannot open file %s for reading\n", filename);
        }
        return NULL;
    }

    // Root reads and broadcasts dimensions
    if (rank == 0) {
        char header[256];
        MPI_Status status;
        MPI_File_read(fh, header, 256, MPI_CHAR, &status);

        // Parse dimensions from header
        if (sscanf(header, "%d %d", H_global, W_global) != 2) {
            fprintf(stderr, "Error: Invalid file format in %s\n", filename);
            MPI_File_close(&fh);
            return NULL;
        }
    }

    // Broadcast dimensions to all processes
    MPI_Bcast(H_global, 1, MPI_INT, 0, comm);
    MPI_Bcast(W_global, 1, MPI_INT, 0, comm);

    // Calculate local dimensions and padding
    int local_H, local_W;
    calculate_local_dimensions(rank, size, *H_global, *W_global, kH, kW,
                               &local_H, &local_W, local_start_row,
                               padded_local_H, padded_local_W);

    int pad_top, pad_bottom, pad_left, pad_right;
    calculate_padding_for_process(rank, size, kH, kW,
                                   &pad_top, &pad_bottom,
                                   &pad_left, &pad_right);

    // Allocate and initialize padded matrix (all zeros)
    float **matrix = allocate_matrix(*padded_local_H, *padded_local_W);
    if (matrix == NULL) {
        fprintf(stderr, "Error: Rank %d failed to allocate local padded matrix\n", rank);
        MPI_File_close(&fh);
        return NULL;
    }
    initialize_matrix(matrix, *padded_local_H, *padded_local_W, 0.0f);

    // Calculate header size (find first newline position)
    MPI_Offset header_size = 0;
    if (rank == 0) {
        char c;
        MPI_Offset pos = 0;
        MPI_Status status;
        do {
            MPI_File_read_at(fh, pos, &c, 1, MPI_CHAR, &status);
            pos++;
        } while (c != '\n');
        header_size = pos;
    }
    MPI_Bcast(&header_size, 1, MPI_OFFSET, 0, comm);

    // File format: "0.xxx " per float (6 chars: "0.xxx " including space and decimal)
    // Last column has no trailing space: "0.xxx" (5 chars)
    // Total: (W-1)*6 + 5 = W*6 - 1
    int chars_per_float = 6;
    int chars_per_row = *W_global * chars_per_float - 1;

    // Read local rows from file
    for (int i = 0; i < local_H; i++) {
        int global_row = *local_start_row + i;
        int padded_row = i + pad_top;

        // Calculate offset for this row (account for \r\n = 2 bytes or \n = 1 byte)
        // We'll use 2 bytes for safety (Windows format)
        MPI_Offset row_offset = header_size + global_row * (chars_per_row + 2);

        // Allocate buffer for row (chars_per_row + 1 for safety)
        char row_buffer[10000];  // Should be enough for most cases
        if (chars_per_row >= 10000) {
            fprintf(stderr, "Error: Row too large (%d chars)\n", chars_per_row);
            free_matrix(matrix, *padded_local_H);
            MPI_File_close(&fh);
            return NULL;
        }

        MPI_Status status;
        MPI_File_read_at(fh, row_offset, row_buffer, chars_per_row,
                        MPI_CHAR, &status);
        row_buffer[chars_per_row] = '\0';  // Null-terminate

        #ifdef DEBUG
        printf("Rank %d: Row %d buffer [%d chars]: '%s'\n", rank, global_row, chars_per_row, row_buffer);
        #endif

        // Parse row into matrix (skip padding columns)
        char *ptr = row_buffer;
        for (int j = 0; j < *W_global; j++) {
            float value;
            int chars_read;
            if (sscanf(ptr, "%f%n", &value, &chars_read) == 1) {
                matrix[padded_row][j + pad_left] = value;
                ptr += chars_read;
            } else {
                #ifdef DEBUG
                fprintf(stderr, "Rank %d: Error parsing row %d col %d, remaining buffer: '%s'\n",
                        rank, global_row, j, ptr);
                #endif
            }
        }
    }

    MPI_File_close(&fh);
    return matrix;
}


// ===================================================================
// MPI Parallel I/O - File Writing input and output
// ===================================================================

int mpi_write_input_parallel(
    const char* filename,
    float **local_padded_matrix,
    int padded_local_H, int padded_local_W,
    int local_start_row,
    int H_global, int W_global,
    int kH, int kW,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);


    // Calculate padding to know which rows/cols are actual data
    int pad_top, pad_bottom, pad_left, pad_right;
    calculate_padding_for_process(rank, size, kH, kW,
                                   &pad_top, &pad_bottom,
                                   &pad_left, &pad_right);


    // Calculate local data dimensions (without padding)
    int local_H = padded_local_H - pad_top - pad_bottom;
    int local_W = padded_local_W - pad_left - pad_right;

    // Open file for writing
    MPI_File fh;
    int result = MPI_File_open(comm, filename,
                              MPI_MODE_CREATE | MPI_MODE_WRONLY,
                              MPI_INFO_NULL, &fh);
    if (result != MPI_SUCCESS) {
        if (rank == 0) {
            fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        }
        return -1;
    }

    // Root writes header
    MPI_Offset header_size = 0;
    if (rank == 0) {
        char header[256];
        int len = sprintf(header, "%d %d\r\n", H_global, W_global);
        MPI_File_write(fh, header, len, MPI_CHAR, MPI_STATUS_IGNORE);
        header_size = len;
    }
    MPI_Bcast(&header_size, 1, MPI_OFFSET, 0, comm);

    MPI_Barrier(comm);  // Ensure header is written

    // File format: "0.xxx " per float (6 chars), last column "0.xxx" (5 chars)
    // Total per row: (W-1)*6 + 5 = W*6 - 1, plus \r\n = W*6 + 1
    int chars_per_row = W_global * 6 + 1;  // Include \r\n

    // Write each local data row
    for (int i = 0; i < local_H; i++) {
        int global_row = local_start_row + i;
        int padded_row = i + pad_top;

        // Calculate file offset for this row
        MPI_Offset row_offset = header_size + global_row * chars_per_row;

        // Build row string
        char row_buffer[10000];
        int pos = 0;

        for (int j = 0; j < local_W; j++) {
            int padded_col = j + pad_left;
            float value = local_padded_matrix[padded_row][padded_col];

            if (j < local_W - 1) {
                // All but last column: "0.xxx "
                pos += sprintf(&row_buffer[pos], "%.3f ", value);
            } else {
                // Last column: "0.xxx\r\n"
                pos += sprintf(&row_buffer[pos], "%.3f\r\n", value);
            }
        }

        // Write row to file at calculated offset
        MPI_File_write_at(fh, row_offset, row_buffer, pos,
                         MPI_CHAR, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&fh);
    return 0;
}


// CAN BE REMOVED ??
int mpi_compare_matrices(float **matrix1, float **matrix2, int rows, int cols,
                         float tolerance, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int local_match = 1;

    // Each process checks the matrices (they should be identical on all processes)
    for (int i = 0; i < rows && local_match; i++) {
        for (int j = 0; j < cols && local_match; j++) {
            if (fabs(matrix1[i][j] - matrix2[i][j]) > tolerance) {
                if (rank == 0) {
                    printf("Mismatch at [%d][%d]: %.6f vs %.6f (diff: %.6f)\n",
                           i, j, matrix1[i][j], matrix2[i][j],
                           fabs(matrix1[i][j] - matrix2[i][j]));
                }
                local_match = 0;
            }
        }
    }

    // Collect results from all processes
    int global_match;
    MPI_Allreduce(&local_match, &global_match, 1, MPI_INT, MPI_LAND, comm);

    return global_match;
}


// ===================================================================
// SERIAL Matrix Operation - Used for kernel or debugging
// ===================================================================
// Serial versions for testing without MPI
int read_matrix_from_file(const char *filename, float ***matrix, int *rows, int *cols) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return -1;
    }

    // Read dimensions
    if (fscanf(file, "%d %d", rows, cols) != 2) {
        fprintf(stderr, "Error: Invalid file format in %s\n", filename);
        fclose(file);
        return -1;
    }

    // Allocate the matrix
    *matrix = allocate_matrix(*rows, *cols);
    if (*matrix == NULL) {
        fprintf(stderr, "Error: Cannot allocate matrix\n");
        fclose(file);
        return -1;
    }

    // Read matrix data
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            if (fscanf(file, "%f", &((*matrix)[i][j])) != 1) {
                fprintf(stderr, "Error: Cannot read matrix element [%d][%d]\n", i, j);
                free_matrix(*matrix, *rows);
                fclose(file);
                return -1;
            }
        }
    }

    fclose(file);
    return 0;
}

int write_matrix_to_file(const char *filename, float **matrix, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return -1;
    }

    // Write dimensions and data
    fprintf(file, "%d %d\n", rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.3f", matrix[i][j]);
            if (j < cols - 1) fprintf(file, " ");
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    return 0;
}

void print_matrix(float **matrix, int rows, int cols) {
    printf("Matrix (%dx%d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.3f", matrix[i][j]);
            if (j < cols - 1) printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}

float **generate_random_matrix(int rows, int cols, float min_val, float max_val) {
    float **matrix = allocate_matrix(rows, cols);
    if (matrix == NULL) {
        return NULL;
    }

    // Pre-calculate range for efficiency
    const float range = max_val - min_val;
    const float inv_max = 1.0f / (float)UINT_MAX;
    const float scale = range * inv_max;

    // Use OpenMP for parallel generation
    // TEMPORARILY DISABLED FOR DEBUG MODE STABILITY
    // #pragma omp parallel
    {
        // Each thread gets its own random state
        unsigned int seed = (unsigned int)(time(NULL) /* + omp_get_thread_num() * 12345 */);

        // #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Generate random float between min_val and max_val using fast xorshift
                unsigned int rand_int = xorshift32(&seed);
                matrix[i][j] = ((float)rand_int * scale) + min_val;
            }
        }
    }

    return matrix;
}

int compare_matrices(float **matrix1, float **matrix2, int rows, int cols, float tolerance) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fabs(matrix1[i][j] - matrix2[i][j]) > tolerance) {
                printf("Mismatch at [%d][%d]: %.6f vs %.6f (diff: %.6f)\n",
                       i, j, matrix1[i][j], matrix2[i][j],
                       fabs(matrix1[i][j] - matrix2[i][j]));
                return 0;
            }
        }
    }
    return 1;
}