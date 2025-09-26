#include "conv2d_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <math.h>

// Basic matrix I/O functions (reused from assignment1 pattern)
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

    // Broadcast the generated matrix to all processes
    for (int i = 0; i < rows; i++) {
        MPI_Bcast(matrix[i], cols, MPI_FLOAT, 0, comm);
    }

    return matrix;
}

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