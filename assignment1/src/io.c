#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <omp.h>

#include "../include/conv2d.h"

// Read matrix from file in the specified format
int read_matrix_from_file(const char *filename, float ***matrix, int *rows,
                          int *cols) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error: Cannot open file");
        return -1;
    }

    // Read dimensions
    if (fscanf(file, "%d %d", rows, cols) != 2) {
        perror("Error: Cannot read matrix dimensions");
        fclose(file);
        return -1;
    }

    // Allocate matrix
    *matrix = allocate_matrix(*rows, *cols);

    // Read matrix data
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            if (fscanf(file, "%f", &(*matrix)[i][j]) != 1) {
                perror("Error: Cannot read matrix element");
                free_matrix(*matrix, *rows);
                fclose(file);
                return -1;
            }
        }
    }

    fclose(file);
    return 0;
}

// Write matrix to file in the specified format
int write_matrix_to_file(const char *filename, float **matrix, int rows,
                         int cols) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error: Cannot create file");
        return -1;
    }

    // Write dimensions
    fprintf(file, "%d %d\n", rows, cols);

    // Write matrix data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.3f", matrix[i][j]);
            if (j < cols - 1) {
                fprintf(file, " ");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);

    return 0;
}

// Print matrix to stdout
void print_matrix(float **matrix, int rows, int cols) {
    printf("%d %d\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.3f", matrix[i][j]);
            if (j < cols - 1) {
                printf(" ");
            }
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


// Generate random matrix with values between min_val and max_val
float **generate_random_matrix(int rows, int cols, float min_val,
                               float max_val) {
    float **matrix = allocate_matrix(rows, cols);
    
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

// Read matrix from file directly into padded format
int read_matrix_into_padded(const char *filename, int kernel_height, int kernel_width,
                           float ***padded, int *padded_height, int *padded_width,
                           int *original_height, int *original_width) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error: Cannot open file");
        return -1;
    }

    // Read dimensions
    if (fscanf(file, "%d %d", original_height, original_width) != 2) {
        perror("Error: Cannot read matrix dimensions");
        fclose(file);
        return -1;
    }

    // Calculate padding
    int pad_top = (kernel_height - 1) / 2;
    int pad_left = (kernel_width - 1) / 2;
    int pad_bottom = kernel_height - 1 - pad_top;
    int pad_right = kernel_width - 1 - pad_left;

    *padded_height = *original_height + pad_top + pad_bottom;
    *padded_width = *original_width + pad_left + pad_right;
    
    // Allocate padded matrix
    *padded = allocate_matrix(*padded_height, *padded_width);
    initialize_matrix(*padded, *padded_height, *padded_width, 0.0f);

    // Read matrix data directly into the center of the padded matrix
    for (int i = 0; i < *original_height; i++) {
        for (int j = 0; j < *original_width; j++) {
            if (fscanf(file, "%f", &(*padded)[i + pad_top][j + pad_left]) != 1) {
                perror("Error: Cannot read matrix element");
                free_matrix(*padded, *padded_height);
                fclose(file);
                return -1;
            }
        }
    }

    fclose(file);
    return 0;
}

// Generate random matrix directly into padded format
float **generate_random_matrix_into_padded(int height, int width, int kernel_height, int kernel_width,
                                          float min_val, float max_val, float ***padded,
                                          int *padded_height, int *padded_width) {
    // Calculate padding
    int pad_top = (kernel_height - 1) / 2;
    int pad_left = (kernel_width - 1) / 2;
    int pad_bottom = kernel_height - 1 - pad_top;
    int pad_right = kernel_width - 1 - pad_left;

    *padded_height = height + pad_top + pad_bottom;
    *padded_width = width + pad_left + pad_right;
    
    // Allocate padded matrix
    *padded = allocate_matrix(*padded_height, *padded_width);
    initialize_matrix(*padded, *padded_height, *padded_width, 0.0f);

    // Pre-calculate range for efficiency
    const float range = max_val - min_val;
    const float inv_max = 1.0f / (float)UINT_MAX;
    const float scale = range * inv_max;

    // Use OpenMP for parallel generation directly into padded matrix
    #pragma omp parallel
    {
        // Each thread gets its own random state
        unsigned int seed = (unsigned int)(time(NULL) + omp_get_thread_num() * 12345);
        
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                // Generate random float between min_val and max_val using fast xorshift
                unsigned int rand_int = xorshift32(&seed);
                (*padded)[i + pad_top][j + pad_left] = ((float)rand_int * scale) + min_val;
            }
        }
    }

    // Return a pointer to the original data area (for compatibility with existing code)
    // This allows the caller to access the original matrix if needed
    return (*padded) + pad_top;
}

// Write matrix header (dimensions) to file
int write_matrix_header(const char *filename, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error: Cannot create file");
        return -1;
    }
    
    fprintf(file, "%d %d\n", rows, cols);
    fclose(file);
    return 0;
}

// Write matrix data in batches to file (appends to existing file)
int write_matrix_data_batch(const char *filename, int start_row, int end_row, int cols, 
                           float min_val, float max_val) {
    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        perror("Error: Cannot open file for writing");
        return -1;
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
        
        // Allocate buffer for this thread's row
        float *row_buffer = (float *)malloc(cols * sizeof(float));
        
        #pragma omp for schedule(static)
        for (int i = start_row; i < end_row; i++) {
            // Generate random values for this row
            for (int j = 0; j < cols; j++) {
                unsigned int rand_int = xorshift32(&seed);
                row_buffer[j] = ((float)rand_int * scale) + min_val;
            }
            
            // Write the row to file (single write operation per row)
            #pragma omp critical
            {
                for (int j = 0; j < cols; j++) {
                    fprintf(file, "%.3f", row_buffer[j]);
                    if (j < cols - 1) {
                        fprintf(file, " ");
                    }
                }
                fprintf(file, "\n");
            }
        }
        
        free(row_buffer);
    }
    
    fclose(file);
    return 0;
}

int generate_feature_map(const char* filename, int height, int width) {
    // Default batch size - can be adjusted based on available memory
    const int BATCH_SIZE = 1000;  // Process 1000 rows at a time
    
    // Write matrix header first
    if (write_matrix_header(filename, height, width) == -1) {
        return -1;
    }
    
    // Process matrix in batches
    for (int start_row = 0; start_row < height; start_row += BATCH_SIZE) {
        int end_row = start_row + BATCH_SIZE;
        if (end_row > height) {
            end_row = height;
        }
        
        // Write this batch to file
        if (write_matrix_data_batch(filename, start_row, end_row, width, 0.0f, 1.0f) == -1) {
            return -1;
        }
    }
    
    return 0;
}
