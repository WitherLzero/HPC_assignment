#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

// Generate random matrix with values between min_val and max_val
float **generate_random_matrix(int rows, int cols, float min_val,
                               float max_val) {
    float **matrix = allocate_matrix(rows, cols);

    srand(time(NULL));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Generate random float between min_val and max_val
            float random_val =
                ((float)rand() / RAND_MAX) * (max_val - min_val) + min_val;
            matrix[i][j] = random_val;
        }
    }

    return matrix;
}
