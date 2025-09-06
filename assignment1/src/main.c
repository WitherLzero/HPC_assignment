#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../include/conv2d.h"

void print_usage(const char *program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("Options:\n");
    printf("  -f FILE     Input feature map file\n");
    printf("  -g FILE     Input kernel file\n");
    printf("  -o FILE     Output file (optional)\n");
    printf("  -H HEIGHT   Height of generated matrix (default: 1000)\n");
    printf("  -W WIDTH    Width of generated matrix (default: 1000)\n");
    printf("  -kH HEIGHT  Height of generated kernel (default: 3)\n");
    printf("  -kW WIDTH   Width of generated kernel (default: 3)\n");
    printf("  -p  PRECI   Enable verify mode, won't output to file\n");
    printf("              with precision of floating point (1 ==> 0.1)\n");
    printf("  -s          Use serial implementation (default: parallel)\n");
    printf("  -t          Time the execution in milliseconds\n");
    printf("  -T          Time the execution in seconds\n");
    printf("  -v          Verbose output\n");
    printf("  -h          Show this help message\n");
    printf("\nExamples:\n");
    printf("  Generate tests:\n");
    printf("  %s -H 1000 -W 1000 -kH 3 -kW 3 -G -o o.txt\n", program_name);
    printf("  Verify with example files and precision 2:\n");
    printf("  %s -f f.txt -g g.txt -o o.txt -p 2\n", program_name);
}

int main(int argc, char *argv[]) {
    // Default values
    char *input_file = NULL;
    char *kernel_file = NULL;
    char *output_file = NULL;
    int height = -1;
    int width = -1;
    int kernel_height = -1;
    int kernel_width = -1;
    int generate = 0;
    // int verify_mode = 0;
    int use_serial = 0;
    int time_execution = 0;
    int time_execution_seconds = 0;
    int verbose = 0;
    int precision = -1;

    // Parse command line arguments using getopt_long_only to support -kH/-kW
    // as single-dash long options.
    opterr = 0;  // suppress automatic error messages
    enum { OPT_KH = 1000, OPT_KW };
    static struct option long_options[] = {{"kH", required_argument, 0, OPT_KH},
                                           {"kW", required_argument, 0, OPT_KW},
                                           {0, 0, 0, 0}};

    int long_index = 0;
    int opt;
    while ((opt = getopt_long_only(argc, argv, "f:g:o:H:W:p:stTvh",
                                   long_options, &long_index)) != -1) {
        switch (opt) {
            case 'f':
                input_file = optarg;
                break;
            case 'g':
                kernel_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'H':
                height = atoi(optarg);
                if (height <= 0) {
                    perror("Error: Height must be positive");
                    exit(EXIT_FAILURE);
                }
                break;
            case 'W':
                width = atoi(optarg);
                if (width <= 0) {
                    perror("Error: Width must be positive");
                    exit(EXIT_FAILURE);
                }
                break;
            case 's':
                use_serial = 1;
                break;
            case 't':
                time_execution = 1;
                break;
            case 'T':
                time_execution_seconds = 1;
                break;
            case 'v':
                verbose = 1;
                break;
            case 'p':
                precision = atoi(optarg);
                if (precision <= 0 && precision != -1) {
                    perror("Error: Precision should larger than 0");
                    exit(EXIT_FAILURE);
                }
                break;
            case 'h':
                print_usage(argv[0]);
                exit(EXIT_SUCCESS);
            case OPT_KH:
                kernel_height = atoi(optarg);
                if (kernel_height <= 0) {
                    perror("Error: Kernel height must be positive");
                    exit(EXIT_FAILURE);
                }
                break;
            case OPT_KW:
                kernel_width = atoi(optarg);
                if (kernel_width <= 0) {
                    perror("Error: Kernel width must be positive");
                    exit(EXIT_FAILURE);
                }
                break;
            default:
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    float **input = NULL;
    float **kernel = NULL;

    if (height != -1 && width != -1 && kernel_width != -1 &&
                     kernel_height != -1) {
        generate = 1;
        // Generate random matrices
        if (verbose) {
            printf("Generating random matrices...\n");
        }

        input = generate_random_matrix(height, width, 0.0f, 1.0f);
        kernel =
            generate_random_matrix(kernel_height, kernel_width, 0.0f, 1.0f);

        if (verbose) {
            printf("Generated feature map: %dx%d\n", height, width);
            printf("Generated kernel: %dx%d\n", kernel_height, kernel_width);
        }
    } else {
        if (!input_file || !kernel_file) {
            perror("Wrong param");
            exit(EXIT_FAILURE);
        }
        if (verbose) {
            printf("Loading input matrices from files...\n");
        }

        if (read_matrix_from_file(kernel_file, &kernel, &kernel_height,
                                  &kernel_width) == -1) {
            perror("Error read kernel file");
            exit(EXIT_FAILURE);
        }
        if (read_matrix_from_file(input_file, &input, &height, &width) == -1) {
            perror("Error read inputfile");
            goto checkpoint1;
        }

        if (verbose) {
            printf("Loaded feature map: %dx%d\n", height, width);
            printf("Loaded kernel: %dx%d\n", kernel_height, kernel_width);
        }
    }

    // Validate dimensions
    if (kernel_height > height || kernel_width > width) {
        perror("Error: Kernel size cannot be larger than input size");
        goto checkpoint2;
    }

    // Generate Padded matrix
    float **padded;
    int padded_height, padded_width;
    generate_padded_matrix(input, height, width, kernel_height, kernel_width,
                           &padded, &padded_height, &padded_width);

    // Allocate output matrix
    float **output = allocate_matrix(height, width);

    // Perform convolution
    double start_time, end_time;

    if (time_execution || time_execution_seconds) {
        start_time = omp_get_wtime();
    }

    if (use_serial) {
        if (verbose) {
            printf("Running serial convolution...\n");
        }
        conv2d_serial(padded, padded_height, padded_width, kernel,
                      kernel_height, kernel_width, output);
    } else {
        if (verbose) {
            printf("Running parallel convolution...\n");
        }
        conv2d_parallel(padded, padded_height, padded_width, kernel,
                        kernel_height, kernel_width, output);
    }

    end_time = omp_get_wtime();
    if (time_execution) {
        printf("Execution time: %.3f ms\n", (end_time - start_time) * 1000);
    }

    if (time_execution_seconds) {
        printf("%d\n", (int)(end_time - start_time));
    }

    if (generate && input_file && kernel_file) {
        if (verbose) {
            printf("Writing input to %s...\nWriting kernel to %s\n", input_file,
                   kernel_file);
        }

        if (write_matrix_to_file(input_file, input, height, width) == -1) {
            goto failure;
        }
        if (write_matrix_to_file(kernel_file, kernel, kernel_height,
                                 kernel_width) == -1) {
            goto failure;
        }
    }

    // Output results
    if (output_file) {
        if (precision == -1) {
            if (verbose) {
                printf("Writing output to %s...\n", output_file);
            }
            if (write_matrix_to_file(output_file, output, height, width) ==
                -1) {
                goto failure;
            }
        } else {
            // compare with given matrix
            float **v_output;
            int v_height, v_width;
            if (read_matrix_from_file(output_file, &v_output, &v_height,
                                      &v_width) == -1) {
                goto failure;
            }
            if (compare_matrices(v_output, output, height, width,
                                 1.0f / powf(10, precision)) == 1) {
                puts("Verify Pass!");
            } else {
                puts("Verify Failed!");
            }
        }
    } else if (verbose) {
        print_matrix(output, height, width);
    }

    // Clean up
    free_matrix(input, height);
    free_matrix(kernel, kernel_height);
    free_matrix(padded, padded_height);
    free_matrix(output, height);

    if (verbose) {
        printf("Done.\n");
    }

    return EXIT_SUCCESS;

failure:
    // Clean up
    free_matrix(output, height);
    free_matrix(padded, padded_height);
checkpoint2:
    free_matrix(input, height);
checkpoint1:
    free_matrix(kernel, kernel_height);

    return EXIT_FAILURE;
}
