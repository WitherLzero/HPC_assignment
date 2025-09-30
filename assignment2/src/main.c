#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

#include "../include/conv2d_mpi.h"

void print_usage(const char *program_name) {
    printf("Usage: mpirun -np <processes> %s [OPTIONS]\n", program_name);
    printf("Options:\n");
    printf("  -f FILE     Input feature map file\n");
    printf("  -g FILE     Input kernel file\n");
    printf("  -o FILE     Output file (optional)\n");
    printf("  -H HEIGHT   Height of generated matrix\n");
    printf("  -W WIDTH    Width of generated matrix\n");
    printf("  -kH HEIGHT  Height of generated kernel\n");
    printf("  -kW WIDTH   Width of generated kernel\n");
    printf("  -sH STRIDE  Vertical stride (default: 1)\n");
    printf("  -sW STRIDE  Horizontal stride (default: 1)\n");
    printf("  -p PRECI    Enable verify mode with precision\n");
    printf("              (1 => 0.1, 2 => 0.01, etc.)\n");
    printf("  -s          Use serial implementation only\n");
    printf("  -m          Use MPI-only implementation\n");
    printf("  -P          Use OpenMP-only implementation\n");
    printf("  -t          Time the execution in milliseconds\n");
    printf("  -T          Time the execution in seconds\n");
    printf("  -v          Verbose output\n");
    printf("  -h          Show this help message\n");
    printf("\nExamples:\n");
    printf("  Generate and test with stride:\n");
    printf("  mpirun -np 2 %s -H 100 -W 100 -kH 3 -kW 3 -sH 2 -sW 2 -t\n", program_name);
    printf("  Use existing files with stride:\n");
    printf("  mpirun -np 1 %s -f f.txt -g g.txt -o output.txt -sH 3 -sW 2 -v\n", program_name);
    printf("  Verify against expected output:\n");
    printf("  mpirun -np 1 %s -f f.txt -g g.txt -o expected.txt -p 2\n", program_name);
}

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default values
    char *input_file = NULL;
    char *kernel_file = NULL;
    char *output_file = NULL;
    int height = -1;
    int width = -1;
    int kernel_height = -1;
    int kernel_width = -1;
    int stride_H = 1;  // Default stride
    int stride_W = 1;  // Default stride
    int use_serial = 0;
    int use_mpi_only = 0;
    int use_openmp_only = 0;
    int time_execution = 0;
    int time_execution_seconds = 0;
    int verbose = 0;
    int precision = -1;

    // Parse command line arguments using getopt_long_only
    opterr = 0;  // suppress automatic error messages
    enum { OPT_KH = 1000, OPT_KW, OPT_SH, OPT_SW };
    static struct option long_options[] = {
        {"kH", required_argument, 0, OPT_KH},
        {"kW", required_argument, 0, OPT_KW},
        {"sH", required_argument, 0, OPT_SH},
        {"sW", required_argument, 0, OPT_SW},
        {0, 0, 0, 0}
    };

    int long_index = 0;
    int opt;
    while ((opt = getopt_long_only(argc, argv, "f:g:o:H:W:p:smPtTvh",
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
                    if (rank == 0) fprintf(stderr, "Error: Height must be positive\n");
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                break;
            case 'W':
                width = atoi(optarg);
                if (width <= 0) {
                    if (rank == 0) fprintf(stderr, "Error: Width must be positive\n");
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                break;
            case OPT_KH:
                kernel_height = atoi(optarg);
                if (kernel_height <= 0) {
                    if (rank == 0) fprintf(stderr, "Error: Kernel height must be positive\n");
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                break;
            case OPT_KW:
                kernel_width = atoi(optarg);
                if (kernel_width <= 0) {
                    if (rank == 0) fprintf(stderr, "Error: Kernel width must be positive\n");
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                break;
            case OPT_SH:
                stride_H = atoi(optarg);
                if (stride_H <= 0) {
                    if (rank == 0) fprintf(stderr, "Error: Vertical stride must be positive\n");
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                break;
            case OPT_SW:
                stride_W = atoi(optarg);
                if (stride_W <= 0) {
                    if (rank == 0) fprintf(stderr, "Error: Horizontal stride must be positive\n");
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                break;
            case 's':
                use_serial = 1;
                break;
            case 'm':
                use_mpi_only = 1;
                break;
            case 'P':
                use_openmp_only = 1;
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
                if (precision <= 0) {
                    if (rank == 0) fprintf(stderr, "Error: Precision must be positive\n");
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                break;
            case 'h':
                if (rank == 0) print_usage(argv[0]);
                MPI_Finalize();
                exit(EXIT_SUCCESS);
            case '?':
                if (rank == 0) {
                    fprintf(stderr, "Unknown option: %c\n", optopt);
                    print_usage(argv[0]);
                }
                MPI_Finalize();
                exit(EXIT_FAILURE);
            default:
                if (rank == 0) {
                    fprintf(stderr, "Error parsing arguments\n");
                    print_usage(argv[0]);
                }
                MPI_Finalize();
                exit(EXIT_FAILURE);
        }
    }

    // Validate input combinations following assignment1 logic
    int has_generation = (height > 0) && (width > 0) && (kernel_height > 0) && (kernel_width > 0);
    int has_input_files = (input_file != NULL) && (kernel_file != NULL);

    if (!has_generation && !has_input_files) {
        if (rank == 0) {
            fprintf(stderr, "Error: Must specify either generation parameters (-H, -W, -kH, -kW) or input files (-f, -g)\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Variables for matrices
    float **input = NULL;
    float **kernel = NULL;
    float **padded_input = NULL;
    float **output = NULL;
    int input_H, input_W;
    int kernel_H, kernel_W;
    int padded_H, padded_W;
    int output_H, output_W;

    // Following assignment1 logic: generation takes precedence
    if (has_generation) {
        // Generate random matrices first
        input_H = height;
        input_W = width;
        kernel_H = kernel_height;
        kernel_W = kernel_width;

        input = mpi_generate_random_matrix(input_H, input_W, 0.0f, 1.0f, MPI_COMM_WORLD);
        kernel = mpi_generate_random_matrix(kernel_H, kernel_W, 0.0f, 1.0f, MPI_COMM_WORLD);

        if (input == NULL || kernel == NULL) {
            if (rank == 0) fprintf(stderr, "Error generating matrices\n");
            if (input) free_matrix(input, input_H);
            if (kernel) free_matrix(kernel, kernel_H);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // Save generated matrices if file paths specified
        if (input_file) {
            mpi_write_matrix_to_file(input_file, input, input_H, input_W, MPI_COMM_WORLD);
            if (rank == 0 && verbose) printf("Generated input saved to %s\n", input_file);
        }
        if (kernel_file) {
            mpi_write_matrix_to_file(kernel_file, kernel, kernel_H, kernel_W, MPI_COMM_WORLD);
            if (rank == 0 && verbose) printf("Generated kernel saved to %s\n", kernel_file);
        }
    } else if (has_input_files) {
        // Only read from files if no generation parameters provided
        if (mpi_read_matrix_from_file(input_file, &input, &input_H, &input_W, MPI_COMM_WORLD) != 0) {
            if (rank == 0) fprintf(stderr, "Error reading input file %s\n", input_file);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        if (mpi_read_matrix_from_file(kernel_file, &kernel, &kernel_H, &kernel_W, MPI_COMM_WORLD) != 0) {
            if (rank == 0) fprintf(stderr, "Error reading kernel file %s\n", kernel_file);
            free_matrix(input, input_H);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        if (rank == 0 && verbose) {
            printf("Loaded input from %s\n", input_file);
            printf("Loaded kernel from %s\n", kernel_file);
        }
    }

    if (rank == 0 && verbose) {
        printf("Input dimensions: %d x %d\n", input_H, input_W);
        printf("Kernel dimensions: %d x %d\n", kernel_H, kernel_W);
        printf("Stride: %d x %d\n", stride_H, stride_W);
    }

    // Generate padded input for "same" padding
    generate_padded_matrix(input, input_H, input_W, kernel_H, kernel_W,
                          &padded_input, &padded_H, &padded_W);

    // Calculate output dimensions
    calculate_stride_output_dims(input_H, input_W, stride_H, stride_W, &output_H, &output_W);

    // Determine optimal number of processes (limit to output rows for MPI distribution)
    int optimal_processes = (size > output_H) ? output_H : size;
    bool is_active_process = (rank < optimal_processes);

    // Create a sub-communicator for active processes only
    MPI_Comm active_comm;
    MPI_Comm_split(MPI_COMM_WORLD, is_active_process ? 0 : MPI_UNDEFINED, rank, &active_comm);

    if (!is_active_process) {
        // Inactive processes: allocate output but skip computation
        output = allocate_matrix(output_H, output_W);
        // Initialize output with zeros
        for (int i = 0; i < output_H; i++) {
            for (int j = 0; j < output_W; j++) {
                output[i][j] = 0.0f;
            }
        }
        // Skip to cleanup section
        goto cleanup;
    }

    output = allocate_matrix(output_H, output_W);

    if (rank == 0 && verbose) {
        printf("Padded input dimensions: %d x %d\n", padded_H, padded_W);
        printf("Output dimensions: %d x %d\n", output_H, output_W);
        if (optimal_processes < size) {
            printf("Using %d active processes (out of %d total) for optimal distribution.\n",
                   optimal_processes, size);
        }
    }

    // Perform convolution with timing
    mpi_timer_t timer;

    if (time_execution || time_execution_seconds) {
        mpi_timer_start(&timer);
    }

    // Choose implementation based on flags
    if (use_serial) {
        if (rank == 0) {
            conv2d_stride_serial(padded_input, padded_H, padded_W, kernel, kernel_H, kernel_W,
                               stride_H, stride_W, output);
        }
    } else if (use_mpi_only) {
        conv2d_stride_mpi(padded_input, padded_H, padded_W, kernel, kernel_H, kernel_W,
                         stride_H, stride_W, output, active_comm);
    } else if (use_openmp_only) {
        if (rank == 0) {
            conv2d_stride_openmp(padded_input, padded_H, padded_W, kernel, kernel_H, kernel_W,
                               stride_H, stride_W, output);
        }
    } else {
        // Default: hybrid implementation
        conv2d_stride_hybrid(padded_input, padded_H, padded_W, kernel, kernel_H, kernel_W,
                           stride_H, stride_W, output, active_comm);
    }

    if (time_execution || time_execution_seconds) {
        mpi_timer_end(&timer, active_comm);
        if (rank == 0) {
            if (time_execution_seconds) {
                printf("Timing - Convolution with stride: %.6f seconds\n", timer.elapsed_time);
            } else {
                printf("Timing - Convolution with stride: %.3f milliseconds\n", timer.elapsed_time * 1000.0);
            }
            fflush(stdout);
        }
    }

    // Handle output
    if (precision > 0 && output_file) {
        // Verify mode: compare with expected output
        float **expected = NULL;
        int expected_H, expected_W;

        if (rank == 0) {
            if (read_matrix_from_file(output_file, &expected, &expected_H, &expected_W) == 0) {
                if (expected_H == output_H && expected_W == output_W) {
                    float tolerance = pow(10.0f, -precision);
                    if (compare_matrices(output, expected, output_H, output_W, tolerance)) {
                        printf("Verify Pass!\n");
                    } else {
                        printf("Verify Failed!\n");
                    }
                } else {
                    printf("Verify Failed! Dimension mismatch: expected %dx%d, got %dx%d\n",
                           expected_H, expected_W, output_H, output_W);
                }
                free_matrix(expected, expected_H);
            } else {
                printf("Error reading expected output file for verification\n");
            }
        }
    } else if (output_file) {
        // Write output to file
        mpi_write_matrix_to_file(output_file, output, output_H, output_W, MPI_COMM_WORLD);
        if (rank == 0 && verbose) {
            printf("Output written to %s\n", output_file);
        }
    } else if (rank == 0 && verbose) {
        // Print small outputs
        if (output_H <= 10 && output_W <= 10) {
            print_matrix(output, output_H, output_W);
        } else {
            printf("Output too large to display (%d x %d)\n", output_H, output_W);
        }
    }

cleanup:
    // Cleanup
    free_matrix(input, input_H);
    free_matrix(kernel, kernel_H);
    free_matrix(padded_input, padded_H);
    free_matrix(output, output_H);

    MPI_Finalize();
    return EXIT_SUCCESS;
}