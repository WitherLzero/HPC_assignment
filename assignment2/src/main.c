/**
 * @file _main.c
 * @brief MPI + OpenMP Hybrid 2D Convolution with Stride Support
 *
 * CITS5507 Assignment 2 - Main Program
 *
 * Architecture:
 * - Hybrid parallelization: MPI (distributed memory) + OpenMP (shared memory)
 * - Dual-path generation: Parallel (stride=1) vs Centralized (stride>1)
 * - MPI parallel I/O for efficient file operations
 * - Stride-aware distribution and computation
 */

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
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
    // ===================================================================
    // PHASE 1: INITIALIZATION
    // ===================================================================

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default values
    char *input_file = NULL;
    char *kernel_file = NULL;
    char *output_file = NULL;
    int input_H = -1;
    int input_W = -1;
    int kernel_H = -1;
    int kernel_W = -1;
    int stride_H = 1;
    int stride_W = 1;
    int use_serial = 0;
    int use_mpi_only = 0;
    int use_openmp_only = 0;
    int time_execution = 0;
    int time_execution_seconds = 0;
    int verbose = 0;
    int precision = -1;

    // ===================================================================
    // Parse command line arguments (all processes)
    // ===================================================================

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
                input_H = atoi(optarg);
                if (input_H <= 0) {
                    if (rank == 0) fprintf(stderr, "Error: Height must be positive\n");
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                break;
            case 'W':
                input_W = atoi(optarg);
                if (input_W <= 0) {
                    if (rank == 0) fprintf(stderr, "Error: Width must be positive\n");
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                break;
            case OPT_KH:
                kernel_H = atoi(optarg);
                if (kernel_H <= 0) {
                    if (rank == 0) fprintf(stderr, "Error: Kernel height must be positive\n");
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                break;
            case OPT_KW:
                kernel_W = atoi(optarg);
                if (kernel_W <= 0) {
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

    // Validate input combinations
    int has_generation = (input_H > 0) && (input_W > 0) && (kernel_H > 0) && (kernel_W > 0);
    int has_input_files = (input_file != NULL) && (kernel_file != NULL);

    if (!has_generation && !has_input_files) {
        if (rank == 0) {
            fprintf(stderr, "Error: Must specify either generation parameters (-H, -W, -kH, -kW) or input files (-f, -g)\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Validate implementation mode vs process count
    // Serial and OpenMP-only require exactly 1 MPI process
    if ((use_serial || use_openmp_only) && size > 1) {
        if (rank == 0) {
            fprintf(stderr, "Error: Serial (-s) and OpenMP-only (-P) modes require exactly 1 MPI process\n");
            fprintf(stderr, "       You launched with %d processes. Use: mpirun -np 1 ...\n", size);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Warn if conflicting implementation flags are set
    int implementation_count = use_serial + use_mpi_only + use_openmp_only;
    if (implementation_count > 1) {
        if (rank == 0) {
            fprintf(stderr, "Warning: Multiple implementation flags set (-s, -m, -P). Using priority: serial > openmp > mpi\n");
        }
        // Resolve conflicts with priority
        if (use_serial) {
            use_mpi_only = 0;
            use_openmp_only = 0;
        } else if (use_openmp_only) {
            use_mpi_only = 0;
        }
    }

    // Read file headers to get dimensions (needed for active_comm creation)
    if (!has_generation && has_input_files) {
        if (rank == 0) {
            // Read header only to get dimensions
            FILE *f = fopen(input_file, "r");
            if (f) {
                fscanf(f, "%d %d", &input_H, &input_W);
                fclose(f);
            }
            f = fopen(kernel_file, "r");
            if (f) {
                fscanf(f, "%d %d", &kernel_H, &kernel_W);
                fclose(f);
            }
        }
        // Broadcast dimensions to all processes
        MPI_Bcast(&input_H, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&input_W, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&kernel_H, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&kernel_W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }


    // ===================================================================
    // PHASE 2: CREATE ACTIVE COMMUNICATOR
    // ===================================================================

    // Calculate output dimensions
    int output_H, output_W;
    calculate_stride_output_dims(input_H, input_W, stride_H, stride_W, &output_H, &output_W);

    // Determine optimal number of processes (limit to output rows)
    int optimal_processes = (size > output_H) ? output_H : size;
    bool is_active_process = (rank < optimal_processes);

    // Create sub-communicator for active processes only
    MPI_Comm active_comm;
    MPI_Comm_split(MPI_COMM_WORLD,
                   is_active_process ? 0 : MPI_UNDEFINED,
                   rank,
                   &active_comm);

    if (rank == 0 && optimal_processes < size) {
        printf("Using %d active processes (out of %d total)\n",
               optimal_processes, size);
    }

    // Inactive processes exit early
    if (!is_active_process) {
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    // Update rank and size for active communicator
    MPI_Comm_rank(active_comm, &rank);
    MPI_Comm_size(active_comm, &size);


    if (rank == 0 && verbose) {
        printf("Input dimensions: %d x %d\n", input_H, input_W);
        printf("Kernel dimensions: %d x %d\n", kernel_H, kernel_W);
        printf("Stride: %d x %d\n", stride_H, stride_W);
        printf("Output dimensions: %d x %d\n", output_H, output_W);
    }

    // ===================================================================
    // PHASE 3: INPUT ACQUISITION
    // ===================================================================

    float **local_padded_input = NULL, **kernel = NULL;
    int padded_local_H, padded_local_W, local_start_row;

    if (has_generation) {
        // ====================================================================
        // DUAL-PATH GENERATE MODE
        // ====================================================================

        // PATH A: stride = 1 -> Parallel generation (no overlaps)
        if (stride_H == 1 && stride_W == 1) {
            if (rank == 0 && verbose) {
                printf("Using PATH A: Parallel generation (stride=1)\n");
            }

            // Each process generates its local portion directly
            local_padded_input = mpi_generate_local_padded_matrix(
                input_H, input_W, kernel_H, kernel_W, stride_H, stride_W,
                &padded_local_H, &padded_local_W, &local_start_row,
                0.0f, 1.0f, active_comm
            );

            if (local_padded_input == NULL) {
                fprintf(stderr, "Rank %d: Failed to generate local padded input\n", rank);
                MPI_Comm_free(&active_comm);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }

            // Save using MPI Parallel I/O
            if (input_file) {
                int write_result = mpi_write_input_parallel(
                    input_file,
                    local_padded_input,
                    padded_local_H, padded_local_W,
                    local_start_row,
                    input_H, input_W,
                    kernel_H, kernel_W,
                    active_comm
                );

                if (write_result == 0 && rank == 0 && verbose) {
                    printf("Generated input saved to %s (parallel write)\n", input_file);
                }
            }
        }
        // PATH B: stride > 1 -> Root centralized generation (avoid overlap inconsistency)
        else {
            if (rank == 0 && verbose) {
                printf("Using PATH B: Root centralized generation (stride>1)\n");
            }

            float **global_padded = NULL;
            int global_padded_H, global_padded_W;

            // Root generates global padded matrix
            if (rank == 0) {
                global_padded = generate_random_matrix_into_padded(
                    input_H, input_W, kernel_H, kernel_W,
                    0.0f, 1.0f,
                    &global_padded_H, &global_padded_W
                );


                if (global_padded == NULL) {
                    fprintf(stderr, "Error: Failed to generate global padded matrix\n");
                    MPI_Comm_free(&active_comm);
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }

            }

            // Broadcast global padded dimensions to all processes
            MPI_Bcast(&global_padded_H, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&global_padded_W, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Distribute global padded matrix to all processes
            mpi_distribute_matrix_stride_aware(
                global_padded, global_padded_H, global_padded_W,
                kernel_H, kernel_W, stride_H, stride_W,
                &local_padded_input,
                &padded_local_H, &padded_local_W,
                &local_start_row,
                active_comm
            );

            if (local_padded_input == NULL) {
                fprintf(stderr, "Rank %d: Failed during distribution\n", rank);
                if (rank == 0 && global_padded) free_matrix(global_padded, global_padded_H);
                MPI_Comm_free(&active_comm);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }

            // Root writes file and frees global matrix
            if (rank == 0 && input_file) {
                int write_result = write_padded_matrix_to_file(
                    global_padded, global_padded_H, global_padded_W,
                    input_H, input_W, kernel_H, kernel_W,
                    input_file
                );

                if (write_result == 0 && verbose) {
                    printf("Generated input saved to %s (root serial write)\n", input_file);
                }

                // Free global matrix immediately
                free_matrix(global_padded, global_padded_H);
                global_padded = NULL;
            }
        }

        // ====================================================================
        // KERNEL GENERATION
        // ====================================================================

        if (rank == 0) {
            kernel = generate_random_matrix(kernel_H, kernel_W, 0.0f, 1.0f);
            if (kernel == NULL) {
                fprintf(stderr, "Error: Failed to generate kernel\n");
                free_matrix(local_padded_input, padded_local_H);
                MPI_Comm_free(&active_comm);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }

            if (kernel_file) {
                write_matrix_to_file(kernel_file, kernel, kernel_H, kernel_W);
                if (verbose) printf("Generated kernel saved to %s\n", kernel_file);
            }
        }

        mpi_broadcast_kernel(&kernel, kernel_H, kernel_W, active_comm);

    } else if (has_input_files){
        // ====================================================================
        // READ MODE: Load from files
        // ====================================================================
        local_padded_input = mpi_read_local_padded_matrix(
            input_file, &input_H, &input_W,
            kernel_H, kernel_W, stride_H, stride_W,
            &padded_local_H, &padded_local_W, &local_start_row,
            active_comm
        );

        if (local_padded_input == NULL) {
            fprintf(stderr, "Rank %d: Failed to read local padded input from file\n", rank);
            MPI_Comm_free(&active_comm);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // Read kernel on root, broadcast to all
        if (rank == 0) {
            read_matrix_from_file(kernel_file, &kernel, &kernel_H, &kernel_W);
            if (kernel == NULL) {
                fprintf(stderr, "Error: Failed to read kernel from file %s\n", kernel_file);
                free_matrix(local_padded_input, padded_local_H);
                MPI_Comm_free(&active_comm);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }
        }

        mpi_broadcast_kernel(&kernel, kernel_H, kernel_W, active_comm);
    }

    // ===================================================================
    // PHASE 4: COMPUTATION
    // ===================================================================
    int local_output_H = (padded_local_H - kernel_H + 1 + stride_H - 1) / stride_H;
    int local_output_W = (padded_local_W - kernel_W + 1 + stride_W - 1) / stride_W;
    float **local_output = allocate_matrix(local_output_H, local_output_W);

    // Start timing
    mpi_timer_t timer;
    if (time_execution || time_execution_seconds) {
        mpi_timer_start(&timer);
    }

    // Select implementation mode
    if (use_serial) {
        conv2d_stride_serial(local_padded_input, padded_local_H, padded_local_W,
                             kernel, kernel_H, kernel_W,
                             stride_H, stride_W, local_output);
    } else if (use_openmp_only) {
        conv2d_stride_openmp(local_padded_input, padded_local_H, padded_local_W,
                             kernel, kernel_H, kernel_W,
                             stride_H, stride_W, local_output);
    } else if (use_mpi_only) {
        conv2d_stride_mpi(local_padded_input, padded_local_H, padded_local_W,
                          kernel, kernel_H, kernel_W,
                          stride_H, stride_W, local_output, active_comm);
    } else {
        conv2d_stride_hybrid(local_padded_input, padded_local_H, padded_local_W,
                             kernel, kernel_H, kernel_W,
                             stride_H, stride_W, local_output, active_comm);
    }

    // End timing
    if (time_execution || time_execution_seconds) {
        mpi_timer_end(&timer, active_comm);
        if (rank == 0) {
            if (time_execution_seconds) {
                printf("Timing - Convolution with stride: %.6f seconds\n", timer.elapsed_time);
            } else {
                printf("Timing - Convolution with stride: %.3f milliseconds\n",
                       timer.elapsed_time * 1000.0);
            }
            fflush(stdout);
        }
    }

    // ===================================================================
    // PHASE 5: OUTPUT HANDLING
    // ===================================================================

    float **full_output = NULL;
    bool need_full_output = (precision > 0) || (!output_file && verbose) || (size == 1);

    if (size == 1) {
        full_output = local_output;
    } else if (need_full_output) {
        // Calculate output start row for gathering
        int output_start_row;
        int output_base_rows = output_H / size;
        int output_remainder = output_H % size;

        if (stride_H > 1 || stride_W > 1) {
            output_start_row = rank * output_base_rows + (rank < output_remainder ? rank : output_remainder);
        } else {
            output_start_row = 0;
            for (int p = 0; p < rank; p++) {
                int p_local_output_H = output_base_rows + (p < output_remainder ? 1 : 0);
                output_start_row += p_local_output_H;
            }
        }

        mpi_gather_output(local_output, local_output_H, local_output_W,
                          output_start_row,
                          &full_output, output_H, output_W,
                          active_comm);
    }

    // Verification mode
    if (precision > 0 && output_file) {
        if (rank == 0) {
            float **expected = NULL;
            int expected_H, expected_W;

            if (read_matrix_from_file(output_file, &expected, &expected_H, &expected_W) == 0) {
                if (expected_H == output_H && expected_W == output_W) {
                    float tolerance = pow(10.0f, -precision);
                    if (full_output != NULL && compare_matrices(full_output, expected, output_H, output_W, tolerance)) {
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
        int output_start_row;
        int output_base_rows = output_H / size;
        int output_remainder = output_H % size;

        if (stride_H > 1 || stride_W > 1) {
            output_start_row = rank * output_base_rows + (rank < output_remainder ? rank : output_remainder);
        } else {
            output_start_row = 0;
            for (int p = 0; p < rank; p++) {
                int p_local_output_H = output_base_rows + (p < output_remainder ? 1 : 0);
                output_start_row += p_local_output_H;
            }
        }

        // Parallel I/O write
        int write_result = mpi_write_output_parallel(
            output_file, local_output,
            local_output_H, local_output_W,
            output_start_row,
            output_H, output_W,
            kernel_H, kernel_W,
            active_comm
        );

        if (write_result != 0 && rank == 0) {
            fprintf(stderr, "Error: Failed to write output file\n");
        }
    } else if (rank == 0 && verbose && full_output != NULL) {
        if (output_H <= 10 && output_W <= 10) {
            print_matrix(full_output, output_H, output_W);
        } else {
            printf("Result computed (%dx%d)\n", output_H, output_W);
        }
    }

    // Cleanup
    if (full_output != NULL && full_output != local_output && rank == 0) {
        free_matrix(full_output, output_H);
    }
    if (local_padded_input) free_matrix(local_padded_input, padded_local_H);
    if (local_output) free_matrix(local_output, local_output_H);
    if (kernel) free_matrix(kernel, kernel_H);

    MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
