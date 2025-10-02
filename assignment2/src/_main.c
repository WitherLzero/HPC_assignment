/**
 * @file _main.c
 * @brief Memory-Optimized MPI Parallel I/O Implementation (Refactored)
 *
 * This is the refactored main file following the Memory-Optimized MPI I/O Plan.
 * Key changes from original main.c:
 * - Active communicator created EARLY (before matrix operations)
 * - Direct local padded matrix generation/reading (no global matrices on root)
 * - MPI Parallel I/O for reading and writing
 * - Conditional gathering only when no output file specified
 * - All functions use active_comm parameter
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

    // If reading from file, we need to get dimensions first (root reads header)
    if (!has_generation && has_input_files) {
        // TODO: Read dimensions from file header (will implement in Phase 4)
        // For now, placeholder - will be replaced with parallel I/O header reading
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
    // Calculate output dimensions (BEFORE creating active_comm)
    // ===================================================================

    int output_H, output_W;
    calculate_stride_output_dims(input_H, input_W, stride_H, stride_W, &output_H, &output_W);

    // ===================================================================
    // CREATE ACTIVE_COMM (EARLY - Phase 1 Key Change!)
    // ===================================================================

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

    // ===================================================================
    // Inactive processes exit early
    // ===================================================================

    if (!is_active_process) {
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    // ===================================================================
    // FROM HERE: Only active processes, all operations use active_comm
    // ===================================================================

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
    // PHASE 2: INPUT ACQUISITION (PARALLEL)
    // TODO: Implement in later phases
    // ===================================================================

    float **local_padded_input = NULL, **kernel = NULL;
    int padded_local_H, padded_local_W, local_start_row;

    if (has_generation) {
        // Phase 3: Direct local padded matrix generation
        local_padded_input = mpi_generate_local_padded_matrix(
            input_H, input_W, kernel_H, kernel_W,
            &padded_local_H, &padded_local_W, &local_start_row,
            0.0f, 1.0f, active_comm
        );

        if (local_padded_input == NULL) {
            fprintf(stderr, "Rank %d: Failed to generate local padded input\n", rank);
            MPI_Comm_free(&active_comm);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // Kernel handling (generate on root, broadcast to all)
        if (rank == 0) {
            // Generate kernel using old function for now
            kernel = generate_random_matrix(kernel_H, kernel_W, 0.0f, 1.0f);
            if (kernel == NULL) {
                fprintf(stderr, "Error: Failed to generate kernel\n");
                free_matrix(local_padded_input, padded_local_H);
                MPI_Comm_free(&active_comm);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }
        }

        // Broadcast kernel to all active processes
        mpi_broadcast_kernel(&kernel, kernel_H, kernel_W, active_comm);

        if (rank == 0 && verbose) {
            printf("Generated local padded input and kernel successfully\n");
        }
    } else if (has_input_files){
        // TODO: Phase 4 - Parallel file reading
        // local_padded_input = mpi_read_local_padded_matrix(
        //     input_file, &input_H, &input_W,
        //     kernel_H, kernel_W,
        //     &padded_local_H, &padded_local_W, &local_start_row,
        //     active_comm
        // );

        // TEMPORARY: Use old approach for now
        if (rank == 0) {
            fprintf(stderr, "TODO: Phase 4 - Parallel file reading not yet implemented\n");
        }
    }

    // ===================================================================
    // PHASE 3: HALO EXCHANGE
    // TODO: Update to use active_comm
    // ===================================================================

    // Halo exchange
    if (local_padded_input != NULL && size > 1) {
        mpi_exchange_halos(local_padded_input, padded_local_H, padded_local_W,
                           kernel_H, active_comm);
    }

    // ===================================================================
    // PHASE 4: COMPUTATION
    // TODO: Update convolution functions to accept active_comm
    // ===================================================================

    // Allocate local output
    int local_output_H = (padded_local_H - kernel_H + 1 + stride_H - 1) / stride_H;
    int local_output_W = (padded_local_W - kernel_W + 1 + stride_W - 1) / stride_W;
    float **local_output = allocate_matrix(local_output_H, local_output_W);

    // Timing
    mpi_timer_t timer;
    if (time_execution || time_execution_seconds) {
        mpi_timer_start(&timer);
    }

    // Choose implementation
    // TODO: Update all conv functions to use active_comm parameter
    if (use_serial && rank == 0) {
        // conv2d_stride_serial(local_padded_input, padded_local_H, padded_local_W,
        //                      kernel, kernel_H, kernel_W,
        //                      stride_H, stride_W, local_output);
    } else if (use_mpi_only) {
        // conv2d_stride_mpi(local_padded_input, padded_local_H, padded_local_W,
        //                   kernel, kernel_H, kernel_W,
        //                   stride_H, stride_W, local_output, active_comm);
    } else if (use_openmp_only && rank == 0) {
        // conv2d_stride_openmp(local_padded_input, padded_local_H, padded_local_W,
        //                      kernel, kernel_H, kernel_W,
        //                      stride_H, stride_W, local_output);
    } else {
        // Default: hybrid
        // conv2d_stride_hybrid(local_padded_input, padded_local_H, padded_local_W,
        //                      kernel, kernel_H, kernel_W,
        //                      stride_H, stride_W, local_output, active_comm);
    }

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
    // PHASE 5: OUTPUT HANDLING (CONDITIONAL)
    // ===================================================================

    if (output_file) {
        // OPTION A: Write to file using parallel I/O
        // TODO: Phase 6 - Implement parallel output writing
        // mpi_write_output_parallel(output_file, local_output,
        //                           local_output_H, local_output_W,
        //                           local_start_row,
        //                           output_H, output_W,
        //                           kernel_H, kernel_W,
        //                           active_comm);

        if (rank == 0 && verbose) {
            printf("TODO: Phase 6 - Parallel output writing not yet implemented\n");
        }
    } else {
        // OPTION B: No file → Gather for logical completeness
        // TODO: Phase 7 - Implement gathering
        float **full_output = NULL;

        // mpi_gather_output_to_root(local_output, local_output_H, local_output_W,
        //                           local_start_row,
        //                           &full_output, output_H, output_W,
        //                           active_comm);

        if (rank == 0 && verbose) {
            if (output_H <= 10 && output_W <= 10) {
                // print_matrix(full_output, output_H, output_W);
            } else {
                printf("Result computed (%dx%d)\n", output_H, output_W);
            }
        }

        // Cleanup full_output on root
        if (rank == 0 && full_output != NULL) {
            free_matrix(full_output, output_H);
        }
    }

    // Verification mode (if precision flag set)
    if (rank == 0 && precision > 0 && output_file) {
        // TODO: Implement verification logic
        // float **expected = NULL;
        // int exp_H, exp_W;
        // read_matrix_from_file(output_file, &expected, &exp_H, &exp_W);
        // ... verification ...
    }

    // ===================================================================
    // Cleanup
    // ===================================================================

    if (local_padded_input) free_matrix(local_padded_input, padded_local_H);
    if (local_output) free_matrix(local_output, local_output_H);
    if (kernel) free_matrix(kernel, kernel_H);

    MPI_Comm_free(&active_comm);
    MPI_Finalize();

    return EXIT_SUCCESS;
}
