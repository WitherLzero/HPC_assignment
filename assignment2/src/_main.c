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
#include "../include/core.h"

int main(int argc, char *argv[]) {
    // ===================================================================
    // PHASE 1: INITIALIZATION
    // ===================================================================
    struct Params param;
    struct CalcInform calc;
    enum AccOpt accopt;
    enum ExecOpt execopt;
    if (init_params(argc, argv, &param, &calc, &accopt, &execopt) == -1) {
        perror("Invalid Parameters");
        exit(EXIT_FAILURE);
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ===================================================================
    // PHASE 1: CREATE ACTIVE_COMM
    // ===================================================================

    // Determine optimal number of processes (limit to output rows)
    int optimal_processes = (size > calc.output_H) ? calc.output_H : size;
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


    if (rank == 0 && param.verbose) {
        printf("Input dimensions: %d x %d\n", calc.input_H, calc.input_W);
        printf("Kernel dimensions: %d x %d\n", calc.kernel_H, calc.kernel_W);
        printf("Stride: %d x %d\n", calc.stride_H, calc.stride_W);
        printf("Output dimensions: %d x %d\n", calc.output_H, calc.output_W);
    }

    // ===================================================================
    // PHASE 2: INPUT ACQUISITION (PARALLEL)
    // ===================================================================

    float **local_padded_input = NULL, **kernel = NULL;
    int padded_local_H, padded_local_W, local_start_row;

    if (execopt == EXEC_GenerateOnly || execopt == EXEC_GenerateSave) {
        // Phase 3: Direct local padded matrix generation
        local_padded_input = mpi_generate_local_padded_matrix(
            calc.input_H, calc.input_W, calc.kernel_H, calc.kernel_W, calc.stride_H, calc.stride_W,
            &padded_local_H, &padded_local_W, &local_start_row,
            0.0f, 1.0f, active_comm
        );

        if (local_padded_input == NULL) {
            fprintf(stderr, "Rank %d: Failed to generate local padded input\n", rank);
            MPI_Comm_free(&active_comm);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // DEBUG: Print local padded input after generating
        if (param.verbose == 1) {
            for (int p = 0; p < size; p++) {
                if(rank == p) {
                    printf("\n=== DEBUG Rank %d: Local padded input (generated) ===\n", rank);
                    printf("Dimensions: %d x %d\n", padded_local_H, padded_local_W);
                    if (padded_local_H <= 10 && padded_local_W <= 10) {
                        print_matrix(local_padded_input, padded_local_H, padded_local_W);
                    }
                    fflush(stdout);
                }
                MPI_Barrier(active_comm);
            }
        }

        // Kernel handling (generate on root, broadcast to all)
        if (rank == 0) {
            // Generate kernel using old function for now
            kernel = generate_random_matrix(calc.kernel_H, calc.kernel_W, 0.0f, 1.0f);
            if (kernel == NULL) {
                fprintf(stderr, "Error: Failed to generate kernel\n");
                free_matrix(local_padded_input, padded_local_H);
                MPI_Comm_free(&active_comm);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }
        }

        // Broadcast kernel to all active processes
        mpi_broadcast_kernel(&kernel, calc.kernel_H, calc.kernel_W, active_comm);

        // Save generated input to file using MPI Parallel I/O (if specified)
        if (execopt == EXEC_GenerateSave) {
            int write_result = mpi_write_input_parallel(
                param.input_filepath,
                local_padded_input,
                padded_local_H, padded_local_W,
                local_start_row,
                calc.input_H, calc.input_W,
                calc.kernel_H, calc.kernel_W,
                active_comm
            );


            if (write_result == 0 && rank == 0 && param.verbose) {
                printf("Generated input saved to %s\n", param.input_filepath);
            }
        }

        // Save generated kernel to file (only on root)
        if (rank == 0 && param.kernel_filepath) {
            write_matrix_to_file(param.kernel_filepath, kernel, calc.kernel_H, calc.kernel_W);
            if (param.verbose) printf("Generated kernel saved to %s\n", param.kernel_filepath);
        }

        if (rank == 0 && param.verbose) {
            printf("Generated local padded input and kernel successfully\n");
        }
    }
    if (execopt == EXEC_Calculate || execopt == EXEC_Verify) {
        // Phase 4: Parallel file reading with MPI I/O
        local_padded_input = mpi_read_local_padded_matrix(
            param.input_filepath, &calc.input_H, &calc.input_W,
            calc.kernel_H, calc.kernel_W, calc.stride_H, calc.stride_W,
            &padded_local_H, &padded_local_W, &local_start_row,
            active_comm
        );

        if (local_padded_input == NULL) {
            fprintf(stderr, "Rank %d: Failed to read local padded input from file\n", rank);
            MPI_Comm_free(&active_comm);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // Kernel handling (read on root, broadcast to all)
        if (rank == 0) {
            read_matrix_from_file(param.kernel_filepath, &kernel, &calc.kernel_H, &calc.kernel_W);
            if (kernel == NULL) {
                fprintf(stderr, "Error: Failed to read kernel from file %s\n", param.kernel_filepath);
                free_matrix(local_padded_input, padded_local_H);
                MPI_Comm_free(&active_comm);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }
        }

        // Broadcast kernel to all active processes
        mpi_broadcast_kernel(&kernel, calc.kernel_H, calc.kernel_W, active_comm);

        if (rank == 0 && param.verbose) {
            printf("Read local padded input from file successfully\n");
        }
    }

    // ===================================================================
    // PHASE 4: COMPUTATION
    // Halo exchange is now handled internally by MPI/Hybrid functions
    // ===================================================================

    // Allocate local output
    int local_output_H = (padded_local_H - calc.kernel_H + 1 + calc.stride_H - 1) / calc.stride_H;
    int local_output_W = (padded_local_W - calc.kernel_W + 1 + calc.stride_W - 1) / calc.stride_W;
    float **local_output = allocate_matrix(local_output_H, local_output_W);

    // if (verbose) {
    //     printf("DEBUG rank %d: ACTUAL padded_local_H=%d, local_output_H=%d\n",
    //            rank, padded_local_H, local_output_H);
    // }

    // Timing
    mpi_timer_t timer;
    if (param.time_execution || param.time_execution_seconds) {
        mpi_timer_start(&timer);
    }

    // Choose implementation
    // Serial and OpenMP-only modes are validated to run with exactly 1 process
    switch (accopt) {
        case ACC_SERIAL:
            // Serial mode: single-threaded, no MPI, no OpenMP
            conv2d_stride_serial(local_padded_input, padded_local_H, padded_local_W,
                                kernel, calc.kernel_H, calc.kernel_W,
                                calc.stride_H, calc.stride_W, local_output);
            break;
        case ACC_OMP:
            // OpenMP-only mode: multi-threaded, no MPI
            conv2d_stride_openmp(local_padded_input, padded_local_H, padded_local_W,
                                kernel, calc.kernel_H, calc.kernel_W,
                                calc.stride_H, calc.stride_W, local_output);
            break;
        case ACC_MPI:
            // MPI-only mode: distributed memory, no OpenMP within each process
            conv2d_stride_mpi(local_padded_input, padded_local_H, padded_local_W,
                            kernel, calc.kernel_H, calc.kernel_W,
                            calc.stride_H, calc.stride_W, local_output, active_comm);
            break;
        case ACC_HYBRID:
            // Default: hybrid mode (MPI + OpenMP)
            conv2d_stride_hybrid(local_padded_input, padded_local_H, padded_local_W,
                                kernel, calc.kernel_H, calc.kernel_W,
                                calc.stride_H, calc.stride_W, local_output, active_comm);
            break;
    }


    float **full_output = NULL;

    if (size == 1) {
        // Single process: local_output IS the full output
        full_output = local_output;
    } else {
        // Multi-process: need to gather
        // For stride > 1, calculate which global output rows this process produces
        // based on its input start row

        int output_start_row;

        if (calc.stride_H > 1 || calc.stride_W > 1) {
            // STRIDE-AWARE: Use OUTPUT-FIRST distribution for stride > 1
            // Distribute output rows evenly across processes
            int output_base_rows = calc.output_H / size;
            int output_remainder = calc.output_H % size;
            output_start_row = rank * output_base_rows + (rank < output_remainder ? rank : output_remainder);
        } else {
            // OUTPUT-FIRST: Distribute output rows evenly (original logic for stride=1)
            int base_output_rows = calc.output_H / size;
            int output_remainder = calc.output_H % size;

            output_start_row = 0;
            for (int p = 0; p < rank; p++) {
                int p_local_output_H = base_output_rows + (p < output_remainder ? 1 : 0);
                output_start_row += p_local_output_H;
            }
        }

        // if (verbose) {
        //     printf("DEBUG: Gathering - rank=%d, local_start_row=%d, output_start_row=%d, local_output_H=%d\n",
        //            rank, local_start_row, output_start_row, local_output_H);
        // }

        mpi_gather_output(local_output, local_output_H, local_output_W,
                        output_start_row,
                        &full_output, calc.output_H, calc.output_W,
                        active_comm);

        // DEBUG: Print gathered output
        // if (verbose && rank == 0) {
        //     printf("\n=== DEBUG Rank 0: Full output (after gather) ===\n");
        //     printf("Dimensions: %d x %d\n", output_H, output_W);
        //     if (output_H <= 10 && output_W <= 10) {
        //         print_matrix(full_output, output_H, output_W);
        //     }
        // }
    }

    // WARNING TODO: This is a wrong implementation of timing. Should have barrier for end of execution.
    if (param.time_execution || param.time_execution_seconds) {
        mpi_timer_end(&timer, active_comm);
        if (rank == 0) {
            if (param.time_execution_seconds) {
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
    // Handle verification mode
    if (execopt == EXEC_Verify) {
        // Verify mode: output_file is the EXPECTED file to compare against
        if (rank == 0) {
            float **expected = NULL;
            int expected_H, expected_W;

            if (read_matrix_from_file(param.output_filepath, &expected, &expected_H, &expected_W) == 0) {
                if (expected_H == calc.output_H && expected_W == calc.output_W) {
                    float tolerance = pow(10.0f, -param.precision);
                    if (full_output != NULL && compare_matrices(full_output, expected, calc.output_H, calc.output_W, tolerance)) {
                        printf("Verify Pass!\n");
                    } else {
                        printf("Verify Failed!\n");
                    }
                } else {
                    printf("Verify Failed! Dimension mismatch: expected %dx%d, got %dx%d\n",
                           expected_H, expected_W, calc.output_H, calc.output_W);
                }
                free_matrix(expected, expected_H);
            } else {
                printf("Error reading expected output file for verification\n");
            }
        }
    } else if (execopt == EXEC_GenerateSave || execopt == EXEC_Calculate) {
        // Write mode: save output to file
        // TODO: Phase 6 - Implement parallel output writing
        // mpi_write_output_parallel(output_file, local_output,
        //                           local_output_H, local_output_W,
        //                           local_start_row,
        //                           output_H, output_W,
        //                           calc.kernel_H, calc.kernel_W,
        //                           active_comm);

        if (rank == 0 && param.verbose) {
            printf("TODO: Phase 6 - Parallel output writing not yet implemented\n");
        }
    } else if (rank == 0 && param.verbose && full_output != NULL) {
        // No file, no verify: just display (verbose mode)
        if (calc.output_H <= 10 && calc.output_W <= 10) {
            print_matrix(full_output, calc.output_H, calc.output_W);
        } else {
            printf("Result computed (%dx%d)\n", calc.output_H, calc.output_W);
        }
    }

    // Cleanup full_output (only if it was gathered, not if it's an alias to local_output)
    if (full_output != NULL && full_output != local_output && rank == 0) {
        free_matrix(full_output, calc.output_H);
    }

    // ===================================================================
    // Cleanup
    // ===================================================================

    if (local_padded_input) free_matrix(local_padded_input, padded_local_H);
    if (local_output) free_matrix(local_output, local_output_H);
    if (kernel) free_matrix(kernel, calc.kernel_H);

    MPI_Comm_free(&active_comm);
    MPI_Finalize();

    return EXIT_SUCCESS;
}
