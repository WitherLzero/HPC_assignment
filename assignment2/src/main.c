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

#include <assert.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <mpi.h>

#include "conv2d_mpi.h"
#include "core.h"
#include "core_mpi.h"

int main(int argc, char *argv[]) {
    struct Params param;
    struct CalcInform calc;
    enum AccOpt accopt;
    enum ExecOpt execopt;
    if (init_params(argc, argv, &param, &calc, &accopt, &execopt) == -1) {
        perror("Invalid Parameters");
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    // Determine optimal number of processes (limit to output rows)
    int optimal_processes = (SIZE > calc.output_H) ? calc.output_H : SIZE;
    bool is_active_process = (RANK < optimal_processes);

    // Create sub-communicator for active processes only
    MPI_Comm active_comm;
    MPI_Comm_split(MPI_COMM_WORLD,
                   is_active_process ? 0 : MPI_UNDEFINED,
                   RANK,
                   &active_comm);

    if (optimal_processes < SIZE) {
        DEBUGF("Using %d active processes (out of %d total)",
               optimal_processes, SIZE);
    }

    // Inactive processes exit early
    if (!is_active_process) {
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    // Update rank and size for active communicator
    MPI_Comm_rank(active_comm, &RANK);
    MPI_Comm_size(active_comm, &SIZE);


    DEBUGF("Input dimensions: %d x %d", calc.input_H, calc.input_W);
    DEBUGF("Kernel dimensions: %d x %d", calc.kernel_H, calc.kernel_W);
    DEBUGF("Stride: %d x %d", calc.stride_H, calc.stride_W);
    DEBUGF("Output dimensions: %d x %d", calc.output_H, calc.output_W);
    DEBUGF("This program is execute under %d acceleration", accopt);

    // ===================================================================
    // PHASE 3: INPUT ACQUISITION
    // ===================================================================

    float **local_padded_input = NULL, **kernel = NULL;
    int padded_local_H, padded_local_W, local_start_row;

    switch (execopt) {
    case EXEC_GenerateSave:
    case EXEC_GenerateOnly: {
        // ====================================================================
        // DUAL-PATH GENERATE MODE
        // ====================================================================

        if (calc.stride_H == 1 && calc.stride_W == 1) {
            // PATH A: stride = 1 -> Parallel generation (no overlaps)
            DEBUGF("Using PATH A: Parallel generation (stride=1)");

            // Each process generates its local portion directly
            local_padded_input = mpi_generate_local_padded_matrix(
                calc.input_H, calc.input_W, calc.kernel_H, calc.kernel_W, calc.stride_H, calc.stride_W,
                &padded_local_H, &padded_local_W, &local_start_row,
                0.0f, 1.0f, active_comm
            );

            if (local_padded_input == NULL) {
                ERRORF("Rank %d: Failed to generate local padded input", RANK);
                MPI_Comm_free(&active_comm);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }

            // Save using MPI Parallel I/O
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

                if (write_result == 0) {
                    DEBUGF("Generated input saved to %s (parallel write)", param.input_filepath);
                } else {
                    ERRORF("Failed to save input file.");
                }
            }
        } else {
            // PATH B: stride > 1 -> Root centralized generation (avoid overlap inconsistency)
            DEBUGF("Using PATH B: Root centralized generation (stride>1)");

            float **global_padded = NULL;
            int global_padded_H = -1, global_padded_W = -1;

            // Root generates global padded matrix
            ROOT_DO({
                global_padded = generate_random_matrix_into_padded(
                    calc.input_H, calc.input_W, calc.kernel_H, calc.kernel_W,
                    0.0f, 1.0f,
                    &global_padded_H, &global_padded_W
                );

                if (global_padded == NULL) {
                    ERRORF("Error: Failed to generate global padded matrix");
                    MPI_Comm_free(&active_comm);
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                // Broadcast global padded dimensions to all processes
            })
            MPI_Bcast(&global_padded_H, 1, MPI_INT, 0, active_comm);
            MPI_Bcast(&global_padded_W, 1, MPI_INT, 0, active_comm);

            assert(global_padded_H != -1 && global_padded_W != -1);

            // Distribute global padded matrix to all processes
            mpi_distribute_matrix_stride_aware(
                global_padded, global_padded_H, global_padded_W,
                calc.kernel_H, calc.kernel_W, calc.stride_H, calc.stride_W,
                &local_padded_input,
                &padded_local_H, &padded_local_W,
                &local_start_row,
                active_comm
            );

            if (local_padded_input == NULL) {
                ERRORF("Rank %d: Failed during distribution", RANK);
                ROOT_DO(free_matrix(global_padded, global_padded_H));
                MPI_Comm_free(&active_comm);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }

            // Root writes file and frees global matrix
            if (execopt == EXEC_GenerateSave) {
                ROOT_DO({
                    int write_result = write_padded_matrix_to_file(
                        global_padded, global_padded_H, global_padded_W,
                        calc.input_H, calc.input_W, calc.kernel_H, calc.kernel_W,
                        param.input_filepath
                    );

                    if (write_result == 0) {
                        DEBUGF("Generated input saved to %s (root serial write)", param.input_filepath);
                    } else {
                        ERRORF("Input save failed");
                    }

                    // Free global matrix immediately
                    free_matrix(global_padded, global_padded_H);
                    global_padded = NULL;
                })
            }
        }

        // ====================================================================
        // KERNEL GENERATION
        // ====================================================================

        ROOT_DO({
            kernel = generate_random_matrix(calc.kernel_H, calc.kernel_W, 0.0f, 1.0f);
            if (kernel == NULL) {
                ERRORF("Error: Failed to generate kernel");
                free_matrix(local_padded_input, padded_local_H);
                MPI_Comm_free(&active_comm);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }

            if (execopt == EXEC_GenerateSave) {
                write_matrix_to_file(param.kernel_filepath, kernel, calc.kernel_H, calc.kernel_W);
                DEBUGF("Generated kernel saved to %s", param.kernel_filepath);
            }
        });

        mpi_broadcast_kernel(&kernel, calc.kernel_H, calc.kernel_W, active_comm);
        break;
    }
    case EXEC_CalcToFile:
    case EXEC_PrintToScreen:
    case EXEC_Verify: {
        // ====================================================================
        // READ MODE: Load from files
        // ====================================================================
        local_padded_input = mpi_read_local_padded_matrix(
            param.input_filepath, &calc.input_H, &calc.input_W,
            calc.kernel_H, calc.kernel_W, calc.stride_H, calc.stride_W,
            &padded_local_H, &padded_local_W, &local_start_row,
            active_comm
        );

        if (local_padded_input == NULL) {
            ERRORF("Rank %d: Failed to read local padded input from file", RANK);
            MPI_Comm_free(&active_comm);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // Read kernel on root, broadcast to all
        ROOT_DO ({
            read_matrix_from_file(param.kernel_filepath, &kernel, &calc.kernel_H, &calc.kernel_W);
            if (kernel == NULL) {
                ERRORF("Error: Failed to read kernel from file %s", param.kernel_filepath);
                free_matrix(local_padded_input, padded_local_H);
                MPI_Comm_free(&active_comm);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }
        })

        mpi_broadcast_kernel(&kernel, calc.kernel_H, calc.kernel_W, active_comm);
        break;
    }
    }

    // ===================================================================
    // PHASE 4: COMPUTATION
    // ===================================================================
    int local_output_H = (padded_local_H - calc.kernel_H + 1 + calc.stride_H - 1) / calc.stride_H;
    int local_output_W = (padded_local_W - calc.kernel_W + 1 + calc.stride_W - 1) / calc.stride_W;
    float **local_output = allocate_matrix(local_output_H, local_output_W);

    // Start timing
    mpi_timer_t timer;
    mpi_timer_start(&timer);

    // Select implementation mode
    switch (accopt) {
        case ACC_SERIAL:
            // Serial mode: single-threaded, no MPI, no OpenMP
            ROOT_DO(
                conv2d_stride_serial(local_padded_input, padded_local_H, padded_local_W,
                                    kernel, calc.kernel_H, calc.kernel_W,
                                    calc.stride_H, calc.stride_W, local_output);
            )
            break;
        case ACC_OMP:
            // OpenMP-only mode: multi-threaded, no MPI
            ROOT_DO(
                conv2d_stride_openmp(local_padded_input, padded_local_H, padded_local_W,
                                    kernel, calc.kernel_H, calc.kernel_W,
                                    calc.stride_H, calc.stride_W, local_output);
            )
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

    // End timing
    mpi_timer_end(&timer);
    if (param.time_execution_seconds) {
        INFOF("Timing - Convolution with stride: %.6f seconds", timer.elapsed_time);
    } else if (param.time_execution) {
        INFOF("Timing - Convolution with stride: %.3f milliseconds",
                timer.elapsed_time * 1000.0);
    }

    // ===================================================================
    // PHASE 5: OUTPUT HANDLING
    // ===================================================================

    float **full_output = NULL;

    if (SIZE == 1) {
        full_output = local_output;
    } else if (execopt == EXEC_Verify || execopt == EXEC_GenerateOnly || execopt == EXEC_PrintToScreen) {
        // Calculate output start row for gathering
        int output_start_row;
        int output_base_rows = calc.output_H / SIZE;
        int output_remainder = calc.output_H % SIZE;

        if (calc.stride_H > 1 || calc.stride_W > 1) {
            output_start_row = RANK * output_base_rows + (RANK < output_remainder ? RANK : output_remainder);
        } else {
            output_start_row = 0;
            for (int p = 0; p < RANK; p++) {
                int p_local_output_H = output_base_rows + (p < output_remainder ? 1 : 0);
                output_start_row += p_local_output_H;
            }
        }

        mpi_gather_output(local_output, local_output_H, local_output_W,
                          output_start_row,
                          &full_output, calc.output_H, calc.output_W,
                          active_comm);
    }

    // Verification mode
    if (execopt == EXEC_Verify) {
        // Verify mode: output_file is the EXPECTED file to compare against
        if (is_root()) {
            float **expected = NULL;
            int expected_H, expected_W;

            if (read_matrix_from_file(param.output_filepath, &expected, &expected_H, &expected_W) == 0) {
                if (expected_H == calc.output_H && expected_W == calc.output_W) {
                    float tolerance = pow(10.0f, -param.precision);
                    if (full_output != NULL && compare_matrices(full_output, expected, calc.output_H, calc.output_W, tolerance)) {
                        INFOF("Verify Pass!\n");
                    } else {
                        INFOF("Verify Failed!\n");
                    }
                } else {
                    ERRORF("Verify Failed! Dimension mismatch: expected %dx%d, got %dx%d",
                           expected_H, expected_W, calc.output_H, calc.output_W);
                }
                free_matrix(expected, expected_H);
            } else {
                ERRORF("Error reading expected output file for verification");
            }
        }
    } else if (execopt == EXEC_GenerateSave || execopt == EXEC_CalcToFile) {
        // Write output to file
        int output_start_row;
        int output_base_rows = calc.output_H / SIZE;
        int output_remainder = calc.output_H % SIZE;

        if (calc.stride_H > 1 || calc.stride_W > 1) {
            output_start_row = RANK * output_base_rows + (RANK < output_remainder ? RANK : output_remainder);
        } else {
            output_start_row = 0;
            for (int p = 0; p < RANK; p++) {
                int p_local_output_H = output_base_rows + (p < output_remainder ? 1 : 0);
                output_start_row += p_local_output_H;
            }
        }

        // Parallel I/O write
        int write_result = mpi_write_output_parallel(
            param.output_filepath, local_output,
            local_output_H, local_output_W,
            output_start_row,
            calc.output_H, calc.output_W,
            calc.kernel_H, calc.kernel_W,
            active_comm
        );

        if (write_result != 0) {
            ERRORF("Error: Failed to write output file");
        }
    } else ROOT_DO(
        if (calc.output_H <= 10 && calc.output_W <= 10) {
            print_matrix(full_output, calc.output_H, calc.output_W);
        } else {
            INFOF("Result computed (%dx%d)", calc.output_H, calc.output_W);
        }
    )

    // Cleanup
    if (full_output != NULL && full_output != local_output && is_root()) {
        free_matrix(full_output, calc.output_H);
    }
    if (local_padded_input) free_matrix(local_padded_input, padded_local_H);
    if (local_output) free_matrix(local_output, local_output_H);
    if (kernel) free_matrix(kernel, calc.kernel_H);

    MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
