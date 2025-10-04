
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

#include "../include/core.h"

// MPI relative
int RANK = -1;
int SIZE = -1;

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

void calculate_stride_output_dims_c(struct CalcInform* calc) {
    calc->output_H = (int)ceil((double)calc->input_H / calc->stride_H);
    calc->output_W = (int)ceil((double)calc->input_W / calc->stride_W);
}

void read_size_from_files(char* input_file, char* kernel_file, struct CalcInform* calc) {
    // Read header only to get dimensions
    FILE *f = fopen(input_file, "r");
    if (f) {
        fscanf(f, "%d %d", &calc->input_H, &calc->input_W);
        fclose(f);
    }
    f = fopen(kernel_file, "r");
    if (f) {
        fscanf(f, "%d %d", &calc->kernel_H, &calc->kernel_W);
        fclose(f);
    }
}

int init_params(
    int argc, char** argv,
    struct Params* param,
    struct CalcInform* calc,
    enum AccOpt* accopt,
    enum ExecOpt* execopt
) {
    // Default values
    int input_H = -1;
    int input_W = -1;
    int kernel_H = -1;
    int kernel_W = -1;
    int stride_H = 1;
    int stride_W = 1;

    int use_serial = 0;
    int use_mpi = 0;
    int time_execution = 0;
    int time_execution_seconds = 0;
    int verbose = 0;
    int precision = 2;

    char *input_file = NULL;
    char *kernel_file = NULL;
    char *output_file = NULL;
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
            case 'h':
                print_usage(argv[0]);
                exit(EXIT_SUCCESS);

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
                    return -1;
                }
                break;
            case 'W':
                input_W = atoi(optarg);
                if (input_W <= 0) {
                    return -1;
                }
                break;
            case OPT_KH:
                kernel_H = atoi(optarg);
                if (kernel_H <= 0) {
                    return -1;
                }
                break;
            case OPT_KW:
                kernel_W = atoi(optarg);
                if (kernel_W <= 0) {
                    return -1;
                }
                break;
            case OPT_SH:
                stride_H = atoi(optarg);
                if (stride_H <= 0) {
                    return -1;
                }
                break;
            case OPT_SW:
                stride_W = atoi(optarg);
                if (stride_W <= 0) {
                    return -1;
                }
                break;
            case 's':
                use_serial = 1;
                break;
            case 'm':
                use_mpi = 1;
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
                    return -1;
                }
                break;
            case '?':
                return -1;
            default:
                return -1;
        }
    }

    // Validate input combinations
    int has_generation = (input_H > 0) && (input_W > 0) && (kernel_H > 0) && (kernel_W > 0);
    int has_input_files = (input_file != NULL) && (kernel_file != NULL);
    int has_output_file = output_file != NULL;

    if (!has_generation && !has_input_files) {
        perror("No inputs");
        return -1;
    }

    if (use_serial && use_mpi) {
        *accopt = ACC_MPI;
    }
    if (!use_serial && use_mpi) {
        *accopt = ACC_HYBRID;
    }
    if (!use_serial && !use_mpi) {
        *accopt = ACC_OMP;
    }
    if (use_serial && !use_mpi) {
        *accopt = ACC_SERIAL;
    }

    if (has_generation && !has_input_files && !has_output_file) {
        *execopt = EXEC_GenerateOnly;
    } else if (has_generation && has_input_files && has_output_file) {
        *execopt = EXEC_GenerateSave;
    } else if (!has_generation && has_input_files && has_output_file) {
        *execopt = EXEC_Verify;
    } else if (!has_generation && has_input_files && !has_output_file) {
        *execopt = EXEC_Calculate;
    } else {
        return -1;
    }

    if (*execopt == EXEC_Verify && *execopt == EXEC_Calculate) {
        read_size_from_files(input_file, kernel_file, calc);
    } else {
        calc->input_H = input_H;
        calc->input_W = input_W;
        calc->kernel_H = kernel_H;
        calc->kernel_W = kernel_W;
        calc->stride_H = stride_H;
        calc->stride_W = stride_W;
    }
    calculate_stride_output_dims_c(calc);

    param->input_filepath = input_file;
    param->kernel_filepath = kernel_file;
    param->output_filepath = output_file;

    param->precision = precision;
    param->time_execution = time_execution;
    param->time_execution_seconds = time_execution_seconds;
    param->verbose = verbose;

    return 0;
}
