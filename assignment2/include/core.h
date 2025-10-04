#ifndef CORE_H
#define CORE_H
#include <stdio.h>

enum AccOpt{ ACC_SERIAL, ACC_OMP, ACC_MPI, ACC_HYBRID };
enum ExecOpt {
    // When matrix size is specified,
    // and no file is specified
    EXEC_GenerateOnly,
    // When all params are specified
    EXEC_GenerateSave,
    // When inputs files but matrix size is specified
    EXEC_Calculate,
    // When all files but matrix size are specified
    EXEC_Verify
    // Rest of the situations are UB
};

struct Params {
    char* input_filepath;
    char* kernel_filepath;
    char* output_filepath;
    int time_execution;
    int time_execution_seconds;
    int verbose;
    int precision;
};

struct CalcInform {
    int input_H;
    int input_W;
    int kernel_H;
    int kernel_W;
    int stride_H;
    int stride_W;
    int output_H;
    int output_W;
};

void print_usage(const char *program_name);
int init_params(
    int argc, char** argv,
    struct Params* param,
    struct CalcInform* calc,
    enum AccOpt* accopt,
    enum ExecOpt* execopt
);

#endif
