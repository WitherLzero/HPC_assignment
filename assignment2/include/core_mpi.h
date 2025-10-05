#pragma once
#include "stdbool.h"

extern int RANK;
extern int SIZE;

void init_mpi(int* argc, char*** argv);
inline bool is_root() { return RANK == 0; }

#define ROOT_DO(block) do { if (is_root()) { block; } } while (0);

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define DEBUGF(...) \
    ROOT_DO( \
        if (VERBOSE) { \
            printf(ANSI_COLOR_BLUE "[DEBUG] " ANSI_COLOR_RESET); \
            printf(__VA_ARGS__); \
        } \
    )

#define ERRORF(...) \
    ROOT_DO( \
        fprintf(stderr, ANSI_COLOR_RED "[ERROR] " ANSI_COLOR_RESET); \
        fprintf(stderr, __VA_ARGS__); \
    )
