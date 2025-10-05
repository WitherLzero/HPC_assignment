#pragma once
#include "stdbool.h"

extern int RANK;
extern int SIZE;

void init_mpi(int* argc, char*** argv);
inline bool is_root() { return RANK == 0; }

#define ROOT_DO(block) do { if (is_root()) { block; } } while (0);

#define DEBUGF(...) \
    ROOT_DO( \
        if (VERBOSE) { \
            printf(__VA_ARGS__); \
        } \
    )

#define ERRORF(...) \
    ROOT_DO( \
        fprintf(stderr, __VA_ARGS__); \
    )
