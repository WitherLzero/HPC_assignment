# Memory-Optimized MPI Parallel I/O Implementation Plan

## Document Purpose
This is a comprehensive implementation plan for refactoring Assignment 2's MPI implementation to eliminate memory waste and leverage MPI Parallel I/O. This plan is based on extensive design discussions and follows an **incremental development approach with testing at each stage**.

---

## IMPLEMENTATION PROGRESS LOG

### Session 1: Phase 1-3 Implementation (Completed)

**Date**: Current session
**Status**: ✅ Phase 1-3 fully implemented and verified

#### What We Completed:

1. **Phase 1: Infrastructure Setup** ✅
   - Created new `_main.c` with refactored structure
   - Moved `active_comm` creation EARLY (before matrix operations)
   - Inactive processes exit immediately after communicator split
   - Updated Makefile to use `_main.c` instead of `main.c`
   - All subsequent operations use `active_comm` parameter

2. **Phase 2: Padding Calculation & Local Dimensions** ✅
   - Implemented `calculate_padding_for_process()` in `src/conv2d_mpi.c`
   - Implemented `calculate_local_dimensions()` in `src/conv2d_mpi.c`
   - **Key distinction clarified**:
     - **Padding** = actual zero-value padding (top/bottom for first/last process only)
     - **Halo** = extra space for neighbor data exchange (separate from padding)
   - Handles non-divisible cases correctly (first `remainder` processes get +1 row)

3. **Phase 3: Direct Local Padded Matrix Generation** ✅
   - Implemented `mpi_generate_local_padded_matrix()` in `src/io_mpi.c`
   - Each process generates ONLY its local portion + padding + halo space
   - Uses xorshift32 random generation (non-deterministic, per-process seed)
   - OpenMP parallelized generation with per-thread seeds
   - Memory savings: NO global matrix on root process!

4. **Halo Exchange Verification** ✅
   - Existing `mpi_exchange_halos()` works correctly with new structure
   - Verified halo exchange fills neighbor data correctly
   - **Confirmed**: Halo size depends ONLY on kernel size, NOT stride
     - 3×3 kernel → halo = 1 row
     - 5×5 kernel → halo = 2 rows
   - Tested with multiple configurations:
     - 2/3/4 processes
     - 3×3 and 5×5 kernels
     - Different strides (verified stride doesn't affect halo)
     - Non-divisible cases (7 rows ÷ 4 processes)

#### Files Modified:
- `src/_main.c` - New refactored main file
- `src/conv2d_mpi.c` - Added Phase 2 padding utilities
- `src/io_mpi.c` - Added Phase 3 generation function, uses `initialize_matrix()`
- `include/conv2d_mpi.h` - Added function declarations
- `Makefile` - Updated to use `_main.c`

#### Key Design Decisions Made:

1. **Padding vs Halo Separation**:
   - First process: top padding + bottom halo
   - Middle processes: top halo + bottom halo (NO padding)
   - Last process: top halo + bottom padding
   - Left/right padding: ALL processes

2. **Random Generation Strategy**:
   - Using xorshift32 (fast, good quality)
   - Non-deterministic seed: `time(NULL) + rank * 12345 + thread_num * 67890`
   - OpenMP parallelized with per-thread seeds

3. **Memory Layout**:
   ```
   Rank 0: [top_pad][data_rows][bottom_halo]
   Rank 1: [top_halo][data_rows][bottom_halo]
   Rank N: [top_halo][data_rows][bottom_pad]
   ```

#### Testing Completed:
- ✅ Padding calculation correctness
- ✅ Local dimension distribution (divisible and non-divisible)
- ✅ Matrix generation with proper padding/halo allocation
- ✅ Halo exchange fills neighbor data correctly
- ✅ Multiple kernel sizes (3×3, 5×5)
- ✅ Different stride values (verified stride independence)
- ✅ Multiple process counts (2, 3, 4)

#### What's NOT Done Yet:
- ~~Phase 4: Parallel file reading (MPI_File_read_at)~~ ✅ COMPLETED
- **Phase 4+: Parallel INPUT file writing (bonus)** ✅ COMPLETED
- Phase 5: Update convolution functions to use active_comm
- Phase 6: Parallel OUTPUT file writing (MPI_File_write_at_all)
- Phase 7: Conditional output gathering
- Phase 8-10: Integration, performance testing, documentation

### Session 2: Phase 4 & 4+ Implementation (Completed)

**Date**: Current session (Oct 2)
**Status**: ✅ Phase 4 and Phase 4+ (bonus) fully implemented and verified

#### What We Completed:

1. **Phase 4: Parallel File Reading** ✅
   - Implemented `mpi_read_local_padded_matrix()` in `src/io_mpi.c`
   - Each process reads only its local portion using `MPI_File_read_at()`
   - Correctly handles text file format: `"0.xxx "` (6 chars per float)
   - Last column format: `"0.xxx"` (5 chars, no trailing space)
   - Handles Windows line endings (`\r\n` = 2 bytes)
   - Allocates padded matrices with proper padding/halo regions
   - Tested successfully with f0.txt (6×6), f2.txt (7×7), and multiple process counts

2. **Phase 4+ (Bonus): Parallel INPUT File Writing** ✅
   - Implemented `mpi_write_input_parallel()` in `src/io_mpi.c`
   - Each process writes only its data portion (excluding padding/halo) directly to file
   - Uses `MPI_File_write_at()` for parallel I/O
   - Avoids memory waste of extracting padding, gathering to root, then writing
   - Successfully tested: Generated matrices written to file and verified
   - Round-trip verified: Write → Read → Compare successful

#### Files Modified:
- `src/io_mpi.c`:
  - Added `mpi_read_local_padded_matrix()` function
  - Added `mpi_write_input_parallel()` function
  - Temporarily disabled OpenMP in `generate_random_matrix()` for debug stability
- `include/conv2d_mpi.h`:
  - Added `mpi_read_local_padded_matrix()` declaration
  - Added `mpi_write_input_parallel()` declaration
  - Added `generate_random_matrix()` declaration
- `src/_main.c`:
  - Integrated parallel file reading for input files
  - Integrated parallel input writing when generating matrices
  - Added debug sections for matrix inspection (DEBUG mode only)

#### Key Design Decisions Made:

1. **File Format Handling**:
   - Correctly identified format: `"0.xxx "` = 6 chars per float (including decimal and space)
   - Last column: `"0.xxx"` = 5 chars (no trailing space)
   - Windows line endings: `\r\n` = 2 bytes per row
   - Total bytes per row: `W*6 - 1 + 2 = W*6 + 1`

2. **Parallel Input Writing Strategy**:
   - Calculate padding for each process to identify data vs padding/halo regions
   - Each process writes only `local_H × local_W` data portion (no padding/halo)
   - Root writes header first, then all processes write their rows at calculated offsets
   - Uses same file format as reading for consistency

3. **Debug Mode Issue**:
   - Segfault in DEBUG mode traced to OpenMP directives in `generate_random_matrix()`
   - Temporarily disabled OpenMP parallelization in kernel generation for stability
   - RELEASE mode works perfectly
   - Debug prints added for verification (hangs on print_matrix in some cases)

#### Testing Completed:
- ✅ Read f0.txt (6×6) with 2 processes - all columns loaded correctly
- ✅ Read f2.txt (7×7) with 2-3 processes - all columns loaded correctly
- ✅ Different process counts (1, 2, 3, 4) produce correct results
- ✅ Generate 6×6 matrix with 2 processes and write to file
- ✅ Generate 10×10 matrix and write to file
- ✅ File format matches expected format (verified manually)
- ✅ Round-trip test: Generate → Write → Read → Verify successful

#### Memory Savings Achieved:
With Phase 4 complete, we now have:
- **NO global matrix on root for input reading** ✅
- Each process reads only its portion (~1/N of file size)
- **NO global matrix on root for input writing** ✅ (bonus)
- Each process writes only its portion directly

#### Next Steps for Future Session:
1. **Fix debug mode segfault** (optional - re-enable OpenMP safely or investigate root cause)
2. **Phase 5**: Update all convolution functions to accept `active_comm` parameter
3. **Phase 6**: Implement `mpi_write_output_parallel()` for distributed OUTPUT writing
4. **Phase 7**: Implement `mpi_gather_output_to_root()` for non-file scenarios
5. **Phase 8**: End-to-end integration testing with actual computation
6. **Phase 9-10**: Performance testing and documentation

---

---

## Table of Contents
1. [Problem Analysis](#problem-analysis)
2. [Architecture Overview](#architecture-overview)
3. [Key Design Decisions](#key-design-decisions)
4. [Implementation Phases](#implementation-phases)
5. [Testing Strategy](#testing-strategy)
6. [API Reference](#api-reference)
7. [Future Optimizations](#future-optimizations)

---

## Problem Analysis

### Current Memory Waste Issues

**Current approach uses 5 large matrices on root process:**
```c
float **input;           // Full global input (H × W)
float **padded_input;    // Full padded input ((H+pad) × (W+pad))
float **output;          // Full global output
float **local_matrix;    // Local portion for computation
float **gathered_output; // Assembled results from all processes
```

**For 10000×10000 matrix:**
- Each matrix: ~400MB (10000×10000×4 bytes)
- Total on root: 5 × 400MB = **2GB wasted**

**For 50000×50000 matrix:**
- Each matrix: ~10GB
- Total on root: 5 × 10GB = **50GB wasted** ⚠️ Exceeds typical node memory!

### Root Causes
1. **Sequential I/O**: Root reads entire file, then scatters
2. **Unnecessary copying**: Input → Padded duplication
3. **Gather overhead**: Local results → gathered_output → output
4. **No direct file writing**: All output goes through root

---

## Architecture Overview

### New Memory-Efficient Flow

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: INITIALIZATION                                     │
│ - MPI_Init                                                  │
│ - Parse CLI arguments (all processes)                       │
│ - Calculate output dimensions                               │
│ - Determine active processes                                │
│ - Create active_comm sub-communicator                       │
│ - Inactive processes exit early                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: INPUT ACQUISITION (PARALLEL)                       │
│                                                             │
│ ┌─────────────────┐        ┌──────────────────┐           │
│ │ Generate Mode   │   OR   │ File Read Mode   │           │
│ └─────────────────┘        └──────────────────┘           │
│         ↓                           ↓                      │
│ Each process:                Each process:                 │
│ - Calculate local portion    - Calculate local portion     │
│ - Calculate padding needs    - MPI_File_read_at() own rows│
│ - Allocate padded matrix     - Allocate padded matrix      │
│ - Generate data + padding    - Fill data + padding         │
│                                                             │
│ Result: Each process has LOCAL_PADDED_INPUT only           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: KERNEL & HALO EXCHANGE                             │
│ - Root reads/generates kernel                               │
│ - MPI_Bcast kernel to all (using active_comm)              │
│ - MPI_Exchange_halos (fill padding with neighbor data)     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: COMPUTATION                                        │
│ - Each process: Allocate LOCAL_OUTPUT                      │
│ - Compute: conv2d_stride_hybrid/mpi/openmp                 │
│ - Result: Each process has LOCAL_OUTPUT                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: OUTPUT HANDLING (CONDITIONAL)                      │
│                                                             │
│ IF output_file specified:                                  │
│   - Each process writes directly via MPI_File_write_at_all│
│   - NO gathering needed! ✅                                │
│                                                             │
│ ELSE (no output file):                                     │
│   - Gather local results to root                          │
│   - Root assembles FULL_OUTPUT                            │
│   - Print/verify if verbose/precision mode                │
│   - Ensures logical completeness ✅                        │
└─────────────────────────────────────────────────────────────┘
```

### Memory Footprint Comparison

| Scenario | Old Approach | New Approach | Savings |
|----------|--------------|--------------|---------|
| **With -o file** | Root: 5× matrix size | All processes: 1× (distributed) | **80% reduction** |
| **Without -o** | Root: 5× matrix size | Root: 2× (local + full output) | **60% reduction** |
| **Large matrix (50000×50000)** | Root: 50GB | Distributed: ~12.5GB per process (4 procs) | **Fits in memory!** |

---

## Key Design Decisions

### Decision 1: Active Process Sub-communicator (Early Creation)

**Location**: Immediately after CLI argument parsing, BEFORE any matrix operations

```c
// Calculate output dimensions to determine active processes
calculate_stride_output_dims(input_H, input_W, stride_H, stride_W,
                             &output_H, &output_W);

// Create sub-communicator (MOVE TO HERE from line 285)
int optimal_processes = (size > output_H) ? output_H : size;
bool is_active_process = (rank < optimal_processes);

MPI_Comm active_comm;
MPI_Comm_split(MPI_COMM_WORLD,
               is_active_process ? 0 : MPI_UNDEFINED,
               rank, &active_comm);

// Inactive processes exit early
if (!is_active_process) {
    MPI_Finalize();
    return EXIT_SUCCESS;
}

// FROM HERE: All operations use active_comm
```

**Rationale**: Prevents inactive processes from allocating unnecessary memory.

### Decision 2: Unified Padded Matrix Generation/Reading

**Single function handles both generation and file reading:**

```c
float** mpi_get_local_padded_input(
    const char* input_file,      // NULL for generation
    int H_global, int W_global,
    int kH, int kW,
    int* padded_local_H,
    int* padded_local_W,
    int* local_start_row,
    MPI_Comm comm
);
```

**Key features:**
- Calculates local portion based on rank
- Determines padding (top/bottom/left/right) based on process position
- Allocates padded matrix directly (no intermediate buffer)
- Fills data region (random generation or file read)
- Initializes padding regions to zero
- Returns ready-to-use padded matrix

### Decision 3: Padding Calculation Algorithm

**Universal padding logic for any rank:**

```c
void calculate_padding_for_process(
    int rank, int size,
    int kH, int kW,
    int* pad_top, int* pad_bottom,
    int* pad_left, int* pad_right
) {
    int halo = (kH - 1) / 2;

    // Top padding
    if (rank == 0) {
        *pad_top = (kH - 1) / 2;        // "Same" padding
    } else {
        *pad_top = halo;                // Halo (filled by exchange)
    }

    // Bottom padding
    if (rank == size - 1) {
        *pad_bottom = kH - 1 - (kH - 1) / 2;  // "Same" padding (asymmetric)
    } else {
        *pad_bottom = halo;             // Halo (filled by exchange)
    }

    // Horizontal padding (all processes same)
    *pad_left = (kW - 1) / 2;
    *pad_right = kW - 1 - (kW - 1) / 2;
}
```

**Visual representation:**
```
Process 0 (top):
┌─────────────────────────┐
│ ZZZZZZZZZZZZZZZZZZZZZZZ │ ← Top: "same" padding
│ Z DATA DATA DATA DATA Z │ ← Left/Right: "same" padding
│ Z DATA DATA DATA DATA Z │
│ Z HALO HALO HALO HALO Z │ ← Bottom: halo for P1
└─────────────────────────┘

Process 1 (middle):
┌─────────────────────────┐
│ Z HALO HALO HALO HALO Z │ ← Top: halo for P0
│ Z DATA DATA DATA DATA Z │
│ Z DATA DATA DATA DATA Z │
│ Z HALO HALO HALO HALO Z │ ← Bottom: halo for P2
└─────────────────────────┘

Process N-1 (bottom):
┌─────────────────────────┐
│ Z HALO HALO HALO HALO Z │ ← Top: halo for P(N-2)
│ Z DATA DATA DATA DATA Z │
│ Z DATA DATA DATA DATA Z │
│ ZZZZZZZZZZZZZZZZZZZZZZZ │ ← Bottom: "same" padding
└─────────────────────────┘
```

### Decision 4: Random Seed Strategy

**For deterministic generation (testing/debugging):**
```c
// Position-based seeding ensures same values across runs
for (int i = 0; i < local_H; i++) {
    int global_row = local_start_row + i;
    for (int j = 0; j < W_global; j++) {
        unsigned int seed = global_row * W_global + j;
        float value = generate_from_seed(seed);
        padded_matrix[i + pad_top][j + pad_left] = value;
    }
}
```

**For production (random values each run):**
```c
unsigned int seed = (unsigned int)time(NULL) + rank * 12345;
srand(seed);
```

### Decision 5: Text File Format (Fixed-Width for Parallel I/O)

**Analysis of existing test files:**
- Input files: Range [0.000, 0.999], format "0.xxx" (5 chars including space)
- Output files: Range [0.000, kH×kW], format varies

**Fixed-width format decision:**

**Input files:**
```c
// Always "0.xxx " (5 characters)
fprintf(fp, "%.3f ", value);  // value ∈ [0, 1)
```

**Output files:**
```c
// Adapt to kernel size:
// 3×3 kernel: max=9    → "x.xxx " (5 chars)
// 5×5 kernel: max=25   → "xx.xxx" (6 chars)
// 7×7 kernel: max=49   → "xx.xxx" (6 chars)
// 11×11 kernel: max=121 → "xxx.xxx" (7 chars)

int max_output = kH * kW;
int chars_per_float = (max_output < 10) ? 5 : (max_output < 100) ? 6 : 7;
```

**Offset calculation (now deterministic!):**
```c
// Input file:
int bytes_per_row = W_global * 5 + 1;  // 5 chars/float + newline
MPI_Offset offset = header_bytes + global_row * bytes_per_row;

// Output file:
int bytes_per_row = W_global * chars_per_float + 1;
MPI_Offset offset = header_bytes + global_row * bytes_per_row;
```

### Decision 6: Halo Exchange Strategy

**Use unified halo exchange for both generation and file reading:**

```c
// After getting local padded input (halos contain zeros)
mpi_exchange_halos(padded_input, padded_H, padded_W, kH, active_comm);

// Now halos contain actual neighbor data
```

**Why not read halos directly from file?**
- ❌ Breaks random generation (P1 can't generate P0's data)
- ❌ Creates divergence between generation and file-reading logic
- ✅ Unified halo exchange works for both cases

### Decision 7: Output Handling Strategy

**Always gather to ensure logical completeness:**

```c
// After local computation
float **full_output = NULL;

if (output_file) {
    // OPTION A: Write to file (parallel I/O)
    mpi_write_output_parallel(output_file, local_output, ...);

    // Still gather for logical completeness (can skip if memory constrained)
    mpi_gather_output_to_root(local_output, ..., &full_output, ...);

} else {
    // OPTION B: No file → Must gather for logical result
    mpi_gather_output_to_root(local_output, ..., &full_output, ...);

    if (rank == 0 && verbose) {
        print_matrix(full_output, output_H, output_W);
    }
}

// Verification mode (if precision flag set)
if (rank == 0 && precision > 0 && expected_file) {
    verify_output(full_output, expected_file, precision);
}
```

**Rationale**: Ensures we always have the complete logical result, not just distributed fragments.

### Decision 8: Function Signature Updates

**All MPI functions must accept `MPI_Comm comm` parameter:**

```c
// OLD (hardcoded MPI_COMM_WORLD):
void mpi_broadcast_kernel(..., MPI_COMM_WORLD);

// NEW (flexible communicator):
void mpi_broadcast_kernel(..., MPI_Comm comm);
```

**Functions requiring update:**
- `mpi_broadcast_kernel()`
- `mpi_exchange_halos()`
- `mpi_distribute_matrix()` (if kept)
- `mpi_distribute_matrix_stride_aware()` (if kept)
- `mpi_gather_output()`
- All convolution functions (`conv2d_stride_mpi`, `conv2d_stride_hybrid`)
- All I/O functions

---

## Implementation Phases

### Phase 1: Infrastructure Setup ✅ TEST AFTER THIS

**Goal**: Reorganize main flow and create active_comm early

**Tasks:**
1. Move output dimension calculation before matrix operations
2. Create `active_comm` sub-communicator immediately after CLI parsing
3. Add early exit for inactive processes
4. Update all function calls to pass `active_comm` instead of `MPI_COMM_WORLD`

**Changes in `src/main.c`:**
```c
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse CLI arguments (all processes)
    // ... getopt parsing ...

    // Calculate output dimensions
    int output_H, output_W;
    calculate_stride_output_dims(input_H, input_W, stride_H, stride_W,
                                 &output_H, &output_W);

    // CREATE ACTIVE_COMM (MOVED HERE!)
    int optimal_processes = (size > output_H) ? output_H : size;
    bool is_active_process = (rank < optimal_processes);

    MPI_Comm active_comm;
    MPI_Comm_split(MPI_COMM_WORLD,
                   is_active_process ? 0 : MPI_UNDEFINED,
                   rank, &active_comm);

    if (rank == 0 && optimal_processes < size) {
        printf("Using %d active processes (out of %d total)\n",
               optimal_processes, size);
    }

    // Inactive processes exit early
    if (!is_active_process) {
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    // FROM HERE: Only active processes, use active_comm
    // ... rest of implementation ...
}
```

**Function signature updates:**
```c
// Update all these function signatures to accept MPI_Comm comm:
void mpi_broadcast_kernel(float ***kernel, int kH, int kW, MPI_Comm comm);
void mpi_exchange_halos(float **matrix, int H, int W, int kH, MPI_Comm comm);
void conv2d_stride_mpi(..., MPI_Comm comm);
void conv2d_stride_hybrid(..., MPI_Comm comm);
```

**TEST PHASE 1:**
```bash
# Test 1: Verify active process calculation
mpirun --allow-run-as-root -np 8 ./build/conv_stride_test -H 10 -W 10 -kH 3 -kW 3 -t
# Expected: Should print "Using 3 active processes (out of 8 total)"
# Expected: Only ranks 0-2 should execute, ranks 3-7 exit early

# Test 2: Verify with enough output rows
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -t
# Expected: All 4 processes active

# Test 3: Check for errors/crashes
mpirun --allow-run-as-root -np 2 ./build/conv_stride_test -H 50 -W 50 -kH 3 -kW 3 -v
# Expected: Should run without errors, verbose output shown
```

---

### Phase 2: Padding Calculation & Local Matrix Allocation ✅ TEST AFTER THIS

**Goal**: Implement intelligent padding calculation for distributed processes

**Tasks:**
1. Implement `calculate_padding_for_process()`
2. Implement `calculate_local_dimensions()`
3. Test padding logic for different process positions

**New file: `src/padding_utils.c`**
```c
void calculate_padding_for_process(
    int rank, int size,
    int kH, int kW,
    int* pad_top, int* pad_bottom,
    int* pad_left, int* pad_right
) {
    int halo = (kH - 1) / 2;

    *pad_top = (rank == 0) ? (kH - 1) / 2 : halo;
    *pad_bottom = (rank == size - 1) ? (kH - 1 - (kH - 1) / 2) : halo;
    *pad_left = (kW - 1) / 2;
    *pad_right = kW - 1 - (kW - 1) / 2;
}

void calculate_local_dimensions(
    int rank, int size,
    int H_global, int W_global,
    int kH, int kW,
    int* local_H, int* local_W,
    int* local_start_row,
    int* padded_local_H, int* padded_local_W
) {
    // Calculate local portion
    int base_rows = H_global / size;
    int remainder = H_global % size;
    *local_H = base_rows + (rank < remainder ? 1 : 0);
    *local_start_row = rank * base_rows + (rank < remainder ? rank : remainder);
    *local_W = W_global;

    // Calculate padding
    int pad_top, pad_bottom, pad_left, pad_right;
    calculate_padding_for_process(rank, size, kH, kW,
                                   &pad_top, &pad_bottom,
                                   &pad_left, &pad_right);

    *padded_local_H = *local_H + pad_top + pad_bottom;
    *padded_local_W = *local_W + pad_left + pad_right;
}
```

**TEST PHASE 2:**
```bash
# Create a test program that prints padding info
mpirun --allow-run-as-root -np 4 ./test_padding -H 100 -W 100 -kH 3 -kW 3

# Expected output (for 3×3 kernel):
# Rank 0: local_H=25, start=0, pad_top=1, pad_bottom=1 (halo), pad_left=1, pad_right=1
# Rank 1: local_H=25, start=25, pad_top=1 (halo), pad_bottom=1 (halo), pad_left=1, pad_right=1
# Rank 2: local_H=25, start=50, pad_top=1 (halo), pad_bottom=1 (halo), pad_left=1, pad_right=1
# Rank 3: local_H=25, start=75, pad_top=1 (halo), pad_bottom=1, pad_left=1, pad_right=1

# Verify total coverage:
# Sum of local_H should equal H_global: 25+25+25+25 = 100 ✓
```

---

### Phase 3: Random Matrix Generation (Direct Padded) ✅ TEST AFTER THIS

**Goal**: Generate padded matrices directly without intermediate buffers

**Tasks:**
1. Implement `mpi_generate_local_padded_matrix()`
2. Use position-based seeding for deterministic generation
3. Test generation correctness

**New function in `src/io_mpi.c`:**
```c
float** mpi_generate_local_padded_matrix(
    int H_global, int W_global,
    int kH, int kW,
    int* padded_local_H,
    int* padded_local_W,
    int* local_start_row,
    float min_val, float max_val,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Calculate local dimensions and padding
    int local_H, local_W;
    calculate_local_dimensions(rank, size, H_global, W_global, kH, kW,
                               &local_H, &local_W, local_start_row,
                               padded_local_H, padded_local_W);

    int pad_top, pad_bottom, pad_left, pad_right;
    calculate_padding_for_process(rank, size, kH, kW,
                                   &pad_top, &pad_bottom,
                                   &pad_left, &pad_right);

    // Allocate padded matrix
    float **matrix = allocate_matrix(*padded_local_H, *padded_local_W);

    // Initialize all to zero (padding regions)
    for (int i = 0; i < *padded_local_H; i++) {
        for (int j = 0; j < *padded_local_W; j++) {
            matrix[i][j] = 0.0f;
        }
    }

    // Generate data region with position-based seeding
    float range = max_val - min_val;

    for (int i = 0; i < local_H; i++) {
        int global_row = *local_start_row + i;
        int padded_row = i + pad_top;

        for (int j = 0; j < local_W; j++) {
            // Deterministic seed based on global position
            unsigned int seed = global_row * W_global + j;

            // Simple LCG random generator
            seed = (1103515245 * seed + 12345) & 0x7fffffff;
            float value = min_val + (float)seed / (float)0x7fffffff * range;

            matrix[padded_row][j + pad_left] = value;
        }
    }

    return matrix;
}
```

**TEST PHASE 3:**
```bash
# Test 1: Generate and verify determinism
mpirun --allow-run-as-root -np 2 ./build/conv_stride_test -H 10 -W 10 -kH 3 -kW 3 -v -o gen1.txt
mpirun --allow-run-as-root -np 2 ./build/conv_stride_test -H 10 -W 10 -kH 3 -kW 3 -v -o gen2.txt
diff gen1.txt gen2.txt
# Expected: Files should be identical (deterministic generation)

# Test 2: Verify values in range [0, 1)
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 20 -W 20 -kH 3 -kW 3 -v
# Expected: All printed values should be in [0.000, 0.999] range

# Test 3: Different process counts produce same result
mpirun --allow-run-as-root -np 2 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -o np2.txt
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -o np4.txt
diff np2.txt np4.txt
# Expected: Files should be identical (same global result regardless of process count)
```

---

### Phase 4: MPI Parallel I/O - File Reading ✅ TEST AFTER THIS

**Goal**: Implement parallel file reading with direct padded allocation

**Tasks:**
1. Implement `mpi_read_local_padded_matrix()` with MPI_File_read_at()
2. Handle text file offset calculation (fixed-width format)
3. Test with existing test files

**New function in `src/io_mpi.c`:**
```c
float** mpi_read_local_padded_matrix(
    const char* filename,
    int* H_global, int* W_global,  // Read from file header
    int kH, int kW,
    int* padded_local_H,
    int* padded_local_W,
    int* local_start_row,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPI_File fh;
    MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    // Root reads and broadcasts dimensions
    if (rank == 0) {
        char header[256];
        MPI_File_read(fh, header, 256, MPI_CHAR, MPI_STATUS_IGNORE);
        sscanf(header, "%d %d", H_global, W_global);
    }
    MPI_Bcast(H_global, 1, MPI_INT, 0, comm);
    MPI_Bcast(W_global, 1, MPI_INT, 0, comm);

    // Calculate local dimensions
    int local_H, local_W;
    calculate_local_dimensions(rank, size, *H_global, *W_global, kH, kW,
                               &local_H, &local_W, local_start_row,
                               padded_local_H, padded_local_W);

    int pad_top, pad_bottom, pad_left, pad_right;
    calculate_padding_for_process(rank, size, kH, kW,
                                   &pad_top, &pad_bottom,
                                   &pad_left, &pad_right);

    // Allocate and initialize padded matrix
    float **matrix = allocate_matrix(*padded_local_H, *padded_local_W);
    for (int i = 0; i < *padded_local_H; i++) {
        for (int j = 0; j < *padded_local_W; j++) {
            matrix[i][j] = 0.0f;
        }
    }

    // Calculate file offsets (assuming fixed-width text: "0.xxx " = 5 chars)
    int chars_per_float = 5;
    int chars_per_row = *W_global * chars_per_float;

    // Calculate header size (find first newline)
    MPI_Offset header_size = 0;
    if (rank == 0) {
        char c;
        MPI_Offset pos = 0;
        do {
            MPI_File_read_at(fh, pos, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
            pos++;
        } while (c != '\n');
        header_size = pos;
    }
    MPI_Bcast(&header_size, 1, MPI_OFFSET, 0, comm);

    // Read local rows
    for (int i = 0; i < local_H; i++) {
        int global_row = *local_start_row + i;
        int padded_row = i + pad_top;

        MPI_Offset row_offset = header_size + global_row * (chars_per_row + 1);

        char row_buffer[10000];
        MPI_File_read_at(fh, row_offset, row_buffer, chars_per_row,
                        MPI_CHAR, MPI_STATUS_IGNORE);

        // Parse row into matrix
        char *token = strtok(row_buffer, " ");
        for (int j = 0; j < *W_global && token != NULL; j++) {
            matrix[padded_row][j + pad_left] = atof(token);
            token = strtok(NULL, " ");
        }
    }

    MPI_File_close(&fh);
    return matrix;
}
```

**TEST PHASE 4:**
```bash
# Test 1: Read existing test file
mpirun --allow-run-as-root -np 2 ./build/conv_stride_test -f f/f0.txt -g g/g0.txt -v
# Expected: Should print correct matrix values from f0.txt

# Test 2: Compare read vs generate
# First generate and save
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 20 -W 20 -kH 3 -kW 3 -f test_gen.txt
# Then read it back
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -f test_gen.txt -g g/g0.txt -v
# Expected: Should read back same values

# Test 3: Different process counts reading same file
mpirun --allow-run-as-root -np 2 ./build/conv_stride_test -f f/f1.txt -g g/g1.txt -o read_np2.txt
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -f f/f1.txt -g g/g1.txt -o read_np4.txt
diff read_np2.txt read_np4.txt
# Expected: Identical results regardless of process count
```

---

### Phase 5: Halo Exchange Implementation ✅ TEST AFTER THIS

**Goal**: Implement MPI halo exchange to fill padding regions with neighbor data

**Tasks:**
1. Review existing `mpi_exchange_halos()` function
2. Ensure it works with new padded matrix structure
3. Test halo exchange correctness

**Existing function (verify/update in `src/conv2d_mpi.c`):**
```c
void mpi_exchange_halos(float **local_matrix, int local_H, int local_W,
                       int kernel_H, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int halo_size = (kernel_H - 1) / 2;

    // Exchange with previous process
    if (rank > 0) {
        for (int i = 0; i < halo_size; i++) {
            MPI_Sendrecv(
                local_matrix[halo_size + i], local_W, MPI_FLOAT, rank - 1, 100 + i,
                local_matrix[i], local_W, MPI_FLOAT, rank - 1, 200 + i,
                comm, MPI_STATUS_IGNORE
            );
        }
    }

    // Exchange with next process
    if (rank < size - 1) {
        int data_end = local_H - halo_size - halo_size;
        for (int i = 0; i < halo_size; i++) {
            MPI_Sendrecv(
                local_matrix[data_end + i], local_W, MPI_FLOAT, rank + 1, 200 + i,
                local_matrix[local_H - halo_size + i], local_W, MPI_FLOAT, rank + 1, 100 + i,
                comm, MPI_STATUS_IGNORE
            );
        }
    }
}
```

**TEST PHASE 5:**
```bash
# Test 1: Visual verification of halo exchange
# Create test program that prints halo regions before/after exchange
mpirun --allow-run-as-root -np 3 ./test_halo_exchange -H 30 -W 10 -kH 3 -kW 3

# Expected output (example):
# Rank 0 before exchange: top halo = [zeros], bottom halo = [zeros]
# Rank 0 after exchange:  top halo = [zeros], bottom halo = [rank1's data]
# Rank 1 before exchange: top halo = [zeros], bottom halo = [zeros]
# Rank 1 after exchange:  top halo = [rank0's data], bottom halo = [rank2's data]
# Rank 2 before exchange: top halo = [zeros], bottom halo = [zeros]
# Rank 2 after exchange:  top halo = [rank1's data], bottom halo = [zeros]

# Test 2: Verify computation correctness with halos
# Compare with serial result
mpirun --allow-run-as-root -np 1 ./build/conv_stride_test -H 50 -W 50 -kH 3 -kW 3 -s -o serial.txt
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 50 -W 50 -kH 3 -kW 3 -o parallel.txt
diff serial.txt parallel.txt
# Expected: Files should be identical (correct halo exchange)
```

---

### Phase 6: MPI Parallel I/O - File Writing ✅ TEST AFTER THIS

**Goal**: Implement parallel output writing to eliminate gather overhead

**Tasks:**
1. Implement `mpi_write_output_parallel()` with MPI_File_write_at_all()
2. Handle fixed-width text format
3. Test write correctness and performance

**New function in `src/io_mpi.c`:**
```c
int mpi_write_output_parallel(
    const char* filename,
    float **local_output,
    int local_H, int local_W,
    int local_start_row,
    int global_H, int global_W,
    int kH, int kW,  // For format width calculation
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPI_File fh;
    MPI_File_open(comm, filename,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);

    // Determine output format width
    int max_output = kH * kW;
    int chars_per_float = (max_output < 10) ? 5 : (max_output < 100) ? 6 : 7;

    // Root writes header
    if (rank == 0) {
        char header[256];
        sprintf(header, "%d %d\n", global_H, global_W);
        MPI_File_write(fh, header, strlen(header), MPI_CHAR, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(comm);  // Ensure header is written

    // Calculate header size
    MPI_Offset header_size = 0;
    if (rank == 0) {
        char header[256];
        sprintf(header, "%d %d\n", global_H, global_W);
        header_size = strlen(header);
    }
    MPI_Bcast(&header_size, 1, MPI_OFFSET, 0, comm);

    // Each process writes its rows
    for (int i = 0; i < local_H; i++) {
        int global_row = local_start_row + i;

        // Format row to string
        char row_str[100000];
        int pos = 0;

        for (int j = 0; j < local_W; j++) {
            pos += sprintf(&row_str[pos], "%.3f", local_output[i][j]);
            if (j < local_W - 1) row_str[pos++] = ' ';
        }
        row_str[pos++] = '\n';
        row_str[pos] = '\0';

        // Calculate file offset
        MPI_Offset offset = header_size +
                           global_row * (global_W * chars_per_float + 1);

        // Write with collective I/O
        MPI_File_write_at_all(fh, offset, row_str, strlen(row_str),
                             MPI_CHAR, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&fh);
    return 0;
}
```

**TEST PHASE 6:**
```bash
# Test 1: Parallel write correctness
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -o par_write.txt
# Verify file format is correct (check header, check values)

# Test 2: Compare parallel write vs serial write
mpirun --allow-run-as-root -np 1 ./build/conv_stride_test -H 50 -W 50 -kH 3 -kW 3 -s -o serial_out.txt
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 50 -W 50 -kH 3 -kW 3 -o par_out.txt
diff serial_out.txt par_out.txt
# Expected: Files should be identical

# Test 3: Performance comparison
time mpirun --allow-run-as-root -np 1 ./build/conv_stride_test -H 5000 -W 5000 -kH 3 -kW 3 -o large_serial.txt
time mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 5000 -W 5000 -kH 3 -kW 3 -o large_par.txt
# Expected: Parallel should be faster for large matrices
```

---

### Phase 7: Output Gathering (For Non-File Cases) ✅ TEST AFTER THIS

**Goal**: Implement gathering for logical completeness when not writing to file

**Tasks:**
1. Implement `mpi_gather_output_to_root()`
2. Integrate into main flow with conditional logic
3. Test gathering correctness

**New function in `src/io_mpi.c`:**
```c
void mpi_gather_output_to_root(
    float **local_output,
    int local_H, int local_W,
    int local_start_row,
    float ***full_output,
    int global_H, int global_W,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        // Root allocates full output
        *full_output = allocate_matrix(global_H, global_W);

        // Copy root's own data
        for (int i = 0; i < local_H; i++) {
            for (int j = 0; j < local_W; j++) {
                (*full_output)[local_start_row + i][j] = local_output[i][j];
            }
        }

        // Receive from other processes
        for (int p = 1; p < size; p++) {
            int p_local_H, p_local_W, p_start_row;

            MPI_Recv(&p_local_H, 1, MPI_INT, p, 100, comm, MPI_STATUS_IGNORE);
            MPI_Recv(&p_local_W, 1, MPI_INT, p, 101, comm, MPI_STATUS_IGNORE);
            MPI_Recv(&p_start_row, 1, MPI_INT, p, 102, comm, MPI_STATUS_IGNORE);

            for (int i = 0; i < p_local_H; i++) {
                MPI_Recv((*full_output)[p_start_row + i], p_local_W,
                        MPI_FLOAT, p, 200 + i, comm, MPI_STATUS_IGNORE);
            }
        }
    } else {
        // Non-root sends data
        MPI_Send(&local_H, 1, MPI_INT, 0, 100, comm);
        MPI_Send(&local_W, 1, MPI_INT, 0, 101, comm);
        MPI_Send(&local_start_row, 1, MPI_INT, 0, 102, comm);

        for (int i = 0; i < local_H; i++) {
            MPI_Send(local_output[i], local_W, MPI_FLOAT, 0, 200 + i, comm);
        }
    }
}
```

**Integration in `src/main.c`:**
```c
// After computation
if (output_file) {
    // Write to file (parallel I/O)
    mpi_write_output_parallel(output_file, local_output,
                              local_output_H, local_output_W,
                              local_start_row,
                              output_H, output_W,
                              kernel_H, kernel_W,
                              active_comm);
} else {
    // No file → Gather for logical completeness
    float **full_output = NULL;

    mpi_gather_output_to_root(local_output, local_output_H, local_output_W,
                              local_start_row,
                              &full_output, output_H, output_W,
                              active_comm);

    if (rank == 0) {
        if (verbose && output_H <= 10 && output_W <= 10) {
            printf("Output (%dx%d):\n", output_H, output_W);
            print_matrix(full_output, output_H, output_W);
        } else if (verbose) {
            printf("Computation complete (%dx%d result)\n", output_H, output_W);
        }

        // Verification if needed
        if (precision > 0 && expected_file) {
            float **expected = NULL;
            int exp_H, exp_W;
            read_matrix_from_file(expected_file, &expected, &exp_H, &exp_W);

            if (exp_H == output_H && exp_W == output_W) {
                float tolerance = pow(10.0, -precision);
                if (compare_matrices(full_output, expected, output_H, output_W, tolerance)) {
                    printf("Verify Pass!\n");
                } else {
                    printf("Verify Failed!\n");
                }
            }
            free_matrix(expected, exp_H);
        }

        free_matrix(full_output, output_H);
    }
}
```

**TEST PHASE 7:**
```bash
# Test 1: Gather without file output
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 10 -W 10 -kH 3 -kW 3 -v
# Expected: Should print complete 10×10 result

# Test 2: Verify gathered result matches file output
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 50 -W 50 -kH 3 -kW 3 -o with_file.txt
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 50 -W 50 -kH 3 -kW 3 -v > no_file_output.txt
# Parse and compare the matrix from verbose output with file
# Expected: Same values

# Test 3: Verification mode
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -f f/f0.txt -g g/g0.txt -p 2 -o o/o0_expected.txt
# Expected: "Verify Pass!" message
```

---

### Phase 8: Integration & End-to-End Testing ✅ TEST AFTER THIS

**Goal**: Integrate all components and test complete workflows

**Tasks:**
1. Ensure all components work together
2. Test all CLI flag combinations
3. Verify memory usage reduction

**Complete main flow in `src/main.c`:**
```c
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse CLI arguments
    // ... (existing getopt code) ...

    // Calculate output dimensions
    int output_H, output_W;
    calculate_stride_output_dims(input_H, input_W, stride_H, stride_W,
                                 &output_H, &output_W);

    // Create active communicator
    int optimal_processes = (size > output_H) ? output_H : size;
    bool is_active_process = (rank < optimal_processes);

    MPI_Comm active_comm;
    MPI_Comm_split(MPI_COMM_WORLD,
                   is_active_process ? 0 : MPI_UNDEFINED,
                   rank, &active_comm);

    if (!is_active_process) {
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    // Get local padded input
    float **padded_input = NULL;
    int padded_H, padded_W, local_start_row;

    if (has_generation) {
        padded_input = mpi_generate_local_padded_matrix(
            input_H, input_W, kernel_H, kernel_W,
            &padded_H, &padded_W, &local_start_row,
            0.0f, 1.0f, active_comm
        );
    } else {
        padded_input = mpi_read_local_padded_matrix(
            input_file, &input_H, &input_W,
            kernel_H, kernel_W,
            &padded_H, &padded_W, &local_start_row,
            active_comm
        );
    }

    // Kernel handling
    float **kernel = NULL;
    if (rank == 0) {
        if (has_generation) {
            kernel = generate_random_matrix(kernel_H, kernel_W, 0.0f, 1.0f);
        } else {
            read_matrix_from_file(kernel_file, &kernel, &kernel_H, &kernel_W);
        }
    }
    mpi_broadcast_kernel(&kernel, kernel_H, kernel_W, active_comm);

    // Halo exchange
    mpi_exchange_halos(padded_input, padded_H, padded_W, kernel_H, active_comm);

    // Allocate local output
    int local_output_H = (padded_H - kernel_H + 1 + stride_H - 1) / stride_H;
    int local_output_W = (padded_W - kernel_W + 1 + stride_W - 1) / stride_W;
    float **local_output = allocate_matrix(local_output_H, local_output_W);

    // Computation
    if (timing) mpi_timer_start(&timer);

    if (use_serial && rank == 0) {
        conv2d_stride_serial(padded_input, padded_H, padded_W,
                            kernel, kernel_H, kernel_W,
                            stride_H, stride_W, local_output);
    } else if (use_mpi_only) {
        conv2d_stride_mpi(padded_input, padded_H, padded_W,
                         kernel, kernel_H, kernel_W,
                         stride_H, stride_W, local_output, active_comm);
    } else if (use_openmp_only && rank == 0) {
        conv2d_stride_openmp(padded_input, padded_H, padded_W,
                            kernel, kernel_H, kernel_W,
                            stride_H, stride_W, local_output);
    } else {
        conv2d_stride_hybrid(padded_input, padded_H, padded_W,
                            kernel, kernel_H, kernel_W,
                            stride_H, stride_W, local_output, active_comm);
    }

    if (timing) {
        mpi_timer_end(&timer, active_comm);
        if (rank == 0) print_timing(&timer);
    }

    // Output handling
    if (output_file) {
        mpi_write_output_parallel(output_file, local_output,
                                  local_output_H, local_output_W,
                                  local_start_row,
                                  output_H, output_W,
                                  kernel_H, kernel_W,
                                  active_comm);
    } else {
        float **full_output = NULL;
        mpi_gather_output_to_root(local_output, local_output_H, local_output_W,
                                  local_start_row,
                                  &full_output, output_H, output_W,
                                  active_comm);

        if (rank == 0) {
            if (verbose) {
                if (output_H <= 10 && output_W <= 10) {
                    print_matrix(full_output, output_H, output_W);
                } else {
                    printf("Result computed (%dx%d)\n", output_H, output_W);
                }
            }
            free_matrix(full_output, output_H);
        }
    }

    // Cleanup
    free_matrix(padded_input, padded_H);
    free_matrix(local_output, local_output_H);
    if (rank == 0) free_matrix(kernel, kernel_H);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
```

**TEST PHASE 8 (Comprehensive):**
```bash
# Test Suite 1: Generation modes
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -t
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -o gen_out.txt -t
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -v

# Test Suite 2: File reading modes
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -f f/f0.txt -g g/g0.txt -t
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -f f/f1.txt -g g/g1.txt -o read_out.txt -t
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -f f/f2.txt -g g/g2.txt -v

# Test Suite 3: Stride variations
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -sH 2 -sW 2 -o stride_22.txt
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -sH 3 -sW 2 -o stride_32.txt

# Test Suite 4: Implementation variants
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -s -o serial.txt
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -m -o mpi_only.txt
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -P -o openmp.txt
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -o hybrid.txt

# Verify all produce same result
diff serial.txt mpi_only.txt && diff mpi_only.txt openmp.txt && diff openmp.txt hybrid.txt
# Expected: All identical

# Test Suite 5: Process count variations
for np in 1 2 4 8; do
    mpirun --allow-run-as-root -np $np ./build/conv_stride_test -H 200 -W 200 -kH 3 -kW 3 -o np${np}.txt
done
# Compare all outputs
diff np1.txt np2.txt && diff np2.txt np4.txt && diff np4.txt np8.txt
# Expected: All identical

# Test Suite 6: Existing test cases verification
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -f f/f0.txt -g g/g0.txt -p 2 -o o/o0_sH_1_sW_1.txt
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -f f/f1.txt -g g/g1.txt -sH 3 -sW 2 -p 2 -o o/o1_sH_3_sW_2.txt
# Expected: "Verify Pass!" for both
```

---

### Phase 9: Performance Measurement & Optimization ✅ TEST AFTER THIS

**Goal**: Measure memory usage and performance improvements

**Tasks:**
1. Add memory profiling
2. Compare old vs new approach
3. Document performance gains

**Memory profiling script (`test_memory.sh`):**
```bash
#!/bin/bash

echo "=== Memory Usage Comparison ==="
echo ""

# Test matrix sizes
SIZES=(1000 5000 10000)

for SIZE in "${SIZES[@]}"; do
    echo "Matrix size: ${SIZE}×${SIZE}"
    echo "-----------------------------------"

    # Old approach (if still available)
    echo "Old approach (with gather):"
    /usr/bin/time -v mpirun --allow-run-as-root -np 4 ./build/conv_stride_test_old \
        -H $SIZE -W $SIZE -kH 3 -kW 3 -o old_${SIZE}.txt 2>&1 | grep "Maximum resident"

    # New approach
    echo "New approach (parallel I/O):"
    /usr/bin/time -v mpirun --allow-run-as-root -np 4 ./build/conv_stride_test \
        -H $SIZE -W $SIZE -kH 3 -kW 3 -o new_${SIZE}.txt 2>&1 | grep "Maximum resident"

    echo ""
done
```

**TEST PHASE 9:**
```bash
# Run memory profiling
chmod +x test_memory.sh
./test_memory.sh

# Expected results (example):
# Matrix 1000×1000:
#   Old: Maximum resident set size: 45MB
#   New: Maximum resident set size: 18MB (60% reduction)
#
# Matrix 5000×5000:
#   Old: Maximum resident set size: 1.2GB
#   New: Maximum resident set size: 480MB (60% reduction)
#
# Matrix 10000×10000:
#   Old: Maximum resident set size: 4.8GB
#   New: Maximum resident set size: 1.9GB (60% reduction)

# Performance comparison
echo "=== Performance Comparison ==="
for np in 1 2 4 8; do
    echo "Processes: $np"
    time mpirun --allow-run-as-root -np $np ./build/conv_stride_test \
        -H 5000 -W 5000 -kH 3 -kW 3 -o perf_${np}.txt
done

# Expected: Near-linear speedup up to number of cores
```

---

### Phase 10: Documentation & Code Cleanup ✅ FINAL REVIEW

**Goal**: Clean up code, add documentation, prepare for submission

**Tasks:**
1. Add function documentation (Doxygen style)
2. Remove deprecated code
3. Update README if exists
4. Final testing on all provided test cases

**Documentation example:**
```c
/**
 * @brief Generate local portion of padded matrix for distributed computation
 *
 * This function generates a local portion of a matrix with padding, suitable for
 * distributed 2D convolution. Each process generates only its assigned rows plus
 * necessary padding. Uses deterministic position-based seeding for reproducibility.
 *
 * @param H_global Global matrix height (without padding)
 * @param W_global Global matrix width (without padding)
 * @param kH Kernel height (for padding calculation)
 * @param kW Kernel width (for padding calculation)
 * @param[out] padded_local_H Returned local matrix height (with padding)
 * @param[out] padded_local_W Returned local matrix width (with padding)
 * @param[out] local_start_row Starting row in global coordinates
 * @param min_val Minimum random value (inclusive)
 * @param max_val Maximum random value (exclusive)
 * @param comm MPI communicator (typically active_comm)
 *
 * @return Allocated padded matrix with generated values
 *
 * @note Padding regions are initialized to zero. Use mpi_exchange_halos()
 *       to fill with actual neighbor data.
 * @note Uses deterministic seeding: same global matrix regardless of process count
 */
float** mpi_generate_local_padded_matrix(
    int H_global, int W_global,
    int kH, int kW,
    int* padded_local_H,
    int* padded_local_W,
    int* local_start_row,
    float min_val, float max_val,
    MPI_Comm comm
);
```

**Final cleanup checklist:**
- [ ] Remove old `mpi_distribute_matrix()` if not used
- [ ] Remove old `mpi_gather_output()` if replaced
- [ ] Remove `generate_padded_matrix()` for global matrices
- [ ] Update all function comments
- [ ] Ensure consistent code style
- [ ] Remove debug print statements
- [ ] Check for memory leaks with valgrind (if available)

**FINAL TEST SUITE:**
```bash
#!/bin/bash
# final_test_suite.sh

echo "=== Final Comprehensive Test Suite ==="

# All provided test cases
test_cases=(
    "-f f/f0.txt -g g/g0.txt -o o/o0_sH_1_sW_1.txt -p 2"
    "-f f/f1.txt -g g/g1.txt -sH 3 -sW 2 -o o/o1_sH_3_sW_2.txt -p 2"
    "-f f/f2.txt -g g/g2.txt -o o/o2_sH_1_sW_1.txt -p 2"
    "-f f/f3.txt -g g/g3.txt -sH 2 -sW 3 -o o/o3_sH_2_sW_3.txt -p 2"
)

for test in "${test_cases[@]}"; do
    echo "Running: mpirun -np 4 ./build/conv_stride_test $test"
    mpirun --allow-run-as-root -np 4 ./build/conv_stride_test $test
    echo ""
done

# Large matrix stress test
echo "=== Large Matrix Stress Test ==="
mpirun --allow-run-as-root -np 4 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -t -o large_test.txt

echo "=== All tests complete ==="
```

---

## Testing Strategy

### Incremental Testing Approach

Each phase ends with a **TEST AFTER THIS** checkpoint:

1. **Phase 1-2**: Infrastructure testing (sub-communicator, padding calculation)
2. **Phase 3**: Generation correctness (determinism, value range)
3. **Phase 4**: File reading (parallel I/O read correctness)
4. **Phase 5**: Halo exchange (boundary data correctness)
5. **Phase 6**: File writing (parallel I/O write correctness)
6. **Phase 7**: Gathering (logical completeness)
7. **Phase 8**: Integration (end-to-end workflows)
8. **Phase 9**: Performance (memory & speed measurements)
9. **Phase 10**: Final validation (all test cases)

### Test Categories

**Unit Tests** (Phases 1-7):
- Single component functionality
- Isolated from other components
- Quick feedback loop

**Integration Tests** (Phase 8):
- Multiple components together
- Real workflow scenarios
- CLI flag combinations

**Performance Tests** (Phase 9):
- Memory usage profiling
- Execution time measurement
- Scalability analysis

**Regression Tests** (Phase 10):
- All provided test cases
- Verification against expected outputs
- Cross-version compatibility

---

## API Reference

### Core Functions

#### Input Acquisition
```c
float** mpi_generate_local_padded_matrix(
    int H_global, int W_global, int kH, int kW,
    int* padded_local_H, int* padded_local_W, int* local_start_row,
    float min_val, float max_val, MPI_Comm comm
);

float** mpi_read_local_padded_matrix(
    const char* filename, int* H_global, int* W_global,
    int kH, int kW,
    int* padded_local_H, int* padded_local_W, int* local_start_row,
    MPI_Comm comm
);
```

#### Padding & Dimensions
```c
void calculate_padding_for_process(
    int rank, int size, int kH, int kW,
    int* pad_top, int* pad_bottom, int* pad_left, int* pad_right
);

void calculate_local_dimensions(
    int rank, int size, int H_global, int W_global, int kH, int kW,
    int* local_H, int* local_W, int* local_start_row,
    int* padded_local_H, int* padded_local_W
);
```

#### Communication
```c
void mpi_broadcast_kernel(float ***kernel, int kH, int kW, MPI_Comm comm);
void mpi_exchange_halos(float **matrix, int H, int W, int kH, MPI_Comm comm);
```

#### Output
```c
int mpi_write_output_parallel(
    const char* filename,
    float **local_output, int local_H, int local_W, int local_start_row,
    int global_H, int global_W, int kH, int kW,
    MPI_Comm comm
);

void mpi_gather_output_to_root(
    float **local_output, int local_H, int local_W, int local_start_row,
    float ***full_output, int global_H, int global_W,
    MPI_Comm comm
);
```

---

## Future Optimizations

### Potential Enhancements (Beyond Current Scope)

1. **Binary File Format**
   - Fixed-size records for perfect offset calculation
   - Even faster parallel I/O
   - Smaller file sizes

2. **Asynchronous I/O**
   - Overlap computation with I/O
   - Use `MPI_File_iwrite_at_all()` for non-blocking writes

3. **Advanced Load Balancing**
   - Stride-aware distribution (already partially implemented)
   - Dynamic load balancing for irregular workloads

4. **Memory-Mapped I/O**
   - For very large files that exceed available RAM
   - Use `MPI_File_set_view()` with memory mapping

5. **Compression**
   - On-the-fly compression for output files
   - Trade CPU for I/O bandwidth

6. **Hybrid File Systems**
   - SSD for temporary data
   - Parallel file system for final output

---

## Summary

### What This Plan Achieves

✅ **Eliminates memory waste**: No more 5× matrix allocation on root
✅ **Enables large matrices**: 50000×50000 now fits in memory
✅ **Leverages MPI Parallel I/O**: True distributed file access
✅ **Maintains correctness**: All test cases pass
✅ **Improves performance**: Faster I/O, better scalability
✅ **Ensures logical completeness**: Always have final result
✅ **Incremental development**: Test after each phase

### Memory Savings Summary

| Matrix Size | Old Approach | New Approach | Savings |
|-------------|--------------|--------------|---------|
| 1000×1000   | 20MB         | 8MB          | 60%     |
| 10000×10000 | 2GB          | 800MB        | 60%     |
| 50000×50000 | 50GB ❌      | 12.5GB ✅    | 75%     |

### Key Takeaways for Implementation

1. **Start with infrastructure** (Phase 1): Get communicators right first
2. **Test incrementally**: Never proceed without testing current phase
3. **Maintain backward compatibility**: Keep existing test cases passing
4. **Document as you go**: Future you will thank you
5. **Profile performance**: Measure actual improvements, not assumptions

---

**End of Implementation Plan**

*This document is a living plan - update as implementation progresses and new insights emerge.*
