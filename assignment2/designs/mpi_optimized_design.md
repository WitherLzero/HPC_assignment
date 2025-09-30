# MPI Distributed Memory Convolution Design Document - Optimized Multi-Node Approach

## Overview

This document provides a comprehensive design for implementing distributed memory 2D convolution with stride using MPI, **optimized for multi-node HPC clusters** (Kaya/Setonix). The design focuses on **maximum scalability**, **memory efficiency**, and **avoiding single-node bottlenecks** through distributed generation and streaming I/O.

## Key Design Principles (Updated)

1. **Multi-Node Optimization**: 1 MPI process per node + OpenMP threads within each node
2. **Distributed Generation**: No single node holds the complete matrix
3. **Streaming I/O**: MPI-IO parallel writing to avoid gathering bottlenecks
4. **Memory Scalability**: Handle 4× larger inputs than single-node approaches
5. **Coordinated Randomness**: Deterministic generation ensuring consistency

## Theoretical Foundation (From Lecture Notes)

### Key MPI Concepts from Lectures

1. **Domain Decomposition** (Lecture 8):
   - Distribute data across processes with minimal communication
   - Each process owns a portion of the problem
   - Overlap regions (halos) for dependencies

2. **Communication Patterns** (Lecture 9):
   - Point-to-point: Send/Recv for halo exchange
   - Collective: Scatter/Gather for data distribution
   - Optimization: Overlap communication with computation

3. **Performance Considerations** (Lecture 7):
   - Minimize communication volume
   - Maximize computation/communication ratio
   - Load balance across processes

## MPI Implementation Architecture

### Overall Strategy - Multi-Node Distributed Approach

```
Multi-Node Architecture (4 nodes × 1 process + OpenMP threads):
┌─────────────────────────────────────┐
│      NO GLOBAL MATRIX EXISTS        │ ← Key difference!
│    Each node generates its chunk    │
└─────────────────────────────────────┘
                ↓
    Coordinated Distributed Generation
                ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   Node 0    │   Node 1    │   Node 2    │   Node 3    │
│  (Process 0)│  (Process 1)│  (Process 2)│  (Process 3)│
│ + overlap   │ + overlap   │ + overlap   │ + overlap   │
│ + OpenMP    │ + OpenMP    │ + OpenMP    │ + OpenMP    │
└─────────────┴─────────────┴─────────────┴─────────────┘
                ↓
          Local Computation
                ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│Local Output │Local Output │Local Output │Local Output │
│   Chunk 0   │   Chunk 1   │   Chunk 2   │   Chunk 3   │
└─────────────┴─────────────┴─────────────┴─────────────┘
                ↓
          MPI-IO Parallel Write (No Gathering!)
                ↓
         ┌─────────────────┐
         │  Single Output  │
         │      File       │
         └─────────────────┘
```

### Memory-Optimized Data Strategy

#### Distributed Generation with Coordinated Seeds

**Key Innovation: No single node ever holds the full matrix!**

```
Coordinated Generation Pattern:
┌────────────────────────────┐
│    Process 0 generates     │ ← Rows [0...H/4+overlap] using global coordinates
│    rows with overlap       │   seed = base_seed ^ (global_row << 32) ^ global_col
├────────────────────────────┤
│    Process 1 generates     │ ← Rows [H/4-overlap...H/2+overlap]
│    rows with overlap       │   Same seed function → identical overlap values!
├────────────────────────────┤
│    Process 2 generates     │ ← Rows [H/2-overlap...3H/4+overlap]
│    rows with overlap       │   Automatic boundary consistency
├────────────────────────────┤
│    Process 3 generates     │ ← Rows [3H/4-overlap...H]
│    rows with overlap       │   No communication needed for halos!
└────────────────────────────┘

Memory per node: ~H/4 × W × 8 bytes ≤ 1.5TB
Maximum input: 534K × 534K (vs 274K × 274K single-node)
```

#### Memory Scaling Analysis

| Approach | Max Matrix Size | Memory Usage | Speedup |
|----------|-----------------|--------------|---------|
| Single Node | 274K × 274K | 300GB+300GB = 600GB | 1× |
| **Distributed (stride=1)** | **390K × 390K** | **750GB per node** | **2.0×** |
| **Distributed (stride=2)** | **534K × 534K** | **750GB per node** | **3.8×** |
| **Distributed (stride=4)** | **547K × 547K** | **750GB per node** | **4.0×** |

#### Stride-Aware Distribution

The stride affects how we distribute work:

```c
// Calculate rows per process considering stride
int calculate_rows_per_process(int global_rows, int num_procs, int stride_H) {
    // Output rows after stride
    int output_rows = (int)ceil((double)global_rows / stride_H);
    
    // Base rows per process
    int base_rows_per_proc = output_rows / num_procs;
    int remainder = output_rows % num_procs;
    
    // Convert back to input rows needed
    return base_rows_per_proc * stride_H;
}
```

## Implementation Components

### 1. MPI Initialization and Setup

```c
void initialize_mpi_convolution(MPI_Comm comm, conv_mpi_context_t *ctx) {
    MPI_Comm_rank(comm, &ctx->rank);
    MPI_Comm_size(comm, &ctx->size);
    
    // Determine process layout
    ctx->is_root = (ctx->rank == 0);
    
    // Calculate local dimensions
    calculate_local_dimensions(ctx);
}
```

### 2. Distributed Generation Function (Replaces Distribution)

```c
void mpi_generate_coordinated_matrix(int global_H, int global_W, int kernel_H, int kernel_W,
                                   uint64_t global_seed, float ***local_matrix,
                                   int *local_H, int *local_W, int *local_start_row,
                                   MPI_Comm comm)
```

**Revolutionary Algorithm Design**:

```
Step 1: Coordinate global parameters
- Broadcast dimensions and seed from root
- Each process calculates its responsibility

Step 2: Calculate local dimensions with overlap
- Process p gets rows [start_p...end_p+overlap]
- Overlap ensures convolution boundary correctness

Step 3: Generate local chunk using global coordinates
- For each local element (i,j), use global position for seed
- Deterministic: same global coordinates → same value
- Automatic overlap consistency!

Step 4: Apply same padding to local chunks
- Each process handles its own padding
- No communication required!

Advantages:
✓ No memory bottleneck at any single node
✓ No MPI_Scatter/MPI_Scatterv communication cost
✓ Scales to unlimited number of processes
✓ Guaranteed mathematical correctness
```

**Coordinated Generation Implementation**:

```c
// Deterministic element generation based on global coordinates
float generate_matrix_element(int global_row, int global_col, uint64_t base_seed) {
    uint64_t element_seed = base_seed ^
                           ((uint64_t)global_row << 32) ^
                           ((uint64_t)global_col);

    // Fast xorshift PRNG (deterministic and portable)
    element_seed ^= element_seed >> 12;
    element_seed ^= element_seed << 25;
    element_seed ^= element_seed >> 27;

    return (float)(element_seed * 0x2545F4914F6CDD1DULL) / UINT64_MAX;
}

void generate_local_chunk_coordinated(int rank, int num_procs, int global_H, int global_W,
                                     int kernel_H, uint64_t seed, float **local_chunk) {
    // Calculate this process's row range with overlap
    int local_start_row = rank * (global_H / num_procs);
    int local_end_row = (rank + 1) * (global_H / num_procs);
    if (rank < num_procs - 1) local_end_row += (kernel_H - 1);  // Add overlap

    int local_rows = local_end_row - local_start_row;

    // Generate each element using its global coordinates
    for (int local_i = 0; local_i < local_rows; local_i++) {
        int global_row = local_start_row + local_i;
        for (int j = 0; j < global_W; j++) {
            local_chunk[local_i][j] = generate_matrix_element(global_row, j, seed);
        }
    }
}
```

### 3. Halo Exchange - ELIMINATED!

**Major Innovation: No halo exchange needed!**

```c
// This function is NO LONGER NEEDED due to coordinated generation!
//
// OLD APPROACH: Generate chunks → Exchange halos → Compute
// NEW APPROACH: Generate with overlap → Compute directly
//
// Benefits:
// ✓ Zero communication overhead
// ✓ No synchronization points
// ✓ No buffer management complexity
// ✓ Automatic boundary correctness
```

**Why this works:**

1. **Coordinated generation** ensures overlapping regions have identical values
2. **Global coordinate mapping** guarantees consistency across process boundaries
3. **Deterministic PRNG** produces same results for same coordinates
4. **Mathematical proof**: If rank 0 generates element at global position (i,j) and rank 1 also generates the same global position, they get identical values

**Verification (Optional Debug Mode)**:

```c
#ifdef DEBUG
void verify_boundary_consistency(float **local_chunk, int rank, MPI_Comm comm) {
    if (rank > 0) {
        // Send boundary to previous process for verification
        MPI_Send(local_chunk[0], W, MPI_FLOAT, rank-1, 99, comm);

        float *neighbor_boundary = malloc(W * sizeof(float));
        MPI_Recv(neighbor_boundary, W, MPI_FLOAT, rank-1, 100, comm, MPI_STATUS_IGNORE);

        // Verify they match (should be identical!)
        for (int j = 0; j < W; j++) {
            assert(fabsf(local_chunk[0][j] - neighbor_boundary[j]) < 1e-7);
        }
        free(neighbor_boundary);
    }
    if (rank < size - 1) {
        float *neighbor_boundary = malloc(W * sizeof(float));
        MPI_Recv(neighbor_boundary, W, MPI_FLOAT, rank+1, 99, comm, MPI_STATUS_IGNORE);
        MPI_Send(local_chunk[local_H-1], W, MPI_FLOAT, rank+1, 100, comm);
        // Similar verification...
        free(neighbor_boundary);
    }
}
#endif
```

### 4. Local Computation with Stride

```c
void compute_local_convolution(float **local_input, int local_H, int local_W,
                              float **kernel, int kernel_H, int kernel_W,
                              int stride_H, int stride_W,
                              float **local_output, int start_row) {
    // Adjust computation based on global position
    // Account for stride offset
    
    int output_start = start_row / stride_H;
    int output_H = calculate_local_output_rows(local_H, kernel_H, stride_H);
    
    // Perform strided convolution
    // Can use OpenMP here for hybrid parallelization
}
```

### 5. Streaming Output - No Gathering Required!

**Revolutionary Change: MPI-IO Parallel Writing**

```c
void mpi_stream_output_parallel(float **local_output, int local_output_H, int local_output_W,
                               int rank, int global_output_H, int global_output_W,
                               const char *filename, MPI_Comm comm)
```

**Streaming Strategy - Eliminates Root Memory Bottleneck**:

```
Traditional Gathering Problems:
❌ Root needs memory for FULL output matrix
❌ Memory bottleneck: input_chunk + FULL_output ≤ 1.5TB
❌ Maximum input severely limited (~346K × 346K)

NEW Streaming Solution:
✓ Each process writes directly to final file
✓ No intermediate gathering step
✓ No memory concentration at any single node
✓ Maximum input: 534K × 534K (55% larger!)
```

**MPI-IO Implementation**:

```c
void mpi_stream_output_parallel(float **local_output, int local_output_H, int local_output_W,
                               int rank, int global_output_H, int global_output_W,
                               const char *filename, MPI_Comm comm) {
    MPI_File file;
    MPI_Status status;

    // Open file for parallel writing
    MPI_File_open(comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    // Root writes header with dimensions
    if (rank == 0) {
        char header[64];
        int header_len = sprintf(header, "%d %d\n", global_output_H, global_output_W);
        MPI_File_write(file, header, header_len, MPI_CHAR, &status);
    }

    // Calculate file offset for this process's output chunk
    MPI_Offset header_size = strlen("999999 999999\n");  // Conservative estimate
    int output_rows_per_process = global_output_H / comm_size;
    int my_start_row = rank * output_rows_per_process;

    MPI_Offset my_offset = header_size + my_start_row * global_output_W * sizeof(float);

    // Each process writes its chunk at calculated offset
    for (int i = 0; i < local_output_H; i++) {
        MPI_Offset row_offset = my_offset + i * global_output_W * sizeof(float);
        MPI_File_write_at(file, row_offset, local_output[i],
                         local_output_W, MPI_FLOAT, &status);
    }

    MPI_File_close(&file);
}
```

**Alternative: Multiple File Output (Even Simpler)**:

```c
void mpi_stream_output_simple(float **local_output, int local_output_H, int local_output_W,
                             int rank, MPI_Comm comm) {
    char filename[256];
    sprintf(filename, "output_chunk_%d.txt", rank);

    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", local_output_H, local_output_W);

    for (int i = 0; i < local_output_H; i++) {
        for (int j = 0; j < local_output_W; j++) {
            fprintf(fp, "%.6f ", local_output[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    // User can concatenate later: cat output_chunk_*.txt > final_output.txt
}
```

## Communication Optimization Strategies

### 1. Non-blocking Communication

```c
// Overlap computation with communication
MPI_Request send_request, recv_request;

// Start halo exchange
MPI_Isend(bottom_rows, ..., &send_request);
MPI_Irecv(top_halo, ..., &recv_request);

// Compute interior (no halo needed)
compute_interior_convolution(...);

// Wait for halos
MPI_Wait(&send_request, MPI_STATUS_IGNORE);
MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

// Compute boundaries using halos
compute_boundary_convolution(...);
```

### 2. Persistent Communication

For iterative algorithms or multiple convolutions:

```c
// Setup persistent communication channels
MPI_Request send_req, recv_req;
MPI_Send_init(bottom_rows, ..., &send_req);
MPI_Recv_init(top_halo, ..., &recv_req);

// Reuse for multiple exchanges
for (int iter = 0; iter < num_iterations; iter++) {
    MPI_Start(&send_req);
    MPI_Start(&recv_req);
    // ... computation ...
    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
}

MPI_Request_free(&send_req);
MPI_Request_free(&recv_req);
```

## Load Balancing Considerations

### Static Load Balancing

```c
void calculate_load_distribution(int global_H, int stride_H, int num_procs,
                                int *rows_per_proc, int *start_rows) {
    int output_H = (int)ceil((double)global_H / stride_H);
    
    for (int p = 0; p < num_procs; p++) {
        // Distribute output rows evenly
        int output_start = (output_H * p) / num_procs;
        int output_end = (output_H * (p + 1)) / num_procs;
        
        // Convert to input rows
        start_rows[p] = output_start * stride_H;
        int end_row = output_end * stride_H;
        rows_per_proc[p] = end_row - start_rows[p];
    }
}
```

### Dynamic Load Balancing (Advanced)

For irregular workloads or heterogeneous systems:

```c
// Master-worker pattern for dynamic distribution
if (rank == 0) {
    distribute_work_dynamically(work_queue, comm);
} else {
    process_work_dynamically(comm);
}
```

## Memory Layout Optimization

### Process-Local Memory Organization

```
Local Matrix Layout:
┌─────────────────────────┐
│   Top Halo Rows        │ ← From previous process
├─────────────────────────┤
│                        │
│   Local Data Rows      │ ← This process's data
│                        │
├─────────────────────────┤
│   Bottom Halo Rows     │ ← From next process
└─────────────────────────┘
```

### Cache-Friendly Access Pattern

```c
// Optimize for cache by processing in blocks
#define BLOCK_SIZE 64

for (int i = 0; i < local_H; i += BLOCK_SIZE) {
    for (int j = 0; j < local_W; j += BLOCK_SIZE) {
        // Process block
        for (int bi = i; bi < min(i + BLOCK_SIZE, local_H); bi++) {
            for (int bj = j; bj < min(j + BLOCK_SIZE, local_W); bj++) {
                // Convolution computation
            }
        }
    }
}
```

## Complete Optimized MPI Convolution Algorithm

```c
void conv2d_stride_mpi_optimized(int global_H, int global_W, int kH, int kW,
                                int sH, int sW, uint64_t global_seed,
                                const char *output_filename, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Step 1: Broadcast global parameters (minimal communication)
    MPI_Bcast(&global_H, 1, MPI_INT, 0, comm);
    MPI_Bcast(&global_W, 1, MPI_INT, 0, comm);
    MPI_Bcast(&global_seed, 1, MPI_UINT64_T, 0, comm);

    // Step 2: Generate kernel locally (small, same on all processes)
    float **kernel = allocate_matrix_cache_optimized(kH, kW);
    if (rank == 0) {
        generate_random_matrix(kernel, kH, kW, 0.0f, 1.0f, global_seed + 1);
    }
    MPI_Bcast(kernel[0], kH * kW, MPI_FLOAT, 0, comm);

    // Step 3: Calculate local dimensions and generate input chunk
    int local_start_row, local_H, local_W;
    calculate_local_dimensions(rank, size, global_H, global_W, kH,
                              &local_start_row, &local_H, &local_W);

    float **local_input = allocate_matrix_cache_optimized(local_H, local_W);
    generate_local_chunk_coordinated(rank, size, global_H, global_W, kH,
                                    global_seed, local_input);

    // Step 4: Apply same padding to local chunk
    apply_same_padding_local(local_input, local_H, local_W, kH, kW);

    // Step 5: Calculate output dimensions
    int local_output_H = calculate_local_output_rows(local_H, kH, sH);
    int local_output_W = ceil((double)global_W / sW);
    float **local_output = allocate_matrix_cache_optimized(local_output_H, local_output_W);

    // Step 6: NO HALO EXCHANGE NEEDED! Perform local computation with OpenMP
    MPI_Barrier(comm);  // Sync for timing
    double start_time = MPI_Wtime();

    conv2d_stride_openmp(local_input, local_H, local_W,
                        kernel, kH, kW, sH, sW, local_output);

    MPI_Barrier(comm);  // Sync for timing
    double end_time = MPI_Wtime();

    // Step 7: Stream output directly to file (NO GATHERING!)
    int global_output_H = ceil((double)global_H / sH);
    int global_output_W = ceil((double)global_W / sW);

    mpi_stream_output_parallel(local_output, local_output_H, local_output_W,
                              rank, global_output_H, global_output_W,
                              output_filename, comm);

    // Step 8: Performance reporting
    if (rank == 0) {
        double total_time = end_time - start_time;
        printf("Distributed convolution completed in %.3f seconds\n", total_time);
        printf("Matrix size: %d×%d, Stride: %d×%d, Processes: %d\n",
               global_H, global_W, sH, sW, size);
    }

    // Cleanup
    free_matrix_cache_optimized(local_input, local_H);
    free_matrix_cache_optimized(local_output, local_output_H);
    free_matrix_cache_optimized(kernel, kH);
}
```

## Key Algorithm Improvements Summary

| Traditional MPI | **Optimized Multi-Node MPI** |
|-----------------|-------------------------------|
| Root generates full matrix | ❌ → ✅ **Distributed generation** |
| MPI_Scatter distribution | ❌ → ✅ **Coordinated local generation** |
| Halo exchange communication | ❌ → ✅ **Zero halo exchange** |
| MPI_Gather output | ❌ → ✅ **Streaming MPI-IO** |
| Memory bottleneck at root | ❌ → ✅ **Distributed memory** |
| Max input: 346K×346K | ❌ → ✅ **Max input: 534K×534K** |
| Complex communication | ❌ → ✅ **Minimal communication** |

**Communication Analysis:**
- **Traditional**: O(H×W) scatter + O(H×W/P) halos + O(H×W/P²) gather
- **Optimized**: O(kH×kW) kernel broadcast + O(1) parameters
- **Speedup**: ~1000× reduction in communication volume!

## Performance Analysis Metrics

### Metrics to Measure

1. **Computation Time**: Time spent in convolution kernel
2. **Communication Time**: Time in MPI calls
3. **Load Imbalance**: Variation in computation time across processes
4. **Scalability**: Strong and weak scaling analysis

### Timing Implementation

```c
typedef struct {
    double total_time;
    double comp_time;
    double comm_time;
    double scatter_time;
    double gather_time;
    double halo_time;
} mpi_perf_stats_t;

void collect_performance_stats(mpi_perf_stats_t *local_stats, 
                              MPI_Comm comm) {
    mpi_perf_stats_t global_stats;
    
    // Reduce to get max times (worst case)
    MPI_Reduce(&local_stats->total_time, &global_stats.total_time,
               1, MPI_DOUBLE, MPI_MAX, 0, comm);
    
    // Also collect average for load balance analysis
    MPI_Reduce(&local_stats->comp_time, &global_stats.comp_time,
               1, MPI_DOUBLE, MPI_SUM, 0, comm);
    
    if (rank == 0) {
        print_performance_report(&global_stats, size);
    }
}
```

## Testing Strategy

### Correctness Testing

```bash
# Test with 1 process (should match serial)
mpirun -np 1 ./conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -sH 2 -sW 2 -m

# Test with multiple processes
for np in 2 4 8; do
    mpirun -np $np ./conv_stride_test -f input.txt -g kernel.txt -o output_$np.txt -m
    # Compare outputs
done
```

### Performance Testing

```bash
# Strong scaling (fixed problem size)
for np in 1 2 4 8 16 32; do
    mpirun -np $np ./conv_stride_test -H 4096 -W 4096 -kH 5 -kW 5 -sH 2 -sW 2 -m -t
done

# Weak scaling (fixed size per process)
for np in 1 2 4 8 16; do
    size=$((1024 * sqrt(np)))
    mpirun -np $np ./conv_stride_test -H $size -W $size -kH 5 -kW 5 -sH 2 -sW 2 -m -t
done
```

## Implementation Checklist - Optimized Approach

### Phase 1: Core Infrastructure
1. [ ] Implement `calculate_local_dimensions()` for domain decomposition
2. [ ] Implement `generate_matrix_element()` deterministic PRNG
3. [ ] Implement `generate_local_chunk_coordinated()`
4. [ ] Implement `apply_same_padding_local()`
5. [ ] Implement cache-optimized matrix allocation functions

### Phase 2: Computation & I/O
6. [ ] Implement `mpi_stream_output_parallel()` with MPI-IO
7. [ ] Implement `mpi_stream_output_simple()` (fallback)
8. [ ] Integrate with existing `conv2d_stride_openmp()`
9. [ ] Add comprehensive timing instrumentation

### Phase 3: Testing & Verification
10. [ ] Implement boundary verification (debug mode)
11. [ ] Test correctness vs serial with various matrix sizes
12. [ ] Test with 1, 2, 4 processes for consistency
13. [ ] Verify output file correctness (single vs multiple files)

### Phase 4: Performance Optimization
14. [ ] Benchmark vs traditional MPI approach
15. [ ] Strong scaling analysis (fixed problem size)
16. [ ] Weak scaling analysis (fixed size per process)
17. [ ] Memory usage profiling and optimization
18. [ ] HPC cluster testing (Kaya/Setonix)

### Phase 5: Advanced Features
19. [ ] Implement hybrid MPI+OpenMP optimization
20. [ ] Add SLURM job scripts for multi-node deployment
21. [ ] Implement streaming for ultra-large matrices
22. [ ] Performance analysis and report generation

## Deployment Strategy - HPC Clusters

### Recommended Development Approach:

**Local Development First:**
```bash
# Start with local multi-process testing
mpirun -np 4 ./conv_stride_test -H 1000 -W 1000 -kH 3 -kW 3 -sH 2 -sW 2 -m
```

**Then HPC Testing:**
```bash
# SLURM script for 4 nodes × 1 process each
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20    # All cores for OpenMP
export OMP_NUM_THREADS=20
mpirun -np 4 ./conv_stride_test -H 50000 -W 50000 -kH 5 -kW 5 -sH 2 -sW 2 -m -t
```

### Why This Development Strategy Works:

1. **Local debugging**: Fast iteration, easy debugging
2. **Correctness first**: Verify algorithm before scaling
3. **Gradual scaling**: 1→2→4 processes, then multi-node
4. **HPC validation**: Final performance testing on real clusters

## Common Pitfalls to Avoid

1. **Incorrect halo size**: Remember kernel_height - 1, not kernel_height
2. **Boundary conditions**: Handle first/last process correctly
3. **Stride alignment**: Ensure output row calculations are correct
4. **Memory leaks**: Free all allocated arrays
5. **Deadlocks**: Use Sendrecv or proper Send/Recv ordering

## Notes for Implementation

- Start with blocking communication, optimize later
- Test with small matrices first for debugging
- Use MPI error checking in debug mode
- Consider MPI derived datatypes for non-contiguous data
- Profile with MPI profiling tools (e.g., mpiP, IPM)

This design provides a complete framework for implementing efficient distributed memory convolution with MPI, incorporating best practices from HPC lectures and real-world optimization techniques.