# MPI Distributed Memory Convolution Design Document

## Overview

This document provides a comprehensive design for implementing distributed memory 2D convolution with stride using MPI, based on HPC principles from CITS3402 lectures and best practices for domain decomposition.

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

### Overall Strategy

```
Global Matrix Distribution (8 processes example):
┌─────────────────────────────────────┐
│         Global Input Matrix         │
│            (H × W)                  │
└─────────────────────────────────────┘
                ↓
       MPI_Scatter (with overlap)
                ↓
┌─────────┬─────────┬─────────┬─────────┐
│ Rank 0  │ Rank 1  │ Rank 2  │ Rank 3  │
│ +halo   │ +halo   │ +halo   │ +halo   │
├─────────┼─────────┼─────────┼─────────┤
│ Rank 4  │ Rank 5  │ Rank 6  │ Rank 7  │
│ +halo   │ +halo   │ +halo   │ +halo   │
└─────────┴─────────┴─────────┴─────────┘
```

### Data Distribution Strategy

#### 1D Row-wise Decomposition (Recommended)

```
Process Assignment with Halo Regions:
┌────────────────────────────┐
│    Process 0 rows          │ ← Original data
│░░░░░░░░░░░░░░░░░░░░░░░░░░░│ ← Halo from Process 1
├────────────────────────────┤
│░░░░░░░░░░░░░░░░░░░░░░░░░░░│ ← Halo from Process 0
│    Process 1 rows          │ ← Original data
│░░░░░░░░░░░░░░░░░░░░░░░░░░░│ ← Halo from Process 2
├────────────────────────────┤
│░░░░░░░░░░░░░░░░░░░░░░░░░░░│ ← Halo from Process 1
│    Process 2 rows          │ ← Original data
└────────────────────────────┘

Halo size = kernel_height - 1
```

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

### 2. Data Distribution Function

```c
void mpi_distribute_matrix(float **global_matrix, int global_H, int global_W,
                          int kernel_H, int kernel_W,
                          float ***local_matrix, int *local_H, int *local_W,
                          int *local_start_row, MPI_Comm comm)
```

**Algorithm Design**:

```
Step 1: Calculate distribution
- Determine rows per process
- Account for stride alignment
- Add halo regions

Step 2: Prepare send counts and displacements
- Each process gets different amount due to halos
- Overlap regions between neighbors

Step 3: Use MPI_Scatterv for distribution
- More flexible than MPI_Scatter
- Handles varying sizes per process

Step 4: Exchange halo regions
- Send bottom rows to next process
- Receive top rows from previous process
```

### 3. Halo Exchange Pattern

```c
void exchange_halos(float **local_matrix, int local_H, int local_W,
                    int kernel_H, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    int halo_size = kernel_H - 1;
    
    // Exchange with neighbors using MPI_Sendrecv
    if (rank > 0) {
        // Send top rows to previous process
        // Receive bottom halo from previous process
    }
    if (rank < size - 1) {
        // Send bottom rows to next process
        // Receive top halo from next process
    }
}
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

### 5. Output Gathering

```c
void mpi_gather_output(float **local_output, int local_output_H, int local_output_W,
                      int local_start_row, float ***global_output,
                      int global_output_H, int global_output_W, MPI_Comm comm)
```

**Gathering Strategy**:

```
Local outputs have different sizes due to:
1. Stride causing non-uniform output distribution
2. Ceiling division in output dimensions

Solution: Use MPI_Gatherv with calculated displacements
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

## Complete MPI Convolution Algorithm

```c
void conv2d_stride_mpi(float **f, int H, int W, float **g, int kH, int kW,
                       int sH, int sW, float **output, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // Step 1: Broadcast kernel to all processes
    mpi_broadcast_kernel(&g, kH, kW, comm);
    
    // Step 2: Distribute input matrix
    float **local_input;
    int local_H, local_W, local_start_row;
    mpi_distribute_matrix(f, H, W, kH, kW, 
                         &local_input, &local_H, &local_W, 
                         &local_start_row, comm);
    
    // Step 3: Calculate local output dimensions
    int local_output_H = calculate_local_output_rows(local_H, kH, sH);
    int local_output_W = (W - kW + 1 + sW - 1) / sW;
    float **local_output = allocate_matrix(local_output_H, local_output_W);
    
    // Step 4: Exchange halos if needed
    if (size > 1) {
        exchange_halos(local_input, local_H, local_W, kH, comm);
    }
    
    // Step 5: Perform local computation
    compute_local_convolution(local_input, local_H, local_W,
                             g, kH, kW, sH, sW,
                             local_output, local_start_row);
    
    // Step 6: Gather results
    int global_output_H = (H - kH + 1 + sH - 1) / sH;
    int global_output_W = local_output_W;
    mpi_gather_output(local_output, local_output_H, local_output_W,
                     local_start_row / sH, &output,
                     global_output_H, global_output_W, comm);
    
    // Cleanup
    free_matrix(local_input, local_H);
    free_matrix(local_output, local_output_H);
}
```

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

## Implementation Checklist

1. [ ] Implement `mpi_broadcast_kernel()`
2. [ ] Implement `calculate_load_distribution()`
3. [ ] Implement `mpi_distribute_matrix()` with Scatterv
4. [ ] Implement `exchange_halos()` with Sendrecv
5. [ ] Implement `compute_local_convolution()`
6. [ ] Implement `mpi_gather_output()` with Gatherv
7. [ ] Add timing instrumentation
8. [ ] Test correctness with various process counts
9. [ ] Optimize communication overlap
10. [ ] Benchmark and analyze performance

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