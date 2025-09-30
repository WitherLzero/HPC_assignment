# OpenMP Stride Convolution Implementation Design Document

## Overview

This document provides a complete design for implementing `conv2d_stride_openmp()` in Assignment 2, building upon the successful patterns from Assignment 1's optimized implementations.

## Algorithm Analysis

### Stride Impact on Parallelization

The stride operation affects parallelization in several ways:

1. **Output size reduction**: Fewer total computations when stride > 1
2. **Memory access pattern**: Strided access can reduce cache efficiency
3. **Load balancing**: Output rows may have different amounts of work

### Parallelization Strategy

```
Original Matrix (8x8) with stride 2x2:
┌─┬─┬─┬─┬─┬─┬─┬─┐
│●│ │●│ │●│ │●│ │  ← Row 0: samples at cols 0,2,4,6
├─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │ │ │ │ │ │ │  ← Row 1: skipped (stride)
├─┼─┼─┼─┼─┼─┼─┼─┤
│●│ │●│ │●│ │●│ │  ← Row 2: samples at cols 0,2,4,6
├─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │ │ │ │ │ │ │  ← Row 3: skipped (stride)
└─┴─┴─┴─┴─┴─┴─┴─┘
Result: 4x4 output matrix
```

## Implementation Design

### Core Function Structure

```c
void conv2d_stride_openmp(float **restrict f, int H, int W, 
                          float **restrict g, int kH, int kW,
                          int sH, int sW, float **restrict output)
```

### Algorithm Phases

#### Phase 1: Setup and Dimension Calculation
```c
// Calculate actual input dimensions (remove padding)
int original_H = H - kH + 1;
int original_W = W - kW + 1;

// Calculate strided output dimensions
int output_H = (int)ceil((double)original_H / sH);
int output_W = (int)ceil((double)original_W / sW);
```

#### Phase 2: Kernel-Specific Optimization Selection

Following Assignment 1's pattern, we'll detect common kernel sizes for optimized paths:

```c
// Check for optimized kernel implementations
if (kH == 3 && kW == 3) {
    conv2d_3x3_stride_optimized_openmp(f, H, W, g, sH, sW, output);
    return;
} else if (kH == 5 && kW == 5) {
    conv2d_5x5_stride_optimized_openmp(f, H, W, g, sH, sW, output);
    return;
}
// Otherwise use general implementation
```

#### Phase 3: General Parallel Implementation

```c
// Main parallel region
#pragma omp parallel
{
    // Thread-local variables for better cache usage
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    
    // Dynamic scheduling for load balancing
    #pragma omp for schedule(dynamic, 4) nowait
    for (int i = 0; i < output_H; i++) {
        // Prefetch hints for next row
        int next_input_row = (i + 1) * sH;
        if (next_input_row < H) {
            __builtin_prefetch(&f[next_input_row][0], 0, 3);
        }
        
        for (int j = 0; j < output_W; j++) {
            float sum = 0.0f;
            int start_row = i * sH;
            int start_col = j * sW;
            
            // Unroll inner loops for better performance
            for (int ki = 0; ki < kH; ki++) {
                #pragma omp simd reduction(+:sum)
                for (int kj = 0; kj < kW; kj++) {
                    sum += f[start_row + ki][start_col + kj] * g[ki][kj];
                }
            }
            
            output[i][j] = sum;
        }
    }
}
```

### Optimized 3x3 Kernel Implementation

```c
void conv2d_3x3_stride_optimized_openmp(float **restrict f, int H, int W, 
                                        float **restrict g, int sH, int sW, 
                                        float **restrict output)
```

Key optimizations:
1. **Kernel unrolling**: Eliminate inner loops
2. **Register caching**: Store kernel values in local variables
3. **SIMD hints**: Enable auto-vectorization

```c
// Cache kernel values
const float g00 = g[0][0], g01 = g[0][1], g02 = g[0][2];
const float g10 = g[1][0], g11 = g[1][1], g12 = g[1][2];
const float g20 = g[2][0], g21 = g[2][1], g22 = g[2][2];

#pragma omp parallel for schedule(static)
for (int i = 0; i < output_H; i++) {
    int row = i * sH;
    for (int j = 0; j < output_W; j++) {
        int col = j * sW;
        
        // Fully unrolled 3x3 convolution
        float sum = f[row][col]     * g00 + f[row][col+1]     * g01 + f[row][col+2]     * g02 +
                    f[row+1][col]   * g10 + f[row+1][col+1]   * g11 + f[row+1][col+2]   * g12 +
                    f[row+2][col]   * g20 + f[row+2][col+1]   * g21 + f[row+2][col+2]   * g22;
        
        output[i][j] = sum;
    }
}
```

### Optimized 5x5 Kernel Implementation

```c
void conv2d_5x5_stride_optimized_openmp(float **restrict f, int H, int W, 
                                        float **restrict g, int sH, int sW, 
                                        float **restrict output)
```

Similar to 3x3 but with 25 kernel values cached:

```c
// Calculate output dimensions
int original_H = H - 4;  // 5x5 kernel needs 4 pixels of padding removed
int original_W = W - 4;
int output_H = (int)ceil((double)original_H / sH);
int output_W = (int)ceil((double)original_W / sW);

// Cache all 25 kernel values
const float g00 = g[0][0], g01 = g[0][1], g02 = g[0][2], g03 = g[0][3], g04 = g[0][4];
const float g10 = g[1][0], g11 = g[1][1], g12 = g[1][2], g13 = g[1][3], g14 = g[1][4];
const float g20 = g[2][0], g21 = g[2][1], g22 = g[2][2], g23 = g[2][3], g24 = g[2][4];
const float g30 = g[3][0], g31 = g[3][1], g32 = g[3][2], g33 = g[3][3], g34 = g[3][4];
const float g40 = g[4][0], g41 = g[4][1], g42 = g[4][2], g43 = g[4][3], g44 = g[4][4];

#pragma omp parallel for schedule(static)
for (int i = 0; i < output_H; i++) {
    int row = i * sH;
    for (int j = 0; j < output_W; j++) {
        int col = j * sW;
        
        // Fully unrolled 5x5 convolution
        float sum = 
            f[row][col]     * g00 + f[row][col+1]     * g01 + f[row][col+2]     * g02 + f[row][col+3]     * g03 + f[row][col+4]     * g04 +
            f[row+1][col]   * g10 + f[row+1][col+1]   * g11 + f[row+1][col+2]   * g12 + f[row+1][col+3]   * g13 + f[row+1][col+4]   * g14 +
            f[row+2][col]   * g20 + f[row+2][col+1]   * g21 + f[row+2][col+2]   * g22 + f[row+2][col+3]   * g23 + f[row+2][col+4]   * g24 +
            f[row+3][col]   * g30 + f[row+3][col+1]   * g31 + f[row+3][col+2]   * g32 + f[row+3][col+3]   * g33 + f[row+3][col+4]   * g34 +
            f[row+4][col]   * g40 + f[row+4][col+1]   * g41 + f[row+4][col+2]   * g42 + f[row+4][col+3]   * g43 + f[row+4][col+4]   * g44;
        
        output[i][j] = sum;
    }
}
```

### Load Balancing Considerations

For strided convolution, load balancing is less of an issue than regular convolution because:
1. Each output element requires the same amount of computation
2. The stride creates regular spacing

However, we still use `schedule(dynamic, 4)` for the general case to handle:
- NUMA effects on large systems
- Cache conflicts between threads
- System load variations

### Memory Access Optimization

The stride pattern creates challenges for cache efficiency:

```
Stride 1 (cache-friendly):     Stride 3 (cache-unfriendly):
[x][x][x][x][x][x]            [x][ ][ ][x][ ][ ][x]
Sequential access             Scattered access
```

Mitigation strategies:
1. **Prefetching**: Explicitly prefetch future rows
2. **Blocking**: Process multiple output elements that share input data
3. **Thread affinity**: Keep threads on same NUMA node

### Performance Tuning Parameters

```c
// Tunable parameters (can be adjusted based on system)
#define OMP_CHUNK_SIZE 4        // Rows per chunk for dynamic scheduling
#define PREFETCH_DISTANCE 2     // Rows to prefetch ahead
#define MIN_PARALLEL_SIZE 100   // Minimum matrix size for parallelization
```

## Expected Performance Characteristics

### Speedup Estimation

For a system with N cores:
- **Stride 1x1**: Near-linear speedup (0.8-0.9 × N)
- **Stride 2x2**: Slightly reduced (0.7-0.8 × N) due to memory access
- **Stride 3x3+**: Further reduced (0.6-0.7 × N) due to cache misses

### Bottlenecks to Watch

1. **Memory bandwidth**: Becomes limiting factor for large strides
2. **Cache conflicts**: Multiple threads accessing strided patterns
3. **Thread overhead**: For small matrices, parallelization may hurt

## Implementation Checklist for Claude Code

1. [ ] Implement basic `conv2d_stride_openmp()` with general algorithm
2. [ ] Add 3x3 optimized version `conv2d_3x3_stride_optimized_openmp()`
3. [ ] Add 5x5 optimized version `conv2d_5x5_stride_optimized_openmp()`
4. [ ] Include prefetching hints
5. [ ] Add SIMD pragmas for inner loops
6. [ ] Implement minimum size threshold for parallelization
7. [ ] Test with various stride combinations
8. [ ] Verify correctness against serial implementation

## Testing Strategy

```bash
# Test correctness
./test_stride -H 100 -W 100 -kH 3 -kW 3 -sH 2 -sW 2 -P -v

# Benchmark performance
for stride in 1 2 3 4; do
    echo "Testing stride ${stride}x${stride}"
    ./test_stride -H 1000 -W 1000 -kH 5 -kW 5 -sH $stride -sW $stride -P -t
done
```

## Code Template for Implementation

```c
// Complete implementation template for conv2d_stride_openmp
void conv2d_stride_openmp(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                          int sH, int sW, float **restrict output) {
    // Calculate dimensions
    int original_H = H - kH + 1;
    int original_W = W - kW + 1;
    int output_H = (int)ceil((double)original_H / sH);
    int output_W = (int)ceil((double)original_W / sW);

    // Check for minimum size threshold
    if (output_H * output_W < MIN_PARALLEL_SIZE) {
        conv2d_stride_serial(f, H, W, g, kH, kW, sH, sW, output);
        return;
    }

    // Select optimized implementation based on kernel size
    if (kH == 3 && kW == 3) {
        conv2d_3x3_stride_optimized_openmp(f, H, W, g, sH, sW, output);
        return;
    } else if (kH == 5 && kW == 5) {
        conv2d_5x5_stride_optimized_openmp(f, H, W, g, sH, sW, output);
        return;
    }

    // General implementation
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, OMP_CHUNK_SIZE) nowait
        for (int i = 0; i < output_H; i++) {
            // Prefetch next input row
            int next_row = (i + PREFETCH_DISTANCE) * sH;
            if (next_row < H) {
                __builtin_prefetch(&f[next_row][0], 0, 3);
            }

            for (int j = 0; j < output_W; j++) {
                float sum = 0.0f;
                int start_row = i * sH;
                int start_col = j * sW;

                // Apply kernel with bounds checking
                if (start_row + kH <= H && start_col + kW <= W) {
                    for (int ki = 0; ki < kH; ki++) {
                        #pragma omp simd reduction(+:sum)
                        for (int kj = 0; kj < kW; kj++) {
                            sum += f[start_row + ki][start_col + kj] * g[ki][kj];
                        }
                    }
                }

                output[i][j] = sum;
            }
        }
    }
}
```

## Notes for Implementation

1. **Reuse Assignment 1 patterns**: The kernel-specific optimizations from Assignment 1 work well with stride
2. **Keep it simple first**: Get the basic parallel version working before optimizing
3. **Verify correctness**: Always compare against serial implementation
4. **Profile before optimizing**: Use `perf` to identify actual bottlenecks
5. **Consider memory patterns**: Stride affects cache usage significantly
6. **Test edge cases**: Ensure correct handling when stride doesn't divide evenly

This design provides a solid foundation for OpenMP parallelization that builds on your successful Assignment 1 approach while handling the stride complexity.