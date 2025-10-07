# Assignment 2 - Setonix Testing Draft


**Date:** October 2025

---

## 1. Testing Environment

### 1.1 Setonix HPC System Specifications

This assignment extends the testing environment from Assignment 1 by adding **Setonix**, Pawsey Supercomputing Centre's flagship HPC system, alongside the existing **Kaya** cluster.

##### **Table 1: Setonix HPC System Specifications**

| **Component** | **Specification** | **Details** |
|---------------|-------------------|-------------|
| **System Name** | Setonix | Pawsey Supercomputing Centre |
| **Location** | Perth, Western Australia | HPE Cray EX System |
| **CPU Model** | AMD EPYC 7763 | 64-Core Processor (Milan) |
| **Architecture** | x86_64 | AMD Zen 3 |
| **Frequency** | 2.45 GHz base | 3.53 GHz boost |
| | | |
| **Cores per Node** | 128 physical cores | 2 sockets × 64 cores |
| **Threads per Node** | 256 logical cores | 2 threads per core (SMT) |
| **Sockets** | 2 | Dual-socket configuration |
| **NUMA Domains** | 8 | Advanced memory locality |
| | | |
| **Memory per Node** | 250 GB | (~1.95 GB/core) |
| **Total Memory** | 262,793,164 KB | ~257 GB total |
| **Available Memory** | ~235 GB | After OS reservation |
| **Swap** | 0 GB | No swap configured |
| | | |
| **Cache L1d** | 4 MB | 128 instances (32 KB each) |
| **Cache L1i** | 4 MB | 128 instances (32 KB each) |
| **Cache L2** | 64 MB | 128 instances (512 KB each) |
| **Cache L3** | 512 MB | 16 instances (32 MB each) |
| | | |
| **MPI Implementation** | Cray MPICH 8.1.32 | HPE Cray MPI (not OpenMPI) |
| **Compiler** | Cray cc wrapper | GCC 14.2 backend (gcc-native) |
| **Programming Environment** | PrgEnv-gnu 8.6.0 | Cray Programming Environment |
| **Interconnect** | Cray Slingshot 11 | High-speed network fabric |
| | | |
| **Node Count** | 1,376 compute nodes | (work partition) |
| **Total System Cores** | 352,256 logical cores | Full system capacity |
| **Partition Used** | work | Default partition |
| **Account** | courses01 | Educational allocation |
| **Max Wall Time** | 24 hours | 1-00:00:00 |
| **Max Nodes per Job** | 256 | (configurable limit) |
| | | |
| **Operating System** | SUSE Linux 6.4.0 | Cray-optimized kernel |
| **Slurm Version** | 24.11.6 | Workload manager |
| **Node Naming** | nid[001000-002823] | Compact node IDs |

**Key Differences from Kaya:**
- **128 physical cores/node** vs. Kaya's 48 cores (2.67× more)
- **250 GB memory/node** vs. Kaya's ~64 GB (3.9× more)
- **Cray MPICH** instead of OpenMPI (requires `cc` compiler wrapper)
- **No hyperthreading on Kaya** vs. 2-way SMT on Setonix
- **HPE Cray supercomputer** vs. commodity cluster architecture



### 1.2 Test Matrix and Methodology

**Performance Testing Sections:**

We conducted 5 comprehensive test sections on both Kaya and Setonix:

1. **Strong Scaling:** Fixed 50K×50K matrix, varying core counts
2. **Weak Scaling:** Proportional problem size to core count
3. **Stride Impact:** 80K×80K matrix with stride 1×1, 2×2, 3×3
4. **Kernel Size Impact:** 80K×80K matrix with 3×3, 5×5, 7×7 kernels
5. **Multi-Node Scaling:** 100K×100K matrix, 1 vs 2 nodes

**Stress Testing:**

Pushed memory limits to maximum capacity:
- **Kaya:** Up to 500K×500K matrices (testing in progress)
- **Setonix:** Successfully tested 230K×230K (222 GB memory usage)
  - 240K×240K failed with OOM (out of memory)
  - Maximum for 2-node config: ~230K×230K


---

## 2. Timing Methodology

**Critical Design Decision: Separate Convolution and Gather Timing**

We measure **two distinct phases** in our parallel algorithm:

### 2.1 Timing Breakdown

```c
// Phase 1: Convolution (Algorithm Core)
mpi_timer_start(&conv_timer);
    conv2d_stride_hybrid(...);  // Includes halo exchange
mpi_timer_end(&conv_timer);

// Phase 2: Output Gathering (Post-processing)
mpi_timer_start(&gather_timer);
    mpi_gather_output(...);     // Collect results to rank 0
mpi_timer_end(&gather_timer);
```

**Output Format:**
```
[INFO] Timing - Convolution: 51.105 milliseconds
[INFO] Timing - Gather: 8678.103 milliseconds
[INFO] Timing - Total: 8729.208 milliseconds
```

### 2.2 Rationale for Separation

**Why Convolution and Gather are Reported Separately:**

| **Aspect** | **Convolution Time** | **Gather Time** |
|------------|---------------------|-----------------|
| **What it measures** | Pure algorithm computation + halo exchange | All-to-one data collection |
| **Part of algorithm?** | ✅ Yes - required for correctness | ❌ No - optional post-processing |
| **Happens every time?** | ✅ Always | ❌ Only when full output needed |
| **Scalability** | Neighbor-to-neighbor (scalable) | All-to-one (bottleneck at rank 0) |
| **Data volume** | O(boundary_rows × W) per process | O(H × W) to single process |
| **Communication pattern** | MPI_Sendrecv (P2P) | MPI_Gather (collective) |
| **Relevant for** | Algorithm performance analysis | I/O and communication overhead analysis |

**Halo Exchange vs. Gather: Why Different Treatment?**

- **Halo exchange** is **intrinsic to the convolution algorithm** itself
  - Each process needs boundary data from neighbors to compute correctly
  - Volume: O(H/P × W) - proportional to local work
  - Pattern: Scalable neighbor-to-neighbor communication
  - **Included in convolution timing**

- **Gather** is **not part of the algorithm** - it's post-processing
  - Only needed when we want centralized output (verification, single-file output)
  - Volume: O(H × W) - **ALL data** goes to rank 0 (memory bottleneck!)
  - Pattern: Non-scalable all-to-one communication
  - **Excluded from convolution timing**

**Observed Impact:**

From our Setonix testing (50K×50K matrix, 16 processes):
- Convolution: 51 ms (actual algorithm)
- Gather: 8,678 ms (data movement)
- **Ratio: 170:1** - gather is 170× slower than computation!

This massive overhead demonstrates why we **must** report them separately to properly analyze:
1. **Algorithmic efficiency** (convolution time)
2. **Communication overhead** (gather time)
3. **Total cost** (sum of both)


---


## 3. Performance Testing Results

**Job ID:** 32930679
**Test Date:** October 8, 2025 (01:53-02:00 AWST)
**Configuration:** 2 nodes, 16 MPI processes × 16 OpenMP threads = 256 cores

### 3.1 Strong Scaling Test

Fixed problem size: 50000×50000 matrix, 3×3 kernel, stride 1×1

| Configuration | Cores | Conv Time (ms) | Gather Time (ms) | Total Time (ms) | Gather % |
|---------------|-------|----------------|------------------|-----------------|----------|
| 16 proc × 16 threads | 256 | 202.5 | 8532.3 | 8734.8 | 97.7% |
| 16 proc × 8 threads | 128 | 96.6 | 8507.2 | 8603.9 | 98.9% |
| 16 proc × 4 threads | 64 | 189.3 | 8522.3 | 8711.6 | 97.8% |
| 8 proc × 16 threads | 128 | 98.9 | 8014.9 | 8113.9 | 98.8% |
| 8 proc × 8 threads | 64 | 190.5 | 8095.0 | 8285.5 | 97.7% |

**Key Observations:**
- **Gather overhead dominates:** Accounts for ~98% of total execution time
- **Convolution scales moderately:** 64→256 cores shows ~2× speedup (189ms→97ms for 128 cores)
- **Gather time nearly constant:** ~8 seconds regardless of core count (primarily depends on matrix size)
- **Optimal configuration:** 128 cores (8 proc × 16 threads) shows best convolution time (98.9ms) with reasonable total time

### 3.2 Weak Scaling Test

Constant work per core, 3×3 kernel, stride 1×1

| Cores | Matrix Size | Work/Core | Conv Time (ms) | Gather Time (ms) | Total Time (ms) |
|-------|-------------|-----------|----------------|------------------|-----------------|
| 64 (4×16) | 50000×50000 | ~39M elements | 193.3 | 7041.8 | 7235.1 |
| 128 (8×16) | 70711×70711 | ~39M elements | 196.3 | 15613.2 | 15809.5 |
| 256 (16×16) | 100000×100000 | ~39M elements | 199.8 | 33247.7 | 33447.5 |

**Key Observations:**
- **Excellent weak scaling for convolution:** 193ms → 200ms (only 3.6% increase, **97.7% efficiency**)
- **Gather time scales with matrix size:** 7s → 16s → 33s (approximately quadratic with dimension)
- **Total time dominated by gather:** As problem size grows, gather becomes increasingly dominant
- **Convolution efficiency:** Near-perfect weak scaling demonstrates good load balancing

### 3.3 Stride Impact Test

80000×80000 matrix, 3×3 kernel, 256 cores

| Stride | Output Size | Conv Time (ms) | Gather Time (ms) | Total Time (ms) | Speedup |
|--------|-------------|----------------|------------------|-----------------|---------|
| 1×1 | 80000×80000 | 170.2 | 21033.0 | 21203.2 | 1.0× |
| 2×2 | 40000×40000 | 60.6 | 8900.1 | 8960.7 | **2.4×** |
| 3×3 | 26667×26667 | 49.9 | 2518.8 | 2568.6 | **8.3×** |

**Key Observations:**
- **Significant performance improvement with stride:** Stride 3×3 is 8.3× faster than stride 1×1
- **Output size reduction drives speedup:** Output reduces to 1/9 for stride 3×3, computation reduces proportionally
- **Gather benefits more:** Gather time drops from 21s to 2.5s (8.4× reduction)
- **Practical application:** High stride useful for downsampling operations

### 3.4 Kernel Size Impact Test

80000×80000 matrix, stride 1×1, 256 cores

| Kernel | Theoretical FLOPs | Conv Time (ms) | Gather Time (ms) | Total Time (ms) | Overhead |
|--------|-------------------|----------------|------------------|-----------------|----------|
| 3×3 | 9 ops/pixel | 181.4 | 21162.7 | 21344.1 | 1.0× |
| 5×5 | 25 ops/pixel | 202.7 | 21116.4 | 21319.1 | **1.1×** |
| 7×7 | 49 ops/pixel | 1589.7 | 21059.3 | 22649.0 | **8.8×** |

**Key Observations:**
- **3×3 and 5×5 well-optimized:** 5×5 only 12% slower despite 2.78× more FLOPs
- **7×7 performance collapse:** 8.8× slower vs. expected 5.4× (based on FLOPs ratio)
- **Cause:** Code has specialized optimizations for 3×3 and 5×5, but uses generic path for 7×7
- **Cache effects:** Larger kernels likely suffer more cache misses
- **Recommendation:** Implement specialized 7×7 kernel optimization

### 3.5 Multi-Node Scaling Test

100000×100000 matrix, 3×3 kernel, stride 1×1

| Configuration | Nodes | Cores | Conv Time (ms) | Gather Time (ms) | Total Time (ms) | Speedup |
|---------------|-------|-------|----------------|------------------|-----------------|---------|
| Single node | 1 | 128 (8×16) | 390.2 | 31358.3 | 31748.5 | 1.0× |
| Dual node | 2 | 256 (16×16) | 200.0 | 33369.8 | 33569.7 | **0.95×** |

**Key Observations:**
- **Convolution speedup: 1.95×** (390ms → 200ms), near-ideal for 2× core increase
- **Gather overhead increases:** Cross-node gather adds ~2s overhead (31.4s → 33.4s)
- **Total time slightly slower:** Due to increased gather overhead
- **Conclusion:** Multi-node beneficial for convolution but hurts total time due to gather bottleneck

---

## 4. Stress Testing Results

### 4.1 Test Summary

| Matrix Size | Kernel | Stride | Status | Conv Time | Gather Time | Total Time | Peak Memory |
|-------------|--------|--------|--------|-----------|-------------|------------|-------------|
| 190K×190K | 5×5 | 2×2 | ✅ SUCCESS | 1.6s | 278.2s | 279.9s | 179 GB |
| 230K×230K | 5×5 | 1×1 | ✅ SUCCESS | 60.0s | 438.9s | 498.9s | **222 GB** |
| 200K×200K | 5×5 | 2×2 | ❌ OOM | - | - | - | 159 GB* |
| 240K×240K | 5×5 | 1×1 | ❌ OOM | - | - | - | 41 GB* |

*Memory data incomplete (job terminated early)

### 4.2 Successful Cases Analysis

**Maximum Successful Test: 230K×230K (stride 1×1)**

Job ID: 32931558
Configuration: 16 processes × 16 threads = 256 cores

- **Convolution time:** 60.0 seconds
- **Gather time:** 438.9 seconds (**7.3× slower** than convolution\!)
- **Total time:** 498.9 seconds (~8.3 minutes)
- **Peak memory:** 222 GB (approaching 250 GB node limit)
- **Gather percentage:** 88% of total time
- **Wall clock time:** 525 seconds (computation + overhead)

**Analysis:**
- This represents the **maximum matrix size** achievable on 2-node configuration
- Memory bottleneck at rank 0 during gather (receives full 230K×230K×4 bytes = 211 GB)
- Convolution-to-gather ratio shows gather dominates large-scale problems


**Stride 2×2 Case: 190K×190K**

Job ID: 32930837

- **Convolution time:** 1.6 seconds (much faster due to stride 2×2 reducing output)
- **Gather time:** 278.2 seconds
- **Total time:** 279.9 seconds (~4.7 minutes)
- **Peak memory:** 179 GB
- **Gather percentage:** 99.4% of total time

**Analysis:**
- Stride 2×2 drastically reduces convolution time but gather still dominates
- Output size: 95K×95K (vs 230K×230K), explaining faster gather
- Gather-to-convolution ratio: **170:1** (even more extreme than stride 1×1)

### 4.3 Failed Cases Analysis

**240K×240K Failure (Job 32931754)**

- **Predicted memory requirement:** ~259 GB at rank 0
- **Node memory limit:** 250 GB per node
- **Result:** OOM killed after 84 seconds
- **Peak memory measured:** 41 GB (killed early during matrix generation)

**Root Cause:**
- Even though total system memory = 500 GB (2 nodes × 250 GB)
- Rank 0 alone needs >250 GB during gather phase
- **Single-node memory limit** prevents using larger matrices
- This is a **fundamental gather bottleneck**


**200K×200K (stride 2×2) Failure (Job 32931246)**

- **Status:** OOM at task 7 after 382 seconds
- **Peak memory:** 159 GB (partial data)
- **Error:** Multiple OOM kill events

**Possible Causes:**
- Memory distribution imbalance across processes
- Stride 2×2 may create uneven work distribution
- Process 7 may have hit local memory limit during intermediate computation

### 4.4 Memory Limit Findings

**2-Node Setonix Configuration (500 GB total):**

- **Maximum matrix (stride 1×1):** ~230K × 230K
- **Limiting factor:** Rank 0 single-node memory (250 GB), NOT total system memory
- **Bottleneck:** MPI_Gather requires all data to fit in one nodes memory

**Recommendations:**
- For matrices >230K: Use 4-node configuration (1 TB total memory, 250 GB/node)
- Better solution: Eliminate gather, use distributed MPI-IO for output
- Alternative: Implement streamed gather to avoid memory spike

---

**Draft Version:** 1.0
**Last Updated:** October 8, 2025
**Data Location:** `t_Setonix/setonix_results/`

