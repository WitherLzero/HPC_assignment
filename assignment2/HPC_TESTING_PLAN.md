# HPC Performance Testing Plan for Assignment 2

## Document Purpose
This document provides a comprehensive testing strategy for running MPI+OpenMP 2D convolution performance analysis on Kaya and Setonix supercomputers. This guide is designed for both human users and AI assistants working on the HPC systems.

---

## Project Context & Current Status

### What Has Been Completed (Local Development)

**Implementation Status: ✅ COMPLETE**

We have successfully implemented all four required variants of 2D strided convolution:

1. **Serial Implementation** (`-s` flag)
   - Single-threaded baseline
   - Function: `conv2d_stride_serial()`

2. **OpenMP-only Implementation** (`-P` flag)
   - Shared memory parallelization
   - Function: `conv2d_stride_openmp()`
   - Includes optimized 3×3 and 5×5 kernel variants

3. **MPI-only Implementation** (`-m` flag)
   - Distributed memory parallelization
   - Function: `conv2d_stride_mpi()`
   - Features: Kernel broadcast, matrix distribution, halo exchange, output gathering

4. **Hybrid MPI+OpenMP Implementation** (default, no flag)
   - Combined distributed + shared memory
   - Function: `conv2d_stride_hybrid()`
   - Uses MPI for inter-process distribution, OpenMP for intra-process threading

**Key Features Implemented:**
- ✅ Stride support (sH × sW) in both vertical and horizontal directions
- ✅ "Same" padding for maintaining dimensions
- ✅ Row-major memory layout
- ✅ Dual distribution strategies (stride=1 vs stride>1)
- ✅ Sub-communicator approach for optimal process utilization
- ✅ Contiguous buffer communication for efficiency
- ✅ Cache-line alignment and SIMD optimizations in OpenMP version

**Verification Status:**
- ✅ All 6 provided test cases pass (f0-f5 with various strides)
- ✅ Tested with 2 and 4 MPI processes
- ✅ Correctness verified for matrices up to 10000×10000

### Local Performance Baseline (Reference)

**Test Configuration:** 10000×10000 matrix, 3×3 kernel, stride 1×1

| Implementation | Processes | Threads/Process | Time (ms) | Speedup |
|----------------|-----------|-----------------|-----------|---------|
| Serial         | 1         | 1               | 1204.3    | 1.00×   |
| OpenMP-only    | 1         | 4               | 327.1     | 3.68×   |
| MPI-only       | 2         | 1               | 1309.9    | 0.92×   |
| Hybrid         | 2         | 2               | 858.6     | 1.40×   |
| MPI-only       | 4         | 1               | 915.1     | 1.32×   |
| Hybrid         | 4         | 4               | 680.7     | 1.77×   |
| Hybrid         | 4         | 2               | 691.6     | 1.74×   |

**Key Insight:** On single node, OpenMP dominates due to zero communication overhead. Hybrid approach will show advantage on multi-node clusters (Kaya/Setonix).

---

## Project Structure

```
assignment2/
├── src/
│   ├── main.c              # CLI interface with getopt
│   ├── conv2d_mpi.c        # All 4 implementations
│   └── io_mpi.c            # Matrix I/O functions
├── include/
│   └── conv2d_mpi.h        # Function declarations
├── f/                      # Test input matrices (f0-f5)
├── g/                      # Test kernels (g0-g5)
├── o/                      # Expected outputs
├── Makefile                # Build system (CRITICAL: must work with simple 'make')
└── HPC_TESTING_PLAN.md     # This document
```

---

## Assignment Requirements Summary

### Critical Build Requirement
**⚠️ ZERO MARKS if this fails:** Code must compile with simple `make` command on Kaya and Setonix.

### Grading Breakdown (30 marks total)

| Component | Marks | Notes |
|-----------|-------|-------|
| Serial stride implementation | 1 | ✅ Done |
| Matrix I/O and main file | 1 | ✅ Done |
| MPI-only implementation | 2 | ✅ Done |
| Hybrid MPI+OpenMP | 3 | ✅ Done |
| **Description of parallelism** | 5 | 📝 Report needed |
| Data decomposition description | 2 | 📝 Report needed |
| Communication strategy description | 3 | 📝 Report needed |
| **Performance metrics & analysis** | **10** | 🎯 **HPC TESTING FOCUS** |
| Formatting and presentation | 3 | 📝 Report needed |

**Note:** Performance analysis is the **single largest component** (10/30 marks).

### Testing Constraints
- **Maximum runtime:** 15 minutes per test
- **Maximum nodes:** 4 nodes (on both Kaya and Setonix)
- **Setonix CPU hours:** Limited allocation - be conservative

---

## Command-Line Interface Reference

### Compilation
```bash
make                    # Build with optimizations (-O3)
make clean              # Remove build artifacts
```

### Basic Usage Pattern
```bash
# With MPI (required for distributed testing)
mpirun -np <N> ./build/conv_stride_test [OPTIONS]

# Or on HPC systems via SLURM
srun ./build/conv_stride_test [OPTIONS]
```

### Key Options

**Input Specification:**
- `-f FILE` : Input feature map file
- `-g FILE` : Input kernel file
- `-H HEIGHT` : Generate random matrix with height
- `-W WIDTH` : Generate random matrix with width
- `-kH HEIGHT` : Kernel height
- `-kW WIDTH` : Kernel width

**Stride Parameters:**
- `-sH VALUE` : Vertical stride (default: 1)
- `-sW VALUE` : Horizontal stride (default: 1)

**Implementation Selection:**
- `-s` : Serial only (single-threaded)
- `-m` : MPI-only (distributed memory)
- `-P` : OpenMP-only (shared memory, multi-threaded)
- (no flag) : Hybrid MPI+OpenMP (default)

**Output & Verification:**
- `-o FILE` : Output file
- `-p PRECISION` : Verify mode (1=0.1, 2=0.01 tolerance)
- `-t` : Time execution in milliseconds
- `-v` : Verbose output

**OpenMP Control (environment variable):**
```bash
export OMP_NUM_THREADS=4    # Set threads per MPI process
```

### Example Commands

```bash
# Generate 10000×10000, 3×3 kernel, time hybrid execution
mpirun -np 4 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -sH 1 -sW 1 -t

# Test with stride 2×3
mpirun -np 2 ./build/conv_stride_test -H 5000 -W 5000 -kH 5 -kW 5 -sH 2 -sW 3 -t

# Verify against test case
mpirun -np 4 ./build/conv_stride_test -f f/f5.txt -g g/g5.txt -o o/o5_sH_2_sW_3.txt -sH 2 -sW 3 -p 2 -m

# Control OpenMP threads in hybrid mode
export OMP_NUM_THREADS=8
mpirun -np 4 ./build/conv_stride_test -H 20000 -W 20000 -kH 3 -kW 3 -t

# Serial baseline
mpirun -np 1 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -s -t

# OpenMP-only with 16 threads
export OMP_NUM_THREADS=16
mpirun -np 1 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -P -t
```

---

## HPC Testing Strategy

### Phase 1: System Setup & Verification (15-30 minutes)

#### On Kaya

**1.1 Initial Setup**
```bash
# SSH to Kaya
ssh <username>@kaya.hpc.uwa.edu.au

# Create working directory
mkdir -p /scratch/cits3402/<username>/assignment2
cd /scratch/cits3402/<username>/assignment2

# Transfer code (use scp/rsync from local machine)
# Or: git clone if using version control
```

**1.2 Module Loading**
```bash
module load gcc/14.3
module load openmpi/5.0.5
```

**1.3 Build Test**
```bash
make clean
make

# Verify binary exists
ls -lh build/conv_stride_test
```

**1.4 Quick Functionality Test**
```bash
# Small serial test (should complete in <1s)
mpirun -np 1 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -s -t

# Small MPI test with 2 processes
mpirun -np 2 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -m -t

# Expected output should show timing in milliseconds
```

#### On Setonix

**1.1 Initial Setup**
```bash
# SSH to Setonix
ssh <username>@setonix.pawsey.org.au

# Create working directory
mkdir -p /scratch/courses0101/<username>/assignment2
cd /scratch/courses0101/<username>/assignment2

# Transfer code
```

**1.2 Build Test (Setonix uses Cray compiler)**
```bash
# Load appropriate modules (check with: module avail)
module load PrgEnv-gnu  # or similar

make clean
make

# Verify build
ls -lh build/conv_stride_test
```

---

### Phase 2: SLURM Job Script Creation

#### Kaya Job Script Template

Create file: `kaya_benchmark.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=conv2d_bench
#SBATCH --nodes=4
#SBATCH --ntasks=16              # Total MPI processes
#SBATCH --ntasks-per-node=4      # 4 MPI processes per node
#SBATCH --cpus-per-task=4        # 4 OpenMP threads per MPI process
#SBATCH --time=00:15:00          # 15 minute limit
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=cits3402
#SBATCH --output=output_%j.txt   # %j = job ID

# Load modules
module load gcc/14.3
module load openmpi/5.0.5

# Set OpenMP threads
export OMP_NUM_THREADS=4

# Print configuration
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "======================================"

# Run benchmark
# MODIFY PARAMETERS AS NEEDED
mpirun -np 16 ./build/conv_stride_test \
    -H 20000 -W 20000 \
    -kH 3 -kW 3 \
    -sH 1 -sW 1 \
    -t

echo "Job completed successfully"
```

#### Setonix Job Script Template

Create file: `setonix_benchmark.slurm`

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --partition=work
#SBATCH --account=courses0101
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --output=output_%j.txt

# Setonix-specific settings
export OMP_NUM_THREADS=16  # Adjust based on cores per node

# Print configuration
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "======================================"

# Run benchmark
srun --ntasks=16 --cpus-per-task=16 ./build/conv_stride_test \
    -H 20000 -W 20000 \
    -kH 3 -kW 3 \
    -sH 1 -sW 1 \
    -t

echo "Job completed successfully"
```

---

### Phase 3: Performance Testing Matrix

#### Required Performance Metrics

**Strong Scaling Analysis:**
- Fixed problem size, varying process/thread count
- Measure: Speedup = T_serial / T_parallel
- Efficiency = Speedup / (Processes × Threads)

**Weak Scaling Analysis:**
- Fixed problem size per process
- Measure: How time changes as both problem size and resources scale proportionally

**Metrics to Collect:**
1. Execution time (milliseconds)
2. Speedup vs serial baseline
3. Parallel efficiency
4. Effect of stride on performance
5. Communication overhead (compare MPI-only vs Hybrid)
6. Cache performance impact

#### Test Configuration Matrix

**Dimension 1: Problem Sizes**
- Small: 1000×1000
- Medium: 5000×5000
- Large: 10000×10000
- Very Large: 20000×20000 (if time permits)
- Maximum: Find largest size that completes in <15 min

**Dimension 2: Kernel Sizes**
- 3×3 (optimized)
- 5×5 (optimized)
- 7×7 (general case)

**Dimension 3: Stride Configurations**
- 1×1 (no stride - most compute-intensive)
- 2×2 (reduced output)
- 3×2 (asymmetric stride)

**Dimension 4: Parallelization Variants**
- Serial: 1 process, 1 thread
- OpenMP-only: 1 process, varying threads (1, 2, 4, 8, 16)
- MPI-only: varying processes (1, 2, 4, 8, 16), 1 thread each
- Hybrid: varying process×thread combinations
  - 2 proc × 8 threads = 16 cores
  - 4 proc × 4 threads = 16 cores
  - 8 proc × 2 threads = 16 cores
  - 16 proc × 1 thread = 16 cores (equivalent to MPI-only)

**Dimension 5: Node Configurations**
- 1 node (baseline)
- 2 nodes
- 4 nodes (maximum)

#### Suggested Testing Priority

**Priority 1: Core Comparison (MUST DO)**
```bash
# 10000×10000, 3×3 kernel, stride 1×1
# Compare all 4 implementations

# Serial baseline
mpirun -np 1 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -s -t

# OpenMP-only (vary threads: 1, 2, 4, 8, 16)
export OMP_NUM_THREADS=1
mpirun -np 1 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -P -t
# Repeat for 2, 4, 8, 16 threads

# MPI-only (vary processes: 2, 4, 8, 16)
mpirun -np 2 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -m -t
# Repeat for 4, 8, 16 processes

# Hybrid (strategic combinations)
export OMP_NUM_THREADS=8
mpirun -np 2 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -t

export OMP_NUM_THREADS=4
mpirun -np 4 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -t

export OMP_NUM_THREADS=2
mpirun -np 8 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -t
```

**Priority 2: Strong Scaling (IMPORTANT)**
```bash
# Fix problem size (20000×20000), vary resources
# Test hybrid with different process×thread combinations

# Baseline
export OMP_NUM_THREADS=1
mpirun -np 1 ./build/conv_stride_test -H 20000 -W 20000 -kH 3 -kW 3 -t

# Scale up
export OMP_NUM_THREADS=4
mpirun -np 1 ./build/conv_stride_test -H 20000 -W 20000 -kH 3 -kW 3 -t
mpirun -np 2 ./build/conv_stride_test -H 20000 -W 20000 -kH 3 -kW 3 -t
mpirun -np 4 ./build/conv_stride_test -H 20000 -W 20000 -kH 3 -kW 3 -t
mpirun -np 8 ./build/conv_stride_test -H 20000 -W 20000 -kH 3 -kW 3 -t
```

**Priority 3: Stride Impact Analysis**
```bash
# Compare stride 1×1 vs 2×2 vs 3×3
# Fixed: 10000×10000, 3×3 kernel, hybrid 4 proc × 4 threads

export OMP_NUM_THREADS=4

# Stride 1×1 (baseline)
mpirun -np 4 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -sH 1 -sW 1 -t

# Stride 2×2 (4× less output)
mpirun -np 4 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -sH 2 -sW 2 -t

# Stride 3×3 (9× less output)
mpirun -np 4 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -sH 3 -sW 3 -t
```

**Priority 4: Multi-Node Scaling (CRITICAL FOR MPI)**
```bash
# Test across 1, 2, 4 nodes
# This demonstrates MPI advantage over pure OpenMP

# 1 node, 4 processes
#SBATCH --nodes=1
#SBATCH --ntasks=4

# 2 nodes, 8 processes
#SBATCH --nodes=2
#SBATCH --ntasks=8

# 4 nodes, 16 processes
#SBATCH --nodes=4
#SBATCH --ntasks=16
```

**Priority 5: Kernel Size Comparison**
```bash
# Compare 3×3 vs 5×5 vs 7×7
# Fixed: 10000×10000, stride 1×1, hybrid 4 proc × 4 threads

export OMP_NUM_THREADS=4

mpirun -np 4 ./build/conv_stride_test -H 10000 -W 10000 -kH 3 -kW 3 -t
mpirun -np 4 ./build/conv_stride_test -H 10000 -W 10000 -kH 5 -kW 5 -t
mpirun -np 4 ./build/conv_stride_test -H 10000 -W 10000 -kH 7 -kW 7 -t
```

---

### Phase 4: Automated Batch Testing Script

Create file: `run_benchmarks.sh`

```bash
#!/bin/bash

# Automated benchmark suite for assignment 2
# This script generates multiple SLURM jobs

OUTPUT_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Creating benchmark jobs in $OUTPUT_DIR"

# Function to create job script
create_job() {
    local name=$1
    local nodes=$2
    local ntasks=$3
    local cpus_per_task=$4
    local omp_threads=$5
    local H=$6
    local W=$7
    local kH=$8
    local kW=$9
    local sH=${10}
    local sW=${11}
    local impl_flag=${12}  # -s, -m, -P, or empty for hybrid

    local job_file="$OUTPUT_DIR/job_${name}.slurm"

    cat > $job_file <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --nodes=$nodes
#SBATCH --ntasks=$ntasks
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=cits3402
#SBATCH --output=$OUTPUT_DIR/${name}_%j.out

module load gcc/14.3
module load openmpi/5.0.5

export OMP_NUM_THREADS=$omp_threads

echo "=== $name ==="
echo "Nodes: $nodes, Tasks: $ntasks, CPUs/task: $cpus_per_task, OMP: $omp_threads"
echo "Matrix: ${H}×${W}, Kernel: ${kH}×${kW}, Stride: ${sH}×${sW}"
echo "Implementation: $impl_flag"

mpirun -np $ntasks ./build/conv_stride_test \\
    -H $H -W $W -kH $kH -kW $kW -sH $sH -sW $sW $impl_flag -t

echo "=== Completed ==="
EOF

    echo "Created: $job_file"
}

# Example: Strong scaling test series
# 10000×10000, 3×3 kernel, stride 1×1, vary processes

create_job "serial_baseline" 1 1 1 1 10000 10000 3 3 1 1 "-s"
create_job "omp_only_4t" 1 1 4 4 10000 10000 3 3 1 1 "-P"
create_job "omp_only_8t" 1 1 8 8 10000 10000 3 3 1 1 "-P"
create_job "mpi_only_4p" 1 4 1 1 10000 10000 3 3 1 1 "-m"
create_job "mpi_only_8p" 2 8 1 1 10000 10000 3 3 1 1 "-m"
create_job "hybrid_2p4t" 1 2 4 4 10000 10000 3 3 1 1 ""
create_job "hybrid_4p4t" 2 4 4 4 10000 10000 3 3 1 1 ""
create_job "hybrid_8p2t" 2 8 2 2 10000 10000 3 3 1 1 ""

# Stride comparison
create_job "stride_1x1" 1 4 4 4 10000 10000 3 3 1 1 ""
create_job "stride_2x2" 1 4 4 4 10000 10000 3 3 2 2 ""
create_job "stride_3x3" 1 4 4 4 10000 10000 3 3 3 3 ""

# Kernel size comparison
create_job "kernel_3x3" 1 4 4 4 10000 10000 3 3 1 1 ""
create_job "kernel_5x5" 1 4 4 4 10000 10000 5 5 1 1 ""
create_job "kernel_7x7" 1 4 4 4 10000 10000 7 7 1 1 ""

echo ""
echo "Job scripts created in $OUTPUT_DIR"
echo ""
echo "To submit all jobs:"
echo "  for job in $OUTPUT_DIR/*.slurm; do sbatch \$job; done"
echo ""
echo "To monitor:"
echo "  squeue --me"
```

Make executable:
```bash
chmod +x run_benchmarks.sh
```

---

### Phase 5: Data Collection & Analysis

#### Collecting Results

```bash
# After jobs complete, extract timing data
cd results_YYYYMMDD_HHMMSS/

# Extract all timing information
grep "Timing - Convolution" *.out > timing_summary.txt

# Example parsing script
cat > parse_results.py <<'EOF'
import re
import sys

for line in sys.stdin:
    # Parse filename and timing
    match = re.search(r'(\w+)_\d+\.out:Timing - Convolution with stride: ([\d.]+) milliseconds', line)
    if match:
        name = match.group(1)
        time_ms = float(match.group(2))
        print(f"{name}\t{time_ms:.3f}")
EOF

cat timing_summary.txt | python3 parse_results.py | sort -t$'\t' -k2 -n
```

#### Required Metrics for Report

**1. Execution Time Table**

| Configuration | Matrix Size | Kernel | Stride | Nodes | Processes | Threads | Time (ms) | Speedup | Efficiency |
|---------------|-------------|--------|--------|-------|-----------|---------|-----------|---------|------------|
| Serial        | 10000×10000 | 3×3    | 1×1    | 1     | 1         | 1       | 1204      | 1.00×   | 100%       |
| OpenMP        | 10000×10000 | 3×3    | 1×1    | 1     | 1         | 4       | 327       | 3.68×   | 92%        |
| ...           | ...         | ...    | ...    | ...   | ...       | ...     | ...       | ...     | ...        |

**2. Speedup Curves**
- Plot: Speedup vs Number of Cores (log-log scale)
- Compare: Serial, OpenMP, MPI, Hybrid
- Show ideal linear speedup line

**3. Strong Scaling Graph**
- Fixed problem size (e.g., 20000×20000)
- X-axis: Number of cores
- Y-axis: Execution time
- Multiple lines for different implementations

**4. Weak Scaling Graph**
- Problem size per core constant
- X-axis: Number of cores
- Y-axis: Execution time (should be flat for perfect scaling)

**5. Stride Impact Analysis**
- Compare computation time vs output size reduction
- Calculate: Time_ratio / Output_reduction_ratio

**6. Communication Overhead**
- Compare MPI-only vs Hybrid at same total core count
- Difference indicates communication cost

---

### Phase 6: Common Issues & Troubleshooting

#### Build Issues

**Problem:** `make` fails with missing MPI compiler
```bash
# Solution: Check loaded modules
module list
module load openmpi/5.0.5  # or appropriate version
```

**Problem:** Undefined reference to `omp_get_num_threads`
```bash
# Solution: Ensure -fopenmp flag in Makefile
# Check: grep fopenmp Makefile
```

#### Runtime Issues

**Problem:** Job killed due to memory
```bash
# Solution: Increase memory request in SLURM script
#SBATCH --mem=32G  # or --mem-per-cpu=4G
```

**Problem:** Job times out (>15 minutes)
```bash
# Solution: Reduce problem size or increase time limit
# Check: Are you testing largest possible size? Start smaller.
```

**Problem:** MPI processes hang
```bash
# Likely cause: Deadlock in MPI communication
# Debug: Add verbose output (-v flag) to see where it stops
# Check: Process count vs output dimensions (sub-communicator logic)
```

**Problem:** Results differ between implementations
```bash
# Solution: Use verification mode
mpirun -np 4 ./build/conv_stride_test -f f/f5.txt -g g/g5.txt -o expected.txt -p 2 -m

# If fails: Check stride calculation and output indexing
```

#### SLURM Issues

**Problem:** Job stuck in queue (PD state)
```bash
# Check queue status
squeue --me

# Check reason
squeue -j <JOBID> --long

# If Priority/Resources: Wait or reduce resource request
# If Dependency: Check dependency chain
```

**Problem:** Can't find output file
```bash
# Default location: Directory where sbatch was run
# Check: ls -lt | head  # Most recent files
# Specify: #SBATCH --output=/path/to/output_%j.txt
```

---

## Quick Start Checklist for HPC Testing

Use this checklist when starting on Kaya or Setonix:

### Initial Setup (Do Once)
- [ ] SSH to HPC system
- [ ] Create working directory in /scratch
- [ ] Transfer assignment2 code
- [ ] Load required modules (gcc, openmpi)
- [ ] Run `make clean && make`
- [ ] Verify build: `ls -lh build/conv_stride_test`
- [ ] Test small serial: `mpirun -np 1 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -s -t`
- [ ] Test small MPI: `mpirun -np 2 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -m -t`

### Job Submission (Repeat for Each Test)
- [ ] Create/modify SLURM script
- [ ] Set nodes, ntasks, cpus-per-task
- [ ] Set OMP_NUM_THREADS
- [ ] Specify matrix size, kernel, stride
- [ ] Set implementation flag (-s/-m/-P/none)
- [ ] Submit: `sbatch job_script.slurm`
- [ ] Monitor: `squeue --me`
- [ ] Check output: `tail -f slurm-<JOBID>.out`

### Data Collection (After Jobs Complete)
- [ ] Extract timing data: `grep "Timing" *.out`
- [ ] Calculate speedup: T_serial / T_parallel
- [ ] Calculate efficiency: Speedup / num_cores
- [ ] Create performance tables
- [ ] Generate graphs (speedup, strong scaling, weak scaling)
- [ ] Document configuration for each test

---

## Report Writing Guidance

### Required Sections (Based on Rubric)

**1. Description of Parallelism (5 marks)**
- Explain MPI distribution strategy (row-based decomposition)
- Explain OpenMP threading within each process
- Describe stride-aware distribution algorithm
- Include code snippets or pseudocode
- Reference specific functions (e.g., `mpi_distribute_matrix_stride_aware()`)

**2. Data Decomposition and Distribution (2 marks)**
- Row-major memory layout justification
- Halo exchange mechanism (for stride=1)
- Output-based distribution (for stride>1)
- Process-to-data mapping
- Load balancing approach

**3. Communication Strategy and Synchronization (3 marks)**
- MPI_Bcast for kernel distribution
- MPI_Sendrecv for halo exchange
- Contiguous buffer approach for non-contiguous data
- MPI_Gather for output collection
- Barrier synchronization points
- Sub-communicator optimization

**4. Performance Metrics and Analysis (10 marks) - MOST IMPORTANT**
- Tables showing all test configurations and timings
- Speedup graphs (vs serial baseline)
- Strong scaling analysis
- Weak scaling analysis (if tested)
- Efficiency calculations
- Comparison: OpenMP vs MPI vs Hybrid
- Multi-node vs single-node performance
- Stride impact on computation and communication
- Kernel size impact
- Discussion of overheads (communication, synchronization)
- Analysis of when each approach is optimal

**5. Cache and Memory Layout Considerations (part of parallelism)**
- Cache-line alignment (64-byte)
- Row-major traversal benefits
- SIMD vectorization in OpenMP (3×3, 5×5 optimized)
- Memory prefetching (`__builtin_prefetch`)
- False sharing prevention

**6. Effect of Stride (part of performance analysis)**
- Computation reduction vs stride (quadratic relationship)
- Communication cost vs stride
- Optimal stride for different scenarios

---

## Expected Outcomes

### What the AI Assistant Should Do on HPC

1. **Verify build works:** Run `make` successfully
2. **Run correctness tests:** Ensure provided test cases pass
3. **Execute performance matrix:** Run all priority tests
4. **Collect results:** Extract timing data systematically
5. **Calculate metrics:** Speedup, efficiency for each configuration
6. **Identify trends:** Which configuration is fastest? Where does MPI help?
7. **Generate tables:** Formatted data for report
8. **Flag anomalies:** Unexpected results that need investigation

### What Should NOT Be Done

- ❌ Don't use more than 4 nodes
- ❌ Don't exceed 15 minute runtime limit
- ❌ Don't waste Setonix CPU hours on redundant tests
- ❌ Don't skip verification - ensure correctness first
- ❌ Don't test only one configuration - need comparison
- ❌ Don't ignore local baseline results - they provide context

---

## Final Notes

**Remember:**
- Implementation is DONE and VERIFIED ✅
- Focus is 100% on PERFORMANCE ANALYSIS 🎯
- HPC systems show MPI advantage (multi-node)
- Report quality matters more than code (already working)
- Performance analysis is 10/30 marks - largest single component

**Success Criteria:**
1. Code builds with `make` on both Kaya and Setonix
2. Comprehensive performance data collected across test matrix
3. Clear speedup and efficiency analysis
4. Multi-node scaling demonstrated
5. Trade-offs between OpenMP/MPI/Hybrid clearly explained

**Questions to Answer in Report:**
- When is OpenMP better than MPI?
- When is Hybrid better than either alone?
- How does stride affect performance?
- What is the communication overhead?
- What is the optimal process×thread configuration?
- How does performance scale across multiple nodes?

---

## Document Version
- **Created:** 2025-10-01
- **Status:** Ready for HPC testing phase
- **Implementation Status:** 100% complete, all variants working
- **Next Step:** Transfer to HPC, run performance benchmarks

Good luck with the testing! 🚀
