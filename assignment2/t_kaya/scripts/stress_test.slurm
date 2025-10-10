#!/bin/bash

#SBATCH --job-name=conv2d_stress
#SBATCH --output=stress_results/stress_%j.out
#SBATCH --error=stress_results/stress_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=8
#SBATCH --mem=1TB
#SBATCH --partition=cits3402
#SBATCH --time=00:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=24745401@student.uwa.edu.au

# CITS3402 Assignment 2 - Stress Testing
# ======================================
# Goal: Find the largest matrix that can be processed in under 15 minutes
#
# Resource allocation:
# - 2 nodes, 8 MPI tasks total, 12 threads per task = 96 cores total
# - Memory: ~1.5TB per node (no mem limit)
# - Focus on hybrid MPI+OpenMP performance

# Parameters (can be overridden when submitting)
HEIGHT=${1:-500000}      # Matrix height
WIDTH=${2:-$HEIGHT}      # Matrix width (default: square)
KERNEL_H=${3:-3}         # Kernel height
KERNEL_W=${4:-$KERNEL_H} # Kernel width
STRIDE_H=${5:-1}         # Vertical stride
STRIDE_W=${6:-1}         # Horizontal stride

# Configuration
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # Threads per MPI process
NUM_PROCESSES=$SLURM_NTASKS            # MPI processes

# Load required modules
module load gcc/14.3
module load openmpi/5.0.5

echo "============================================"
echo "STRESS TEST - Assignment 2"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "MPI Processes: $NUM_PROCESSES"
echo "Threads per process: $OMP_NUM_THREADS"
echo "Total cores: $((NUM_PROCESSES * OMP_NUM_THREADS))"
echo "============================================"
echo ""

# Build the project
echo "Building project..."
make clean && make
if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi
echo "Build completed."
echo ""

# Create results directory
mkdir -p t_kaya/stress_results
RESULT_FILE="t_kaya/stress_results/stress_${HEIGHT}x${WIDTH}_k${KERNEL_H}x${KERNEL_W}_s${STRIDE_H}x${STRIDE_W}_${SLURM_JOB_ID}.txt"

echo "Test Configuration:"
echo "  Matrix: ${HEIGHT}x${WIDTH}"
echo "  Kernel: ${KERNEL_H}x${KERNEL_W}"
echo "  Stride: ${STRIDE_H}x${STRIDE_W}"
echo "  Mode: Hybrid MPI+OpenMP"
echo ""

# Start wall clock timer
WALL_START=$(date +%s)
echo "Start time: $(date)"
echo ""

# Run the stress test with /usr/bin/time to capture memory usage
# Use -m flag for MPI mode (hybrid MPI+OpenMP)
# Use -T flag for timing in seconds
OUTPUT=$(/usr/bin/time -v mpirun -np $NUM_PROCESSES ./build/conv_stride_test \
    -H $HEIGHT -W $WIDTH \
    -kH $KERNEL_H -kW $KERNEL_W \
    -sH $STRIDE_H -sW $STRIDE_W \
    -m -T 2>&1)

EXEC_STATUS=$?

# End wall clock timer
WALL_END=$(date +%s)
WALL_TIME=$((WALL_END - WALL_START))

echo ""
echo "============================================"
echo "RESULTS"
echo "============================================"

# Extract timing breakdown (new format with separate convolution/gather/total)
CONV_TIME=$(echo "$OUTPUT" | grep "Timing - Convolution:" | grep -oP '\d+\.\d+')
GATHER_TIME=$(echo "$OUTPUT" | grep "Timing - Gather:" | grep -oP '\d+\.\d+')
TOTAL_TIME=$(echo "$OUTPUT" | grep "Timing - Total:" | grep -oP '\d+\.\d+')

# Fallback: if old format (single line), use that
if [ -z "$CONV_TIME" ]; then
    CONV_TIME=$(echo "$OUTPUT" | grep "Timing - Convolution with stride" | grep -oP '\d+\.\d+')
fi

# Extract peak memory usage from /usr/bin/time output
PEAK_MEMORY_KB=$(echo "$OUTPUT" | grep "Maximum resident set size" | awk '{print $6}')
PEAK_MEMORY_MB=$((PEAK_MEMORY_KB / 1024))
PEAK_MEMORY_GB=$(echo "scale=2; $PEAK_MEMORY_MB / 1024" | bc)

# Check for errors
ERROR_MSG=$(echo "$OUTPUT" | grep -E "Error:|error:|failed|cannot allocate memory|Out of memory|Killed" | head -3)

# Determine status
if [ $EXEC_STATUS -eq 0 ] && [ -z "$ERROR_MSG" ]; then
    STATUS="SUCCESS"
else
    STATUS="FAILED"
    if [ -n "$ERROR_MSG" ]; then
        STATUS="FAILED: $ERROR_MSG"
    fi
fi

# Display summary
echo "Status: $STATUS"
echo "Wall clock time: ${WALL_TIME} seconds ($(echo "scale=2; $WALL_TIME / 60" | bc) minutes)"
if [ -n "$CONV_TIME" ]; then
    echo "Convolution time: ${CONV_TIME} seconds"
else
    echo "Convolution time: N/A"
fi
if [ -n "$GATHER_TIME" ]; then
    echo "Gather time: ${GATHER_TIME} seconds"
fi
if [ -n "$TOTAL_TIME" ]; then
    echo "Total parallel time: ${TOTAL_TIME} seconds"
fi
echo "Peak Memory: ${PEAK_MEMORY_MB:-0} MB (${PEAK_MEMORY_GB:-0} GB)"
echo ""

# Save detailed results to file
{
    echo "============================================"
    echo "STRESS TEST RESULTS"
    echo "============================================"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Date: $(date)"
    echo ""
    echo "Configuration:"
    echo "  Matrix: ${HEIGHT}x${WIDTH}"
    echo "  Kernel: ${KERNEL_H}x${KERNEL_W}"
    echo "  Stride: ${STRIDE_H}x${STRIDE_W}"
    echo "  Nodes: $SLURM_JOB_NUM_NODES"
    echo "  MPI Processes: $NUM_PROCESSES"
    echo "  OpenMP Threads per process: $OMP_NUM_THREADS"
    echo "  Total cores: $((NUM_PROCESSES * OMP_NUM_THREADS))"
    echo ""
    echo "Results:"
    echo "  Status: $STATUS"
    echo "  Wall clock time: ${WALL_TIME} seconds ($(echo "scale=2; $WALL_TIME / 60" | bc) minutes)"
    if [ -n "$CONV_TIME" ]; then
        echo "  Convolution time: ${CONV_TIME} seconds"
    else
        echo "  Convolution time: N/A"
    fi
    if [ -n "$GATHER_TIME" ]; then
        echo "  Gather time: ${GATHER_TIME} seconds"
    else
        echo "  Gather time: N/A (single process or no gather needed)"
    fi
    if [ -n "$TOTAL_TIME" ]; then
        echo "  Total parallel time: ${TOTAL_TIME} seconds"
    else
        echo "  Total parallel time: N/A"
    fi
    echo "  Peak Memory: ${PEAK_MEMORY_MB:-0} MB (${PEAK_MEMORY_GB:-0} GB)"
    echo "  Start time: $(date -d @${WALL_START})"
    echo "  End time: $(date -d @${WALL_END})"
    echo ""
    echo "============================================"
    echo "FULL OUTPUT"
    echo "============================================"
    echo "$OUTPUT"
    echo ""
    echo "============================================"
    echo "SLURM ERROR OUTPUT"
    echo "============================================"
} > "$RESULT_FILE"

# Append SLURM error output
cat "stress_results/stress_${SLURM_JOB_ID}.err" >> "$RESULT_FILE" 2>/dev/null

echo "Results saved to: $RESULT_FILE"
echo ""
echo "End time: $(date)"
echo "============================================"

# Exit with the status of the main program
exit $EXEC_STATUS
