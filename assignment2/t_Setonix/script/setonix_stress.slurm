#!/bin/bash

#SBATCH --job-name=conv2d_stress_setonix
#SBATCH --account=courses0101
#SBATCH --partition=work

# ========================================================================
# RESOURCE ALLOCATION - Choose ONE configuration
# ========================================================================

# OPTION 1: 2 nodes × 256 logical cores = 512 cores total (ACTIVE)
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16


#SBATCH --time=00:15:00
#SBATCH --output=stress_results/setonix_stress_%j.out
#SBATCH --error=stress_results/setonix_stress_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=24745401@student.uwa.edu.au

# CITS3402 Assignment 2 - Setonix Stress Testing
# ==============================================
# Goal: Find the largest matrix that can be processed in under 15 minutes
#
# OPTION 1: 2 nodes, 16 MPI × 32 threads = 512 logical cores


# Parameters (can be overridden when submitting)
HEIGHT=${1:-500000}
WIDTH=${2:-$HEIGHT}
KERNEL_H=${3:-3}
KERNEL_W=${4:-$KERNEL_H}
STRIDE_H=${5:-1}
STRIDE_W=${6:-1}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NUM_PROCESSES=$SLURM_NTASKS
TOTAL_CORES=$((NUM_PROCESSES * OMP_NUM_THREADS))

echo "============================================"
echo "STRESS TEST - Setonix"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES (AMD EPYC 7763)"
echo "Node list: $SLURM_NODELIST"
echo "MPI Processes: $NUM_PROCESSES"
echo "Threads per process: $OMP_NUM_THREADS"
echo "Total cores: $TOTAL_CORES"
echo "============================================"
echo ""

# Load Cray environment
module load PrgEnv-gnu
module load cray-mpich/8.1.32

echo "Loaded modules:"
module list 2>&1
echo ""

# Update Makefile for Cray
echo "Updating Makefile for Cray environment..."
cp Makefile Makefile.backup
sed -i 's/CC = mpicc/CC = cc/' Makefile

echo "Building project..."
make clean && make
BUILD_STATUS=$?

mv Makefile.backup Makefile

if [ $BUILD_STATUS -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi
echo "Build completed."
echo ""

mkdir -p stress_results
RESULT_FILE="stress_results/setonix_stress_${HEIGHT}x${WIDTH}_k${KERNEL_H}x${KERNEL_W}_s${STRIDE_H}x${STRIDE_W}_${SLURM_JOB_ID}.txt"

echo "Test Configuration:"
echo "  Matrix: ${HEIGHT}x${WIDTH}"
echo "  Kernel: ${KERNEL_H}x${KERNEL_W}"
echo "  Stride: ${STRIDE_H}x${STRIDE_W}"
echo "  Mode: Hybrid MPI+OpenMP"
echo ""

WALL_START=$(date +%s)
echo "Start time: $(date)"
echo ""

# Run the test and capture output
OUTPUT=$(srun -n $NUM_PROCESSES -c $OMP_NUM_THREADS ./build/conv_stride_test \
    -H $HEIGHT -W $WIDTH \
    -kH $KERNEL_H -kW $KERNEL_W \
    -sH $STRIDE_H -sW $STRIDE_W \
    -m -T 2>&1)

EXEC_STATUS=$?
WALL_END=$(date +%s)
WALL_TIME=$((WALL_END - WALL_START))

# Get memory usage from SLURM sacct (after job completes)
# Wait for SLURM accounting to update
sleep 5

# Try multiple methods to get memory info
# Method 1: sacct with MaxRSS for all tasks
SLURM_MEM_RAW=$(sacct -j ${SLURM_JOB_ID}.batch --format=MaxRSS --noheader 2>/dev/null | sort -h | tail -1 | tr -d ' ')

# Method 2: If that fails, try getting all steps
if [ -z "$SLURM_MEM_RAW" ] || [ "$SLURM_MEM_RAW" = "0" ]; then
    SLURM_MEM_RAW=$(sacct -j ${SLURM_JOB_ID} --format=MaxRSS --noheader 2>/dev/null | grep -v "^$" | sort -h | tail -1 | tr -d ' ')
fi

PEAK_MEMORY_KB=0
PEAK_MEMORY_MB=0
PEAK_MEMORY_GB=0

# Parse SLURM memory format (e.g., "1234K" or "123M" or "12G" or just "1234")
if [ -n "$SLURM_MEM_RAW" ]; then
    # Extract number and unit
    if [[ $SLURM_MEM_RAW =~ ^([0-9]+)([KMG]?)$ ]]; then
        MEM_VALUE=${BASH_REMATCH[1]}
        MEM_UNIT=${BASH_REMATCH[2]}

        case $MEM_UNIT in
            K|k) PEAK_MEMORY_KB=$MEM_VALUE ;;
            M|m) PEAK_MEMORY_KB=$((MEM_VALUE * 1024)) ;;
            G|g) PEAK_MEMORY_KB=$((MEM_VALUE * 1024 * 1024)) ;;
            "") PEAK_MEMORY_KB=$MEM_VALUE ;;  # Assume KB if no unit
        esac

        PEAK_MEMORY_MB=$((PEAK_MEMORY_KB / 1024))
        if [ $PEAK_MEMORY_MB -gt 0 ]; then
            PEAK_MEMORY_GB=$(echo "scale=2; $PEAK_MEMORY_MB / 1024" | bc)
        fi
    fi
fi

# If still no memory data, note it
if [ $PEAK_MEMORY_KB -eq 0 ]; then
    MEMORY_NOTE=" (SLURM accounting not available yet - check sacct later)"
else
    MEMORY_NOTE=""
fi

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

ERROR_MSG=$(echo "$OUTPUT" | grep -E "Error:|error:|failed|cannot allocate memory|Out of memory|Killed" | head -3)

if [ $EXEC_STATUS -eq 0 ] && [ -z "$ERROR_MSG" ]; then
    STATUS="SUCCESS"
else
    STATUS="FAILED"
    if [ -n "$ERROR_MSG" ]; then
        STATUS="FAILED: $ERROR_MSG"
    fi
fi

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
echo "Peak Memory: ${PEAK_MEMORY_MB:-0} MB (${PEAK_MEMORY_GB:-0} GB)${MEMORY_NOTE}"
echo ""

# Show sacct command for manual check
echo "To check memory usage later, run:"
echo "  sacct -j ${SLURM_JOB_ID} --format=JobID,MaxRSS,Elapsed"
echo ""

# Save results
{
    echo "============================================"
    echo "STRESS TEST RESULTS - Setonix"
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
    echo "  OpenMP Threads: $OMP_NUM_THREADS"
    echo "  Total cores: $TOTAL_CORES"
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
} > "$RESULT_FILE"

# Append error output
cat "stress_results/setonix_stress_${SLURM_JOB_ID}.err" >> "$RESULT_FILE" 2>/dev/null

echo "Results saved to: $RESULT_FILE"
echo "End time: $(date)"
echo "============================================"

exit $EXEC_STATUS
