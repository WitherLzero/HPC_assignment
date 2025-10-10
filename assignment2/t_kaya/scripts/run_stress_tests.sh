#!/bin/bash

# CITS3402 Assignment 2 - Automated Stress Testing
# =================================================
# This script submits multiple stress tests with increasing matrix sizes
# to find the maximum problem size that can be handled in under 15 minutes

echo "============================================"
echo "Assignment 2 - Stress Test Suite"
echo "============================================"
echo ""

# Create results directory
mkdir -p t_kaya/stress_results

# Test configurations - fixed 3x3 kernel
KERNEL_SIZE=3

# Stride 1 tests - can push to 450K 
STRIDE1_MATRIX_SIZES=(
    350000    # 350K x 350K 
    400000    # 400K x 400K
    450000    # 450K x 450K
)

# Stride 2 tests - limited to 400K due to calculation overhead
STRIDE2_MATRIX_SIZES=(
    360000    # 360K x 360K
    380000    # 380K x 380K
    400000    # 400K x 400K
)

echo "Kernel size: ${KERNEL_SIZE}x${KERNEL_SIZE} (fixed)"
echo ""

echo "Stride 1 test sizes (up to 600K):"
for size in "${STRIDE1_MATRIX_SIZES[@]}"; do
    echo "  - ${size}x${size}"
done
echo ""

echo "Stride 2 test sizes (up to 400K):"
for size in "${STRIDE2_MATRIX_SIZES[@]}"; do
    echo "  - ${size}x${size}"
done
echo ""

# Ask for confirmation
read -p "Submit all stress tests? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Submit jobs
JOB_COUNT=0

# Submit stride 1 tests
echo "Submitting stride 1 tests..."
for size in "${STRIDE1_MATRIX_SIZES[@]}"; do
    echo "  Matrix ${size}x${size}, Kernel ${KERNEL_SIZE}x${KERNEL_SIZE}, Stride 1x1"

    JOB_ID=$(sbatch --parsable stress_test.slurm $size $size $KERNEL_SIZE $KERNEL_SIZE 1 1)

    if [ $? -eq 0 ]; then
        echo "    Job ID: $JOB_ID"
        ((JOB_COUNT++))
    else
        echo "    Failed to submit!"
    fi

    sleep 0.5
done

echo ""
echo "Submitting stride 2 tests..."
# Submit stride 2 tests
for size in "${STRIDE2_MATRIX_SIZES[@]}"; do
    echo "  Matrix ${size}x${size}, Kernel ${KERNEL_SIZE}x${KERNEL_SIZE}, Stride 2x2"

    JOB_ID=$(sbatch --parsable stress_test.slurm $size $size $KERNEL_SIZE $KERNEL_SIZE 2 2)

    if [ $? -eq 0 ]; then
        echo "    Job ID: $JOB_ID"
        ((JOB_COUNT++))
    else
        echo "    Failed to submit!"
    fi

    sleep 0.5
done

echo ""
echo "============================================"
echo "Submitted $JOB_COUNT stress test jobs"
echo "============================================"
echo ""
echo "Monitor job status with: squeue -u $USER"
echo "View results in: stress_results/"
echo ""
echo "To view a specific job's output:"
echo "  cat stress_results/stress_<JOBID>.out"
echo ""
echo "To check for successful completions:"
echo "  grep -r \"Status: SUCCESS\" stress_results/"
echo ""