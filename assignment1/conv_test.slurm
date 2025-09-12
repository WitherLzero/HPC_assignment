#!/bin/bash

#SBATCH --job-name=conv_test
#SBATCH --output=conv.out
#SBATCH --error=conv.err
#SBATCH --cpus-per-task=48
#SBATCH --partition=cits3402
#SBATCH --time=01:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=24745401@student.uwa.edu.au

# Parameters
HEIGHT=${1:-500000}     # Matrix height
WIDTH=${2:-$HEIGHT}     # Matrix width
KERNEL_H=${3:-3}        # Kernel height
KERNEL_W=${4:-$KERNEL_H}  # Kernel width
MODE=${5:-0}            # 0=parallel, 1=serial

# Build the project
make

# Create results directory
mkdir -p stress_results
RESULT_FILE="stress_results/test_optimized_${HEIGHT}x${WIDTH}_${KERNEL_H}x${KERNEL_W}_${MODE}.txt"

# Start time measurement
WALL_START=$(date +%s)

echo "Test Configuration: ${HEIGHT}x${WIDTH} matrix, ${KERNEL_H}x${KERNEL_W} kernel, Mode: $([[ $MODE -eq 0 ]] && echo "Parallel" || echo "Serial")"
echo "Start time: $(date)"

# Capture the output to variables
OUTPUT=$(/usr/bin/time -v ./build/conv_test -T -H $HEIGHT -W $WIDTH -kH $KERNEL_H -kW $KERNEL_W -s $MODE 2>&1)
EXEC_STATUS=$?

# End time measurement
WALL_END=$(date +%s)
WALL_TIME=$((WALL_END - WALL_START))

# Extract execution time from program output
PROGRAM_TIME=$(echo "$OUTPUT" | head -n 1)

# Extract peak memory usage
PEAK_MEMORY_KB=$(echo "$OUTPUT" | grep "Maximum resident set size" | awk '{print $6}')
PEAK_MEMORY_MB=$((PEAK_MEMORY_KB / 1024))
PEAK_MEMORY_GB=$(echo "scale=2; $PEAK_MEMORY_MB / 1024" | bc)

# Check for OOM or other errors
ERROR_MSG=$(echo "$OUTPUT" | grep -E "Error:|error:|failed|cannot allocate memory|Out of memory|Killed" | head -3)

# Format status message
if [ $EXEC_STATUS -eq 0 ] && [ -z "$ERROR_MSG" ]; then
  STATUS="SUCCESS"
else
  STATUS="FAILED: $ERROR_MSG"
fi

# Capture SLURM error output at the end of the job
# This must be done after job completion
trap "cat conv.err >> $RESULT_FILE" EXIT

# Save detailed results to file
{
  echo "Matrix Size: ${HEIGHT}x${WIDTH}"
  echo "Kernel Size: ${KERNEL_H}x${KERNEL_W}"
  echo "Mode: $([[ $MODE -eq 0 ]] && echo "Parallel" || echo "Serial")"
  echo "Wall clock time: ${WALL_TIME} seconds"
  echo "Program ${PROGRAM_TIME:-N/A}"
  echo "Peak Memory: ${PEAK_MEMORY_MB:-0} MB (${PEAK_MEMORY_GB:-0} GB)"
  echo "Start time: $(date -d @${WALL_START})"
  echo "End time: $(date -d @${WALL_END})"
  echo ""
  echo "== Error Output =="
} > "$RESULT_FILE"

# Display summary
echo "Test completed"
echo "Wall clock time: ${WALL_TIME} seconds"
echo "Program time: ${PROGRAM_TIME:-N/A} ms"
echo "Peak Memory: ${PEAK_MEMORY_MB:-0} MB (${PEAK_MEMORY_GB:-0} GB)"
echo "Results saved to: $RESULT_FILE"
echo "End time: $(date)"
