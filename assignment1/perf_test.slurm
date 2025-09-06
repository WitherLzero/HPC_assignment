#!/bin/bash

#SBATCH --job-name=perf_test
#SBATCH --output=perf_test.out
#SBATCH --error=perf_test.err
#SBATCH --cpus-per-task=64
#SBATCH --partition=cits3402
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_student_number>@student.uwa.edu.au

# Build the project
make

# Performance testing script
# Start from 10^4 and increase by 10x each time until execution time exceeds 1 hour

echo "Starting performance test..."
echo "Matrix Size | Execution Time (seconds) | Memory Usage (MB) | Status"
echo "--------------------------------------------------------------------"
echo ""

# Initialize matrix size starting from 10^4
matrix_size=10000
max_time=3600  # 1 hour in seconds
time_limit_reached=false

while [ "$time_limit_reached" = false ]; do
    echo "Testing matrix size: ${matrix_size}x${matrix_size}"
    
    # Run the convolution test and capture execution time and memory usage
    # Use /usr/bin/time to get memory usage while running the program
    time_output=$(/usr/bin/time -v ./build/conv_test -t -H $matrix_size -W $matrix_size -kH 3 -kW 3 2>&1)
    exit_code=$?
    
    # # Extract execution time from program output (the program outputs just the time in seconds)
    # conv_time=$(echo "$time_output" | grep -E "^[0-9]+$" | tail -1)
    # if [ -z "$conv_time" ]; then
    #     conv_time="N/A"
    # fi
    conv_time=$(echo "$time_output" | head -n 1)
    
    # Extract peak memory usage from time output
    peak_memory_kb=$(echo "$time_output" | grep "Maximum resident set size" | awk '{print $6}')
    if [ -n "$peak_memory_kb" ] && [ "$peak_memory_kb" -gt 0 ]; then
        peak_memory_mb=$((peak_memory_kb / 1024))
    else
        peak_memory_mb="N/A"
    fi
    
    # Check if execution was successful
    if [ $exit_code -eq 0 ]; then
        echo "${matrix_size}x${matrix_size} | ${conv_time} | ${peak_memory_mb} | SUCCESS"
        
        # Check if we've exceeded the time limit using program's execution time
        # if [ "$conv_time" != "N/A" ] && [ "$conv_time" -gt $max_time ]; then
        #     echo "Time limit exceeded! Matrix size ${matrix_size}x${matrix_size} took ${conv_time} seconds (> 1 hour)"
        #     time_limit_reached=true
        # else
            # Increase matrix size by 10x
        matrix_size=$((matrix_size * 10))
        # fi
    else
        if [ $exit_code -eq 137 ]; then
            echo "${matrix_size}x${matrix_size} | OOM | ${peak_memory_mb} | OUT_OF_MEMORY (killed by system)"
            echo "Matrix size ${matrix_size}x${matrix_size} caused out-of-memory condition"
        else
            echo "${matrix_size}x${matrix_size} | FAILED | ${peak_memory_mb} | ERROR (exit code: $exit_code)"
            echo "Error output: $output"
        fi
        time_limit_reached=true
    fi
    
    # Safety check to prevent infinite loop
    if [ $matrix_size -gt 100000000 ]; then
        echo "Safety limit reached: matrix size too large (${matrix_size})"
        time_limit_reached=true
    fi
done

echo "Performance test completed."
echo "Final matrix size tested: ${matrix_size}x${matrix_size}"
