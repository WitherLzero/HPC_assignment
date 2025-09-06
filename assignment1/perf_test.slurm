#!/bin/bash

#SBATCH --job-name=perf_test
#SBATCH --output=perf_test.out
#SBATCH --error=perf_test.err
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --partition=cits3402
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_student_number>@student.uwa.edu.au

# Build the project
make

# Performance testing script
# Start from 10^4 and increase by 10x each time until execution time exceeds 1 hour

echo "Starting performance test..."
echo "Matrix Size | Execution Time (seconds) | Status"
echo "-----------------------------------------------"

# Initialize matrix size starting from 10^4
matrix_size=10000
max_time=3600  # 1 hour in seconds
time_limit_reached=false

while [ "$time_limit_reached" = false ]; do
    echo "Testing matrix size: ${matrix_size}x${matrix_size}"
    
    # Run the convolution test and capture the execution time
    # Using the hint: ./build/conv_test -T -H 20000 -W 20000 -kH 3 -kW 3
    start_time=$(date +%s)
    
    # Run the test and capture both output and timing
    output=$(./build/conv_test -T -H $matrix_size -W $matrix_size -kH 3 -kW 3 2>&1)
    exit_code=$?
    
    end_time=$(date +%s)
    actual_time=$((end_time - start_time))
    
    # Extract the actual convolution time from the output if available
    conv_time=$(echo "$output" | grep -o "Convolution time: [0-9.]* seconds" | grep -o "[0-9.]*" | tail -1)
    if [ -z "$conv_time" ]; then
        conv_time="N/A"
    fi
    
    # Check if execution was successful
    if [ $exit_code -eq 0 ]; then
        echo "${matrix_size}x${matrix_size} | ${conv_time} | SUCCESS"
        
        # Check if we've exceeded the time limit
        if [ $actual_time -gt $max_time ]; then
            echo "Time limit exceeded! Matrix size ${matrix_size}x${matrix_size} took ${actual_time} seconds (> 1 hour)"
            time_limit_reached=true
        else
            # Increase matrix size by 10x
            matrix_size=$((matrix_size * 10))
        fi
    else
        echo "${matrix_size}x${matrix_size} | FAILED | ERROR (exit code: $exit_code)"
        echo "Error output: $output"
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
