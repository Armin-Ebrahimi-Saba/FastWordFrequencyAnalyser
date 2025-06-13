#!/bin/bash

# Command to run the executable
executable=$1

# Number of times to run the executable
num_runs=1

# Variable to store the sum of outputs
sum_outputs=0

# Loop to run the executable multiple times
for ((i=1; i<=$num_runs; i++)); do
    $executable
    # Run the executable and capture the output
    output=$?
    echo $output

    # Perform any necessary processing on the output
    # For simplicity, let's assume the output is an integer value

    # Add the output to the sum
    sum_outputs=$((sum_outputs + output))
done

# Calculate the average of the outputs
average=$((sum_outputs / num_runs))

# Print the average
echo "Average of outputs: $average"

