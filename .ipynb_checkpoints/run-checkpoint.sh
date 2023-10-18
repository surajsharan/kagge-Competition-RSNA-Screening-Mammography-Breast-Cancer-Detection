#!/bin/bash

# Set CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8,11,12,13,15

# Loop over values of i from 0 to 5
for i in {0..4}; do
    # Run train.py with i as an argument
    python -u train.py $i

#     # Find and kill running Python processes
#     pids=$(pgrep -f "python train.py $i")
#     if [ ! -z "$pids" ]; then
#         echo "Killing processes for i=$i: $pids"
#         kill $pids
#     fi

#     # Clean up GPU memory
#     python -c "import torch; torch.cuda.empty_cache()"
done
