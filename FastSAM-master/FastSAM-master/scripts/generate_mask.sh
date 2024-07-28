#!/bin/bash

python fast_sam_generate_mask.py --training_idx 0 &
python fast_sam_generate_mask.py --training_idx 1 &
python fast_sam_generate_mask.py --training_idx 2 &
python fast_sam_generate_mask.py --training_idx 3 &
python fast_sam_generate_mask.py --training_idx 4 &
python fast_sam_generate_mask.py --training_idx 5 &
python fast_sam_generate_mask.py --training_idx 6 &
python fast_sam_generate_mask.py --training_idx 7 &

wait

python fast_sam_generate_mask.py --training_idx 8 &
python fast_sam_generate_mask.py --training_idx 9 &
python fast_sam_generate_mask.py --training_idx 10 &
python fast_sam_generate_mask.py --training_idx 11 &
python fast_sam_generate_mask.py --training_idx 12 &
python fast_sam_generate_mask.py --training_idx 13 &
python fast_sam_generate_mask.py --training_idx 14 &