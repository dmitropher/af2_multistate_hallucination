#!/bin/bash
#test cases

source activate SE3

#0
test_argument_string="--oligo AB+ --L 10,10 --amber_relax 0 --steps 20 --out tests/test_0 "
echo "Testing: " $test_argument_string
python AF2_multistate_hallucination.py $test_argument_string 

#1
test_argument_string="--oligo AB+ --L 10,10 --amber_relax 1 --steps 20 --out tests/test_1 "
echo "Testing: " $test_argument_string
python AF2_multistate_hallucination.py $test_argument_string 

