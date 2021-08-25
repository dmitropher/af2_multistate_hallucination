#!/bin/bash
#SBATCH -p medium 
#SBATCH -o slr_af2h_tests_%A_%a.log
#SBATCH -c 2 
#SBATCH -p gpu 
#SBATCH --gres=gpu
#SBATCH --exclude=gpu30,gpu[11-20],gpu[3-10,25-27],dig[61-64]
#SBATCH --mem=12g
#test cases

source activate SE3

#0
test_argument_string="--oligo AB+ --L 10,10 --amber_relax 0 --steps 10 --out tests/test_0 "
echo "Testing: " $test_argument_string
python AF2_multistate_hallucination.py $test_argument_string 

#1
test_argument_string="--oligo AB+ --L 10,10 --amber_relax 1 --steps 10 --out tests/test_1 "
echo "Testing: " $test_argument_string
python AF2_multistate_hallucination.py $test_argument_string 

#2
test_argument_string="--oligo AB+,AA-,BB- --L 10,10 --amber_relax 1 --steps 10 --out tests/test_2 "
echo "Testing: " $test_argument_string
python AF2_multistate_hallucination.py $test_argument_string 

#3 - dual tmalign loss with template pdb and .af2h file
test_argument_string="--oligo A+ --loss dual_tmalign --update tests/test3.af2h --template tests/test3.pdb --steps 10 --out tests/test_3 "
echo "Testing: " $test_argument_string
python AF2_multistate_hallucination.py $test_argument_string 

