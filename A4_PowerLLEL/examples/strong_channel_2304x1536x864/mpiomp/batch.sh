#!/bin/bash

# $1: test path
# $2: executable
# $3: array of case names
# $4: test rounds
# $5: nnodes (-N)
# $6: ntasks (-n)
# $7: cpu per tasks (-c)
run_test_case(){
cases_arr=$3

for icase in ${cases_arr[*]}
do
  cd $1/$icase

  for ((i=1; i<=$4; i++))
  do
    echo ">>> Start  test "$2"-"$icase" Round "$i": -N "$5" -n "$6" -c "$7
    OMP_NUM_THREADS=$7 srun -N $5 -n $6 -c $7 $2 2>&1 | tee run.$i.log
    if [ -f "timing.summary" ];then
      mv timing.summary timing.summary.$i
    fi
    echo ">>> Finish test "$2"-"$icase" Round "$i"!"
  done

done
}

base=$(pwd)
exe=../../bin/PowerLLEL_cpu.mpiomp
nrounds=3

# 192 Nodes, 4608 Cores
cases=(32x12)
run_test_case $base $exe "${cases[*]}" $nrounds 192 384 12

# 96 Nodes, 2304 Cores
cases=(16x12)
run_test_case $base $exe "${cases[*]}" $nrounds 96 192 12

nrounds=2

# 48 Nodes, 1152 Cores
cases=(8x12)
run_test_case $base $exe "${cases[*]}" $nrounds 48 96 12

# 24 Nodes, 576 Cores
cases=(4x12)
run_test_case $base $exe "${cases[*]}" $nrounds 24 48 12

# 12 Nodes, 288 Cores
cases=(2x12)
run_test_case $base $exe "${cases[*]}" $nrounds 12 24 12
