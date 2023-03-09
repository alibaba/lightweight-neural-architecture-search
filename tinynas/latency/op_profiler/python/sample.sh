# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

nbits="${2:-"8"}"

DataFile="sample.int8.txt"

LD_LIBRARY_PATH=sample_lib/:$LD_LIBRARY_PATH
insmod /system/init/soc-nna.ko

if [ ! -e $DataFile ]; then
        echo "Fail to open file"
        exit 1
fi

iter=0

# {Conv_type, Batch, In_C, In_H, In_W, Out_C, Kernel, Stride, ElmtFused} Latency
while read line
do
        [ -z "$line" ] && continue
        set -- $line
        iter=$((iter+1))
        echo "iter-$iter: $line"

        Conv_type=$(($1))
        Batch=$(($2))
        In_C=$(($3))
        In_H=$(($4))
        In_W=$(($5))
        Out_C=$(($6))
        Kernel=$(($7))
        Stride=$(($8))
        # ElmtFused=$(($9))
        # Latency=$((${10}))

        # call venus_eval_test_uclibc, an example for profiling each convolution.
        if [ $iter -gt 3539 ]
        then
                ./venus_eval_test_uclibc $In_H $In_W $In_C $nbits $Out_C $nbits $Kernel $Kernel $Stride $nbits
        fi
done < $DataFile
