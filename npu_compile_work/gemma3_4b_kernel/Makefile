
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=gemma3_4b_kernel
targetname=gemma3_4b_kernel
kernels=mm

# Gemma3 4B dimensions
M?=256
K?=128
N?=128

# Single core dimensions
m?=32
k?=32
n?=32

b_col_maj?=0

kernels=mm_${m}x${k}x${n}
aieargs+=-m $m -k $k -n $n --b-col-maj ${b_col_maj}
runargs+=--b_col_maj ${b_col_maj}
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}

buffer_aloc_flag=basic-sequential

include ${srcdir}/../single_core/Makefile

# Override to use our kernel
build/mm_${m}x${k}x${n}.o: ${srcdir}/../single_core/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}
