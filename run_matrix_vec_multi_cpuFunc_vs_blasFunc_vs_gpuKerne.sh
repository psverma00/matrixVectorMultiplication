nvcc matrix_vec_multi_cpuFunc_vs_blasFunc_vs_gpuKernel.cu  ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a  -lpthread -lm -ldl

./a.out
