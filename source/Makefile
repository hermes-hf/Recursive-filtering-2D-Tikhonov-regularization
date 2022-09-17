ALL:  fftwCode deriche_block_filter deriche_block_filter_openmp

CFLAGS := -Wno-unknown-pragmas -Ofast -march=native -Wall -pedantic \
	-DNDEBUG \
	-DEIGEN_NO_DEBUG
#	-DEIGEN_MAX_ALIGN_BYTES=64 \
#	-DEIGEN_DONT_PARALLELIZE=1 \
#	-DEIGEN_UNALIGNED_VECTORIZE=1 \
#	-DEIGEN_FAST_MATH=1 \
#	-DEIGEN_UNROLLING_LIMIT=110 \
#	-DEIGEN_STACK_ALLOCATION_LIMIT=524288


deriche_block_filter: deriche_block_filter.cpp
	g++ -I ../../code/extern/eigen/ -o "$@" "$^" -Ofast -march=native

deriche_block_filter_openmp: deriche_block_filter.cpp
	g++ -I ../../code/extern/eigen/ -o "$@" "$^" -Ofast -march=native -fopenmp

fftwCode: fftwCode.cpp
	g++ -I ../../code/extern/eigen/ -o "$@" "$^" -Ofast -march=native -lfftw3f

clean:
	rm -f tikhonov_recursive_filter_eigen tikhonov_recursive_filter tikhonov_recursive_filter_openmp tikhonov_recursive_filter_eigen_openmp avx_filter avx_filter_openmp fftwCode deriche_block_filter deriche_block_filter_openmp

.PHONY: ALL clean
