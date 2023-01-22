ALL:  fftwCode deriche_block_filter deriche_block_filter_openmp tikhonov_recursive_filter

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

tikhonov_recursive_filter: edge_aware_tikhonov_filter.cpp
	g++ edge_aware_tikhonov_filter.cpp -I ../../code/extern/eigen/ -o recursiveTikhonov -O3 -march=native

clean:
	rm -f recursiveTikhonov recursiveTikhonov fftwCode deriche_block_filter deriche_block_filter_openmp

.PHONY: ALL clean
