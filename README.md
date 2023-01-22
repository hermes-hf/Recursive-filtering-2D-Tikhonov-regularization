# Recursive-filtering-2D-Tikhonov-regularization

Notebook folder
- Main jupyter notebooks used for analysis are provided in /notebooks
- The file source/homogeneous_error_threaded.jl was used to compute the precision results faster in a threaded environment

Data folder
- The images used for analysis are in the Data folder

Source folder
- C++ Deriche implementation provided in deriche_block_filter.cpp
- C++ FFTW implementation provided in fftwCode.cpp
- C++ BICGSTAB and CG (eigen) implementaitons provided in edge_aware_tikhonov_filter.cpp
- The binaries can be generated using the makefile
- Binaries can be run by executing "deriche_block_filter imagesize" or "fftwCode imagesize" where imagesize is the desired dimension for the image (square dimensions only)


Extern folder
- If planning to run the C++ Deriche filter implementation, the user should download the Eigen library in the extern/eigen folder (link provided in the extern folder's readme)

Supplementary material
- We also provide the supplementary material for our paper appendix.pdf