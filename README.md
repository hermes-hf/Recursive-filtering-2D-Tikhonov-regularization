# Recursive-filtering-2D-Tikhonov-regularization

Notebook folder
- Main jupyter notebook used for analysis is provided in /notebooks/notebook.ipynb
- The file notebooks/filterError_threads.jl was used to compute the precision results faster in a threaded environment

Data folder
- The data used for analysis is in the Data folder

Source folder
- C++ Deriche implementation provided in deriche_block_filter.cpp
- C++ FFTW implementation provided in fftwCode.cpp
- The binaries can be generated using the makefile
- Binaries can be run by executing "deriche_block_filter imagesize" or "fftwCode imagesize" where imagesize is the desired dimension for the image (square dimensions only)


Extern folder
- If planning to run the C++ Deriche filter implementation, the user should download the Eigen library in the extern/eigen folder (link provided in the extern folder's readme)

Supplementary material
- We also provide the supplementary material for our paper