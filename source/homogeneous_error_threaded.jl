using Pkg
using Plots
using Statistics
using LinearAlgebra
using ForwardDiff
using FFTW
using SpecialFunctions
using ColorSchemes
using DelimitedFiles
using Images
using SparseArrays
using BenchmarkTools
using DelimitedFiles
using JLD

include("tikhonov2d_iterative.jl");
include("dericheFilter.jl");


function main(skip = 0)
    
    function fftfilter2D(img,lx,ly)
        channels, N, M = size(img)
        N2 = 2*N; #increase image size to avoid circular filtering
        M2 = 2*M
        
        g = zeros(ComplexF64, channels, N2, M2)
        g[:,1:N,1:M] = img #embbed image
        
        g = fft(g)
        for i =1:N2, j=1:M2
            w = 2*pi*(i-1)/N2
            xi = 2*pi*(j-1)/M2
            g[:,i,j] = g[:,i,j]/(1+2*lx*(1-cos(w))+2*ly*(1-cos(xi)))
        end
        
        return min.(max.(real.(ifft(g)),0),1)[:,1:N,1:M] #clamp values in [0,1]
    end;
    
   
    
    data = readdir("../data/"; join=true)
    max_ngaussian = 6
    min_ngaussian = 3
    numL = 10
    nimgs = length(data)
    lambdas = LinRange(10,100,numL)

    errs1 = zeros(max_ngaussian-min_ngaussian,numL,numL,nimgs)
    errs2 = zeros(max_ngaussian-min_ngaussian,numL,numL,nimgs)
    psnrs = zeros(max_ngaussian-min_ngaussian,numL,numL,nimgs) .+ 1e5

    ij_plan = collect(vec([(i,j) for i = 1:numL, j = 1:numL]))
    filter!(Base.splat((i,j) -> (j <= i)), ij_plan)

    if skip == 0    
        for g = 1: max_ngaussian-min_ngaussian  
            ng = g + min_ngaussian
            print("numgaussians = ", ng, "\n");

            Threads.@threads for (i,j) in collect(ij_plan)
                    lx = lambdas[i]
                    ly = lambdas[j]

                    for (ind,path) in collect(enumerate(data))
                        
                        img = load(path) .|>RGB{Float64}  |>channelview
                        fftFiltered = fftfilter2D(img,lx,ly)
                        gaussianFiltered = recursiveFilter2D(img, lx,ly, ng)

                        se = norm(fftFiltered - gaussianFiltered,2) #squared error
                        mse = se/length(img) #mean squared error

                        norm1 = norm(fftFiltered - gaussianFiltered,1)/norm(fftFiltered,1) #relative l1 error
                        norm2 = se/norm(fftFiltered,2) #relative l2 error
                        psnrval = assess_psnr(fftFiltered, gaussianFiltered,1)

                        errs1[g,i,j,ind] = norm1
                        errs2[g,i,j,ind] = norm2
                        psnrs[g,i,j,ind] = psnrval

                        errs1[g,j,i,ind] = norm1
                        errs2[g,j,i,ind] = norm2
                        psnrs[g,j,i,ind] = psnrval
                    end
            end

        end


        save("../homog_error/err1.jld", "data", errs1)
        save("../homog_error/err2.jld", "data", errs2)
        save("../homog_error/psnr.jld", "data", psnrs)
    else
        print("skipping first optimization\n");
    end

    #RUN WITH NONLINEAR OPTIMIZATION
    data = readdir("../data/"; join=true)
    max_ngaussian = 6
    min_ngaussian = 3
    numL = 10
    nimgs = length(data)
    lambdas = LinRange(10,100,numL)

    errs1 = zeros(max_ngaussian-min_ngaussian,numL,numL,nimgs)
    errs2 = zeros(max_ngaussian-min_ngaussian,numL,numL,nimgs)
    psnrs = zeros(max_ngaussian-min_ngaussian,numL,numL,nimgs) .+ 1e5
    ij_plan = collect(vec([(i,j) for i = 1:numL, j = 1:numL]))
    filter!(Base.splat((i,j) -> (j <= i)), ij_plan)

    #MUST PRECOMPUTE PARAMETERS: THREADING IN PYCALL GIVES ERROR
    Params = [[] for g =1:max_ngaussian-min_ngaussian, i=1:numL,j=1:numL]

    for g = 1: max_ngaussian-min_ngaussian  
        ng = g + min_ngaussian

        for (i,j) in collect(ij_plan)
            lx = lambdas[i]
            ly = lambdas[j]
            A, Sx, Sy, c0 = gaussianKernelFit(lx,ly,ng);
            A, Sx, Sy, c0 = improve(A,Sx,Sy,c0,lx,ly);
            Params[g,i,j] = [(A,Sx,Sy,c0)]

        end

    end


    for g = 1: max_ngaussian-min_ngaussian  
        ng = g + min_ngaussian
        print("numgaussians = ", ng, "\n");

        Threads.@threads for (i,j) in collect(ij_plan)
                lx = lambdas[i]
                ly = lambdas[j]

                for (ind,path) in collect(enumerate(data))
                    
                    img = load(path) .|>RGB{Float64}  |>channelview
                    fftFiltered = fftfilter2D(img,lx,ly)
                    gaussianFiltered = recursiveFilter2DCustom(img, Params[g,i,j])

                    se = norm(fftFiltered - gaussianFiltered,2) #squared error
                    mse = se/length(img) #mean squared error

                    norm1 = norm(fftFiltered - gaussianFiltered,1)/norm(fftFiltered,1) #relative l1 error
                    norm2 = se/norm(fftFiltered,2) #relative l2 error
                    psnrval = assess_psnr(fftFiltered, gaussianFiltered,1)

                    errs1[g,i,j,ind] = norm1
                    errs2[g,i,j,ind] = norm2
                    psnrs[g,i,j,ind] = psnrval

                    errs1[g,j,i,ind] = norm1
                    errs2[g,j,i,ind] = norm2
                    psnrs[g,j,i,ind] = psnrval
                end
        end

    end
    save("../homog_error/improv_err1.jld", "data", errs1)
    save("../homog_error/improv_err2.jld", "data", errs2)
    save("../homog_error/improv_psnr.jld", "data", psnrs)

end

main()