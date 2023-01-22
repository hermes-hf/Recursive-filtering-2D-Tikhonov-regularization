using Pkg
using DSP
using FFTW
import PyPlot as py
using Images
using Statistics
using Polynomials
using ColorSchemes
using OffsetArrays
using LinearAlgebra
using ImageFiltering
using Interact
using Plots
using Distributions
using IterativeSolvers
using SparseArrays

#-----------------------------------------------------------------------------------------------------
#Utility ---------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

function displayimg(img)
    #imresize(colorview(Gray,10 .*imadjustintensity(abs.(img))), ratio=4.0) 
     py.imshow(img, cmap=:CMRmap);
end;
 
#-----------------------------------------------------------------------------------------------------
#Linearizing -----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

function cm(x,n)
    return mod(x-1,n) +1
end


#-----------------------------------------------------------------------------------------------------
#Sparse version---------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

#values lx ly to use according to image scale (avoid excessive smoothing)
function getTikhonovMatrixSparse(img,lx,ly; homog=0, max_l = 30, err = 1e-10)
    n,m = size(img)    
    Wx = zeros(n,m)
    Wy = zeros(n,m)
    
    function imf(x,y)
        if x>n || x<1 || y>m || y<1
            return 0 
        end
        return img[x,y]
    end
    
    for x=1:n, y=1:m
        Wx[x,y] = min(max_l,lx/(err+abs(imf(x,y)-imf(x-1,y))))
        Wy[x,y] = min(max_l,ly/(err+abs(imf(x,y)-imf(x,y-1))))
        
        Wx[x,y] = lx*(1- abs(imf(x,y)-imf(x-1,y)))^max_l + err
        Wy[x,y] = ly*(1- abs(imf(x,y)-imf(x,y-1)))^max_l + err
        if homog==1
            Wx[x,y] = lx
            Wy[x,y] = ly
        end
    end


    
    if homog==0
        Wx[1,:] .= err
        Wx[:,1] .= err
        Wy[1,:] .= err
        Wy[:,1] .= err
    else
        Wx[1,:] .= err
        Wx[:,1] .= err
        Wy[1,:] .= err
        Wy[:,1] .= err

    end
    function wx(x,y)
        if x>n || x<1 || y>m || y<1
            return 0 
        end
        return Wx[x,y]
    end
    function wy(x,y)
        if x>n || x<1 || y>m || y<1
            return 0 
        end
        return Wy[x,y]
    end
    
    function st(A,val,i,j,x,y)
        if i>n*m || i<1 || j> n*m || j<1
            return 
        end
        if x>n || x<1 || y>m || y<1
            return 
        end
        A[i,j] = val
        return
    end
    
    I = zeros(5*n*m)
    J = zeros(5*n*m)
    V = zeros(5*n*m)
    
    #fill matrix
    
    nrow = 1
    for x = 1:n, y=1:m
        i = (y-1)*n + x
        I[nrow] = i
        J[nrow] = i
        V[nrow] = 1 + wx(x,y) + wx(x+1,y) + wy(x,y) + wy(x,y+1)
        nrow +=1
    end
    
      
    for x=1:n, y=1:m
        i = (y-1)*n + x
        j = i + m
        if !(i>n*m || i<1 || j> n*m || j<1)
            if !(x>n || x<1 || y+1>m || y+1<1)
                I[nrow] = i
                J[nrow] = j
                V[nrow] = -wy(x,y+1)
                nrow +=1
            end
        end
    end
    
      
    for x=1:n, y=1:m
        i = (y-1)*n + x
        j = i - m
        if !(i>n*m || i<1 || j> n*m || j<1)
            if !(x>n || x<1 || y-1>m || y-1<1)
                I[nrow] = i
                J[nrow] = j
                V[nrow] = -wy(x,y)
                nrow +=1
            end
        end
    end
    
    for x=1:n, y=1:m
        i = (y-1)*n + x
        j = i + 1
        if !(i>n*m || i<1 || j> n*m || j<1)
            if !(x+1>n || x+1<1 || y>m || y<1)
                I[nrow] = i
                J[nrow] = j
                V[nrow] = -wx(x+1,y)
                nrow +=1
            end
        end
    end
    
    for x=1:n, y=1:m
        i = (y-1)*n + x
        j = i - 1
        if !(i>n*m || i<1 || j> n*m || j<1)
            if !(x-1>n || x-1<1 || y>m || y<1)
                I[nrow] = i
                J[nrow] = j
                V[nrow] = -wx(x,y)
                nrow +=1
            end
        end
    end

   
    
    I = I[1:nrow-1]
    J = J[1:nrow-1]
    V = V[1:nrow-1]
    
    A = sparse(I,J,V)
    
    return Wx, Wy, A
end


function iterativeSolveSparse(img,lx,ly, A; iter = 500)
    b = vec(img)
    x = vec(copy(img))
    
    @time IterativeSolvers.cg!(x, A, b; initially_zero = false, maxiter = iter)
    
    return x
end;

function get_B(W)
    B = zeros(length(W)) 
    B[end] = 1 + W[end] 
    for i = length(W)-1:-1:1
        B[i] = 1 + W[i] + W[i+1] - W[i+1]^2/B[i+1]
    end    
    return B
end;

function get_a(W) 
    B = get_B(W)
    a = zeros(length(W))
    a[1] = 1/B[1]
    for i = 2:length(a)
        a[i] = 1/B[i] + a[i-1]*W[i]^2/B[i]^2
    end    
    return a
end;

function get_1d_domain_transform(W)
    t = zeros(length(W))
    a = get_a(W)
    for i=2:length(t)
        t[i] = -log((-1 + sqrt(4*(a[i-1]*a[i]*W[i])^2 + 1))/(2*a[i-1]*a[i]*W[i]))
    end
    
    return t
end;


function get1Dtransform(Wx,Wy,y)
    N,M = size(Wx)
    
    function val2(W,x,y)
        if (x>=1) && (y>=1) && (x<=N) && (y<=N)
            return W[x,y]
        end
        return 0
    end
    
    function val1(W,x)
        if (x>=1) && (x<=N)
            return W[x]
        end
        return 0
    end
    
    #compute B
    B = zeros(N)
    B[end] = 1 + Wx[end,y] + Wy[N,y]+val2(Wy,N,y+1) #condicao inicial B
    
    for x = (N-1): -1:1
        B[x] = (1+Wx[x,y]+Wx[x+1,y] + Wy[x,y] + val2(Wy,x,y+1)) - Wx[x+1,y]^2/B[x+1]
    end
    B0 = (1+Wx[1,y] ) - Wx[1,y]^2/B[1]

    a = zeros(N)
    a[1] = 1/B[1]
    for i = 2:length(a)
        a[i] = 1/B[i] + a[i-1]*Wx[i,y]^2/B[i]^2
    end    
    
    t = zeros(N)
    
    for i=2:length(t)
        t[i] = -log((-1 + sqrt(4*(a[i-1]*a[i]*Wx[i,y])^2 + 1))/(2*a[i-1]*a[i]*Wx[i,y]))
    end
    
    return t
end;

function get_2d_domain_transform_old(Wx,Wy)
    Nx,Ny = size(Wx)
    Tx = zeros(size(Wx)) 
    Ty = zeros(size(Wy)) 
    WyT = collect(transpose(Wy)) #faster memory access (row major)
    
    for y = 1 : Ny
        x=y
        Tx[:,y] = get_1d_domain_transform(Wx[:,y]) 
        Ty[:,x] = get_1d_domain_transform(WyT[:,x]) 

    end
    
    return Tx, Ty
end;


function get_2d_domain_transform(Wx,Wy)
    Nx,Ny = size(Wx)
    Tx = zeros(size(Wx)) 
    Ty = zeros(size(Wy)) 
    WyT = collect(transpose(Wy)) #faster memory access (row major)
    
    for y = 1 : Ny
        x=y
        Tx[:,y] = get1Dtransform(Wx, Wy, y) 
        Ty[:,x] = get1Dtransform(Wy',Wx',x)

    end
    
    return Tx, Ty
end;


function edgeaware_1d_gaussian_filter(f,dt, amp, σ)
    g0f = zeros(ComplexF64,length(f))
    g0b = zeros(ComplexF64,length(f))
    
    g1f = zeros(ComplexF64,length(f))
    g1b = zeros(ComplexF64,length(f))
    
    
    α0 = (1.680 + 3.735im)
    α1 = -(0.6803 + 0.2598im)
    λ0 = 1.783 + 0.6318im
    λ1 = 1.723 + 1.997im
    
    γ = real(α0*(1 + exp(-λ0/σ))/(1-exp(-λ0/σ)) + α1*(1 + exp(-λ1/σ))/(1-exp(-λ1/σ)))
    #γ = 1 #above not required 

    a0 = α0/γ*amp
    a1 = α1/γ*amp
    
    b0 = exp(-λ0/σ)
    b1 = exp(-λ1/σ)
    
    
    r00 = (b0-1)^2/(a0*b0)
    r01 = a0/(b0-1)
    
    r10 = (b1-1)^2/(a1*b1)
    r11 = a1/(b1-1)
    
    
    #forward filter
    g0f[1] = a0/(1-b0) * f[1]
    g1f[1] = a1/(1-b1) *f[1]
    for k =2:length(g0f)
        g0f[k] = a0*f[k] + (b0^dt[k])*g0f[k-1] + ((b0^dt[k]-1)/(r00*dt[k]) - r01*b0)*f[k] - ((b0^dt[k]-1)/(r00*dt[k]) - r01*b0^dt[k])*f[k-1]
        g1f[k] = a1*f[k] + (b1^dt[k])*g1f[k-1] + ((b1^dt[k]-1)/(r10*dt[k]) - r11*b1)*f[k] - ((b1^dt[k]-1)/(r10*dt[k]) - r11*b1^dt[k])*f[k-1]
    end
    
    #backward filter
    g0b[end] = a0*b0/(1-b0)*f[end]
    g1b[end] = a1*b1/(1-b1)*f[end]
    
    for k = length(g0b)-1:-1:1
        g0b[k] = a0*b0^dt[k+1]*f[k+1] + b0^dt[k+1]*g0b[k+1] + ((b0^dt[k+1]-1)/(r00*dt[k+1]) - r01*b0)*f[k+1] - ((b0^dt[k+1]-1)/(r00*dt[k+1]) - r01*b0^dt[k+1])*f[k]
        g1b[k] = a1*b1^dt[k+1]*f[k+1] + b1^dt[k+1]*g1b[k+1] + ((b1^dt[k+1]-1)/(r10*dt[k+1]) - r11*b1)*f[k+1] - ((b1^dt[k+1]-1)/(r10*dt[k+1]) - r11*b1^dt[k+1])*f[k]
    end
    
    return  real(g0f) + real(g1f) + real(g0b) + real(g1b)
end


function edgeaware_2d_gaussian_filter(img,dtx, dty, σx, σy, amp)
    nx,ny = size(img)
    res1 = copy(img)
    res2 = copy(img)
    
    nit = 3
    for it = 1:nit
        σxit = σx*sqrt(3)*2^(nit-it)/sqrt(4^nit-1)
        σyit = σy*sqrt(3)*2^(nit-it)/sqrt(4^nit-1)
        for y=1:ny
            res2[:,y] = edgeaware_1d_gaussian_filter(res1[:,y], dtx[:,y], amp, σxit)
        end

        res2 = collect(transpose(res2))
        res1 = collect(transpose(res1))

        for x =1:nx
            res1[:,x] = edgeaware_1d_gaussian_filter(res2[:,x], dty[:,x], amp, σyit)
        end

        res1 = collect(transpose(res1))
    end
    
    return res1
end