using ForwardDiff
using IterativeSolvers
using PyCall


function Jfun(λx,λy,w)
    return 1 /
    sqrt((1 + 2 *λx *(1 -cos(w)) + 2 *λy)^2 - 4 *λy ^2)
end


#routine to compute inverse of monotonous decreasing function
function inversefun(f,val,a,b)
    tol = 0.00001
    
    if abs(a-b)<tol
        return (a+b)/2;
    end
    
    center = f((a+b)/2)
    
    if center == val
        return (a+b)/2 
    end
    
    if center < val
        b = (a+b)/2
    else
        a = (a+b)/2
    end
    
    return inversefun(f,val,a,b)
end;

#here s is variance, aliased gaussian term
function gauss_dtft(s,x)
    return  exp(-(x-4*pi)^2/2*s) + exp(-(x-2*pi)^2/2*s)+exp(-x^2/2*s)+exp(-(x+2*pi)^2/2*s)+ exp(-(x+4*pi)^2/2*s)
end;


function gaussianKernelFitPlot(λx, λy, ng; linOpt=0)
    ff = Jfun
    c0 = ff(λx,λy,pi)
    A = get_amps(c0, ff, λx, λy, 1:ng);
    Sx = fitSigma(c0, ff, A,λx,λy)
    Sy = Sx .* λy./λx
    
    if linOpt==1
        A, c0 = linsolveA(A, Sx, Sy, c0, λx, λy);
    end

    N = 100
    res = zeros(N)
    for i = 1:N
        for k = 1:ng
         res[i] += A[k]*gauss_dtft(Sx[k],pi*i/N);
        end
        res[i] += c0
    end
    return res
end

function gaussianfitpts(λx, λy, ng; linOpt=0)
    ff = Jfun
    c0 = ff(λx,λy,pi)
    A = get_amps(c0, ff, λx, λy, 1:ng);
    Sx, X = fitSigmaX(c0, ff, A,λx,λy)
    Sy = Sx .* λy./λx
    return X
end



function get_amps(c0, F, λx, λy , i, power=1)
    m = length(i)
    div = 0
    for k =1:m
        div += k^power
    end
    return (F(λx,λy,0) - c0)./div.*(i .^ power)
end;

function f(c0, F, A, S, λx, λy, k, x)
    res = F(λx, λy , x) .- c0
    for i=1:k
        res = res - A[i]*gauss_dtft(S[i],x)
    end
    return res
end

function fitSigma(c0, F, A, λx, λy)
    S = zeros(length(A))
    ng = length(A)
    X = zeros(ng+1)
    X[1] = pi
    
    for i =1:ng
        
        function lambdaf(x)
            return f(c0, F, A, S, λx, λy, i-1, x)
        end
        
        x1 = inversefun(lambdaf,A[i],0,pi)
        x2 = X[i]
        xi = (x1+x2)/2 
        
        
        nit = 0


        maxit = 10
        while nit<maxit
                        
            S[i] = abs((log(A[i]) - log(abs(lambdaf(xi))))*2/xi^2)
            #fix aliasing ----------------------------------------
            if S[i]<4.0
                lambdaf2(s) = A[i]*gauss_dtft(s,xi)
                S[i] = inversefun(lambdaf2,lambdaf(xi),0.0,4) 
            end
            
            if nit==maxit-1
                break
            end
            
            f1(x) = A[i]*gauss_dtft(S[i],x)
            f2(x) = lambdaf(x)

            f1d(x) = ForwardDiff.derivative(f1, x)
            f2d(x) = ForwardDiff.derivative(f2, x)
            
            d1 = f1d(xi)
            d2 = f2d(xi)
            
            diff = ((d1) - (d2))
            
            err = 1e-7
            
            if diff < -err 
                x2 = xi
                xi = (x1+x2)/2
            elseif diff> err && lambdaf(xi)>= 5e-4
                x1 = xi
                xi = (x1+x2)/2
            else
                break
                
            end
            
            nit+=1
        end
        #---------------------------------------------------------
        
        X[i+1] = xi
        
    end
    
    return S
end

function fitSigmaX(c0, F, A, λx, λy)
    S = zeros(length(A))
    ng = length(A)
    X = zeros(ng+1)
    X[1] = pi
    
    for i =1:ng
        
        function lambdaf(x)
            return f(c0, F, A, S, λx, λy, i-1, x)
        end
        
        x1 = inversefun(lambdaf,A[i],0,pi)
        x2 = X[i]
        xi = (x1+x2)/2 
        
        
        nit = 0


        maxit = 10
        while nit<maxit
                        
            S[i] = abs((log(A[i]) - log(abs(lambdaf(xi))))*2/xi^2)
            #fix aliasing ----------------------------------------
            if S[i]<4.0
                lambdaf2(s) = A[i]*gauss_dtft(s,xi)
                S[i] = inversefun(lambdaf2,lambdaf(xi),0.0,4) 
            end
            
            if nit==maxit-1
                break
            end
            
            f1(x) = A[i]*gauss_dtft(S[i],x)
            f2(x) = lambdaf(x)

            f1d(x) = ForwardDiff.derivative(f1, x)
            f2d(x) = ForwardDiff.derivative(f2, x)
            
            d1 = f1d(xi)
            d2 = f2d(xi)
            
            diff = ((d1) - (d2))
            
            err = 1e-7
            
            if diff < -err 
                x2 = xi
                xi = (x1+x2)/2
            elseif diff> err && lambdaf(xi)>= 5e-4
                x1 = xi
                xi = (x1+x2)/2
            else
                break
                
            end
            
            nit+=1
        end
        #---------------------------------------------------------
        
        X[i+1] = xi
        
    end
    
    return S,X
end


function linsolveA(A, Sx, Sy, c0, λx, λy; npts=100, nits=10)
    F = Jfun;
    ng = length(A)
    Z = LinRange(0,pi,npts)
    M = zeros(npts, ng+1)
    b = zeros(npts)       
    multiplier = 1e3
    

    for i = 1:npts
        b[i] = F(λx, λy, Z[i])
    end
        
    for i =1:npts, j=1:ng
        M[i,j] = gauss_dtft(Sx[j], Z[i])
    end
    M[:,end] .= 1 

    M[1,:] .*= multiplier
    b[1]    *= multiplier
    
    t= zeros(ng+1)
    t[1:ng] = A
    t[end] = c0
    
    lsmr!(t, M, b)
    #t = M\b
    
    A = t[1:end-1]
    c0 = t[end]
    
    return A,c0
end

function gaussianKernelFit(λx, λy, ng; linOpt=0)
    ff = Jfun
    c0 = ff(λx,λy,pi)
    A = get_amps(c0, ff, λx, λy, 1:ng);
    Sx = fitSigma(c0, ff, A,λx,λy)
    Sy = Sx .* λy./λx
    
    if linOpt==1
        A, c0 = linsolveA(A, Sx, Sy, c0, λx, λy);
    end

    A = A ./ sqrt.(2*pi .* Sx)
    return A, sqrt.(Sx), sqrt.(Sy), c0
end


function recursiveFilter2D(img,λx,λy, nGaussian; linOpt=1)
    out = copy(img)
    if λx>=λy
        A, Sx, Sy, c0 = gaussianKernelFit(λx,λy,nGaussian; linOpt=linOpt);
    else
        A, Sy, Sx, c0 = gaussianKernelFit(λy,λx,nGaussian; linOpt=linOpt);
    end

    channels, N,M = size(img)
    
    for i = 1:channels
        out[i,:,:] = min.(max.(gaussianFilter(copy(img[i,:,:]),A,Sx,Sy,c0),0),1);
    end
    return out;
end

#Non-linear optimization
#PYTHON CODE START--------------------------------------------------------
py"""

import numpy as np
from scipy.optimize import least_squares

def Fv(p, N, c0 =0):
    P = np.array(p)
    n = len(P)//3
    α = P[0::3]
    β = P[1::3]
    γ = P[2::3]
    x = np.arange(N)
    X =  np.repeat(x[None,:], repeats=[n], axis=0)
    xres = np.sum(α[:,None]**2 * np.exp(-(X*β[:,None])**2/2),0)
    yres = np.sum(α[:,None]**2 * np.exp(-(X*γ[:,None])**2/2),0)
   
    xres[0] += c0
    yres[0] += c0
    
    return np.concatenate([xres,yres])

def loss(p, K, c0 = 0):
    res = Fv(p, K.shape[0]/2, c0)
    return res - K

def optimize(K, Nparam, c0 = 0, p = None):

    if p is None:
        P  = np.zeros(Nparam*3)
        P[0::3] = np.linspace(0.1,1.,Nparam)
        P[1::3] = np.linspace(0.1,1.,Nparam)**2
        P[2::3] = np.linspace(0.1,1.,Nparam)**2
    else:
        P = np.array(p)
    
    def func(*p):
        return loss(p[0],K, c0)
    
    P = least_squares(func, x0 =P, max_nfev = 5, tr_solver = 'lsmr', method='trf', loss='soft_l1', x_scale = 'jac', jac='2-point')["x"]
    return least_squares(func, x0 =P, max_nfev = 15, tr_solver = 'exact', method='trf', loss='soft_l1', x_scale = 'jac', jac='2-point')["x"]

"""
#PYTHON CODE END----------------------------------------------------------

function get1DspatialKernel(λx, λy, n)
    n=2*n
    osize = n
    n = n÷2 + 1 #size for rfft
    J = zeros(n) 
    for i = 1:n
        w = 2*pi*(i-1)/osize
        J[i] = Jfun(λx,λy,w)
    end
    return irfft(J, osize)[1:osize÷2]
end;


function improve(A,Sx,Sy,c0,lx,ly)
    nopt = 200
    ng = length(A)
    K = zeros(nopt)
    K = vcat(get1DspatialKernel(lx,ly,nopt),get1DspatialKernel(ly,lx,nopt))
    P = zeros(3*ng)
    P[1:3:end] = sqrt.(A)
    P[2:3:end] = 1 ./Sx
    P[3:3:end] = 1 ./Sy
    
    P = py"optimize"(K,ng,c0,P)
    A = P[1:3:end].^2
    Sx = 1 ./ P[2:3:end]
    Sy = 1 ./ P[3:3:end]
    
    return A,Sx,Sy,c0
end

function recursiveFilter2Doptimize(img,lx,ly, nGaussian; linOpt = 0)
    out = copy(img)
    if lx>=ly
        A, Sx, Sy, c0 = gaussianKernelFit(lx,ly,nGaussian; linOpt = linOpt);
    else
        A, Sy, Sx, c0 = gaussianKernelFit(ly,lx,nGaussian; linOpt = linOpt);
    end
     
    channels,N,M = size(img)
    A, Sx, Sy, c0 = improve(A,Sx,Sy,c0,lx,ly);
    
    for i = 1:channels
        out[i,:,:] = min.(max.(gaussianFilter(img[i,:,:],A,Sx,Sy,c0),0),1);
    end
    return out;
end;

function recursiveFilter2DCustom(img,Params)
    out = copy(img)
    A, Sx, Sy, c0 = Params[1];
    channels,N,M = size(img)
    for i = 1:channels
        out[i,:,:] = min.(max.(gaussianFilter(img[i,:,:],A,Sx,Sy,c0),0),1);
    end
    return out;
end;

