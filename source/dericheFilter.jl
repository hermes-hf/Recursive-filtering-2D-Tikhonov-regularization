function dericheCoefficients(s)
    da0 = 1.6800;
    da1 = 3.7350;
    dw0 = 0.6318;
    db0 = 1.7830;
    dc0 = -0.6803;
    dc1 = -0.2598;
    dw1 = 1.9970;
    db1 = 1.7230;
    cos0 = cos(dw0 / s);
    cos1 = cos(dw1 / s);
    sin0 = sin(dw0 / s);
    sin1 = sin(dw1 / s);
    exp0 = exp(-db0 / s);
    exp1 = exp(-db1 / s);
    
    dn0 = da0 + dc0;
    dn1 = exp1 * (dc1 * sin1 - (dc0 + 2.0 * da0) * cos1) + exp0 * (da1 * sin0 - (2.0 * dc0 + da0) * cos0);
    dn2 = 2.0 * exp0 * exp1 * ((da0 + dc0) * cos1 * cos0 - cos1 * da1 * sin0 - cos0 * dc1 * sin1) + dc0 * exp0 * exp0 + da0 * exp1 * exp1;
    dn3 = exp1 * exp0 * exp0 * (dc1 * sin1 - cos1 * dc0) + exp0 * exp1 * exp1 * (da1 * sin0 - cos0 * da0);
    dd1 = -2.0 * exp1 * cos1 - 2.0 * exp0 * cos0;
    dd2 = 4.0 * cos1 * cos0 * exp0 * exp1 + exp1 * exp1 + exp0 * exp0;
    dd3 = -2.0 * cos0 * exp0 * exp1 * exp1 - 2.0 * cos1 * exp1 * exp0 * exp0;
    dd4 = exp0 * exp0 * exp1 * exp1;

    dbn1 = dn1 - dd1 * dn0;
    dbn2 = dn2 - dd2 * dn0;
    dbn3 = dn3 - dd3 * dn0;
    dbn4 = -dd4 * dn0;

    deta0 = dn0 - dd1 * dd1 * dn0 - dd2 * dd2 * dn0 - dd3 * dd3 * dn0 - dd4 * dd4 * dn0 + 2 * dd1 * dn1 + 2 * dd2 * dn2 + 2 * dd3 * dn3;
    deta1 = -dd1 * dd2 * dn0 - dd2 * dd3 * dn0 - dd3 * dd4 * dn0 + dn1 + dd2 * dn1 + dd1 * dn2 + dd3 * dn2 + dd2 * dn3 + dd4 * dn3;
    deta2 = -dd1 * dd3 * dn0 - dd2 * dd4 * dn0 + dd3 * dn1 + dn2 + dd4 * dn2 + dd1 * dn3;
    deta3 = -dd1 * dd4 * dn0 + dd4 * dn1 + dn3;
    
    return [deta0, deta1, deta2, deta3, dd1, dd2, dd3, dd4, dn0, dn1, dn2, dn3]
end


function psi_old(s,eta)
    minvar = 0.18128301024333837
    if s<=1
        psimin(x) = exp(-x^2/(2*minvar))/sqrt(2*pi*minvar)
        return psimin
    end
    var_psi = abs(s^2 - 2*(eta[2] + 2^2*eta[3]+3^2*eta[4])/(eta[1] + 2*sum(eta[2:end])))
    psi(x) = exp(-x^2/(2*var_psi))/sqrt(2*pi*var_psi)
    return psi
end

function psi_(d)
    A = zeros(4,4)
    A[1,:] = -d
    A[2,1] = 1
    A[3,2] = 1
    A[4,3] = 1

    V = eigvecs(A)
    eigs = eigvals(A)
    D = diagm(eigs)
    Vi = inv(V)

    phi = Vi[:,1]
    vphi = V[1,:]

    as = zeros(ComplexF64,4)
    for i =1:4,j=1:4
        as[i] += phi[i]*vphi[i]*phi[j]*vphi[j]/(1 - eigs[i]*eigs[j])
    end


    phi(x) = real((1+sum(d))^2*(as[1]*eigs[1]^abs(x)
                             +as[2]*eigs[2]^abs(x)
                             +as[3]*eigs[3]^abs(x)
                             +as[4]*eigs[4]^abs(x)))

    return phi
end

function gaussianFilter(inp, A, Sx, Sy, c0)
    N = size(inp)[1]
    M = size(inp)[2]
    
    ng = length(A)
    out = zeros(M,N) #output needs to be transposed later
    for g = 1:ng
        buffer = zeros(N,M)
        params =  dericheCoefficients(Sy[g])
        eta= params[1:4]
        d = params[5:8]
        n = params[9:12]
        amp = eta./(eta[1] + 2*sum(eta[2:end]))
        norm = 1/(1+sum(d))
        
        #-----------------------------------------------------------------
        #block filter-----------------------------------------------------
        #-----------------------------------------------------------------
        for x = 1:M
            minK = min(3,x-1)
            maxK = min(3,M-x)
            for k = -minK:maxK
                buffer[:,x] += amp[abs(k)+1]*inp[:,x+k]
            end
        end
        
        buffer0 =  amp[2].*inp[:,1] + amp[3].*inp[:,2] + amp[4].*inp[:,3] #row 0
        buffer1 =  amp[3].*inp[:,1] + amp[4].*inp[:,2]         #negative row 1
        buffer2 =  amp[4].*inp[:,1]                            #negative row 2
        
                         
        #-----------------------------------------------------------------
        #Forward filter---------------------------------------------------
        #-----------------------------------------------------------------
     
        buffer2 = buffer2/norm
        buffer1 = buffer1/norm - buffer2.*d[1]
        buffer0 = buffer0/norm - buffer1.*d[1] - buffer2.*d[2]
        
        buffer[:,1] = buffer[:,1]/norm - d[1].*buffer0 - d[2].*buffer1 - d[3].*buffer2
        buffer[:,2] = buffer[:,2]/norm - d[1].*buffer[:,1] - d[2].*buffer0 -d[3].*buffer1 -d[4].*buffer2
        buffer[:,3] = buffer[:,3]/norm - d[1].*buffer[:,2] - d[2].*buffer[:,1] -d[3].*buffer0 - d[4].*buffer1
        buffer[:,4] = buffer[:,4]/norm - d[1].*buffer[:,3] - d[2].*buffer[:,2] -d[3].*buffer[:,1] - d[4].*buffer0
        
        for x = 5:M, y=1:N
            buffer[y,x] = buffer[y,x]/norm -
                          d[1]*buffer[y,x-1]-
                          d[2]*buffer[y,x-2]-
                          d[3]*buffer[y,x-3]-
                          d[4]*buffer[y,x-4];
        end
        
         
        #compute fwd filter outside image (for backwards initial conditions)
        bufferM1 =    (amp[2]*inp[:,M]+ 
                     amp[3]*inp[:,M-1]+ 
                     amp[4]*inp[:,M-2])/norm-
                     d[1]*buffer[:,M]- 
                     d[2]*buffer[:,M-1]- 
                     d[3]*buffer[:,M-2]- 
                     d[4]*buffer[:,M-3]; 
        
        bufferM2 =    (amp[3].*inp[:,M]+
                      amp[4].*inp[:,M-1])/norm-
                     d[1].*bufferM1- 
                     d[2].*buffer[:,M]- 
                     d[3].*buffer[:,M-1]- 
                     d[4].*buffer[:,M-2];
        
        bufferM3 =    (amp[4].*inp[:,M])/norm -
                     d[1].*bufferM2-
                     d[2].*bufferM1-
                     d[3].*buffer[:,M]-
                     d[4].*buffer[:,M-1];        
        
        h0 = buffer[:,M]*norm
        h1 = (bufferM1 + d[1].*buffer[:,M])*norm
        h2 = (bufferM2 + d[1].*bufferM1 + d[2].*buffer[:,M])*norm
        h3 = (bufferM3 + d[1].*bufferM2 + d[2].*bufferM1 + d[3].*buffer[:,M])*norm
   
        #-----------------------------------------------------------------
        #Backward filter--------------------------------------------------
        #-----------------------------------------------------------------
        
        psi = psi_(d)
        
        bwdM  = h0.*psi(0) + h1.*psi(-1) + h2.*psi(-2) + h3.*psi(-3)
        bwdM1 = h0.*psi(1) + h1.*psi(0)  + h2.*psi(-1) + h3.*psi(-2)
        bwdM2 = h0.*psi(2) + h1.*psi(1)  + h2.*psi(0)  + h3.*psi(-1)
        bwdM3 = h0.*psi(3) + h1.*psi(2)  + h2.*psi(1)  + h3.*psi(0)
        
        buffer[:,M]   = bwdM
        buffer[:,M-1] = buffer[:,M-1]/norm -d[1].*bwdM - d[2].*bwdM1 -d[3].*bwdM2 -d[4].*bwdM3
        buffer[:,M-2] = buffer[:,M-2]/norm -d[1].*buffer[:,M-1] - d[2].*bwdM -d[3].*bwdM1 -d[4].*bwdM2
        buffer[:,M-3] = buffer[:,M-3]/norm -d[1].*buffer[:,M-2] - d[2].*buffer[:,M-1] -d[3].*bwdM -d[4].*bwdM1
        
        
        for x = M-4:-1:1, y=1:N
            buffer[y,x] = buffer[y,x]./norm -
                          d[1]*buffer[y,x+1]-
                          d[2]*buffer[y,x+2]-
                          d[3]*buffer[y,x+3]-
                          d[4]*buffer[y,x+4];
        end
        
        #Tranpose an compute vertical filter and swap dimensions
        buffer = collect(buffer')
        aux = M
        M = N
        N = aux

        #-----------------------------------------------------------------
        #Vertical filter--------------------------------------------------
        #-----------------------------------------------------------------
        
        params =  dericheCoefficients(Sx[g])
        eta= params[1:4]
        amp = eta./(eta[1] + 2*sum(eta[2:end]))
        d = params[5:8]
        n = params[9:12]
        norm = 1/(1+sum(d))
        
          
                            
        #-----------------------------------------------------------------
        #Forward filter---------------------------------------------------
        #-----------------------------------------------------------------
    
        buffer[:,1] = buffer[:,1]/norm 
        buffer[:,2] = buffer[:,2]/norm - d[1].*buffer[:,1]
        buffer[:,3] = buffer[:,3]/norm - d[1].*buffer[:,2] - d[2].*buffer[:,1]
        buffer[:,4] = buffer[:,4]/norm - d[1].*buffer[:,3] - d[2].*buffer[:,2] -d[3].*buffer[:,1]
        
        
        
        for x = 5:M, y=1:N
            buffer[y,x] = buffer[y,x]/norm -
                          d[1]*buffer[y,x-1]-
                          d[2]*buffer[y,x-2]-
                          d[3]*buffer[y,x-3]-
                          d[4]*buffer[y,x-4];
        end
        
      
        #-----------------------------------------------------------------
        #Backward filter--------------------------------------------------
        #-----------------------------------------------------------------
        
            
        h0 = buffer[:,M-3]*norm
        h1 = (buffer[:,M-2] + d[1]*buffer[:,M-3])*norm
        h2 = (buffer[:,M-1] + d[1]*buffer[:,M-2] + d[2]*buffer[:,M-3])*norm
        h3 = (buffer[:,M] + d[1]*buffer[:,M-1] + d[2]*buffer[:,M-2] + d[3]*buffer[:,M-3])*norm
           
        psi2 = psi_(d)
        #bwd
        buffer[:,M-3] = h0*psi2(0) + h1*psi2(-1) + h2*psi2(-2) + h3*psi2(-3)
        buffer[:,M-2] = h0*psi2(1) + h1*psi2(0) + h2*psi2(-1) + h3*psi2(-2)
        buffer[:,M-1] = h0*psi2(2) + h1*psi2(1) + h2*psi2(0) + h3*psi2(1)
        buffer[:,M]   = h0*psi2(3) + h1*psi2(2) + h2*psi2(1) + h3*psi2(0)
        
        #compute conditions outside image
        bufferM1 = h0*psi2(4) + h1*psi2(3) + h2*psi2(2) + h3*psi2(1)
        bufferM2 = h0*psi2(5) + h1*psi2(4) + h2*psi2(3) + h3*psi2(2)
        bufferM3 = h0*psi2(6) + h1*psi2(5) + h2*psi2(4) + h3*psi2(3)
     
        
        
        for x = M-4:-1:1, y =1:N
            buffer[y,x] = buffer[y,x]./norm -
                          d[1]*buffer[y,x+1]-
                          d[2]*buffer[y,x+2]-
                          d[3]*buffer[y,x+3]-
                          d[4]*buffer[y,x+4];
        end
        
        #compute conditions outside image (backwards)
        buffer0 = -d[1]*buffer[:,1]-
                  d[2]*buffer[:,2]-
                  d[3]*buffer[:,3]-
                  d[4]*buffer[:,4];
        
        buffer1 = -d[1]*buffer0 -
                  d[2]*buffer[:,1]-
                  d[3]*buffer[:,2]-
                  d[4]*buffer[:,3];
        
        buffer2 = -d[1]*buffer1 -
                  d[2]*buffer0 -
                  d[3]*buffer[:,1]-
                  d[4]*buffer[:,2];
       #block filter
        for x = 1:M, y = 1:N
            minK = min(3,x-1)
            maxK = min(3,M-x)
            for k = -minK:maxK
                out[y,x] += amp[abs(k)+1]*buffer[y,x+k]*A[g]*2*pi*Sx[g]*Sy[g]
            end
        end
        out[:,1] +=  amp[4]*buffer2*A[g]*2*pi*Sx[g]*Sy[g]+
                     amp[3]*buffer1*A[g]*2*pi*Sx[g]*Sy[g]+
                     amp[2]*buffer0*A[g]*2*pi*Sx[g]*Sy[g];
        
        out[:,2] +=  amp[4]*buffer1*A[g]*2*pi*Sx[g]*Sy[g]+
                     amp[3]*buffer0*A[g]*2*pi*Sx[g]*Sy[g];
        
        out[:,3] +=  amp[4]*buffer0*A[g]*2*pi*Sx[g]*Sy[g];
        
        
        out[:,M]  += amp[2]*bufferM1*A[g]*2*pi*Sx[g]*Sy[g]+
                     amp[3]*bufferM2*A[g]*2*pi*Sx[g]*Sy[g]+
                     amp[4]*bufferM3*A[g]*2*pi*Sx[g]*Sy[g];
        
        out[:,M-1]+= amp[3]*bufferM1*A[g]*2*pi*Sx[g]*Sy[g]+
                     amp[4]*bufferM2*A[g]*2*pi*Sx[g]*Sy[g];
        
        out[:,M-2]+= amp[4]*bufferM1*A[g]*2*pi*Sx[g]*Sy[g];
                     
        #unswap dimensions
        aux = M
        M = N
        N = aux
        
        
    end
    
    out = out' + c0.*inp
    return  out
end
