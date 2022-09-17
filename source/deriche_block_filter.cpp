#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <omp.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign> Matrixd;


#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(a,b) (((a)>(b))?(a):(b))
#define alignment 64 //number of bytes for SIMD alignment

#define L 64

using namespace std;
using namespace std::chrono;


class dericheCoefficients{
private:
    
public:
    
    float d1,d2,d3,d4;
    float n0,n1,n2,n3;
    float bn1,bn2,bn3,bn4;
    float eta0,eta1,eta2,eta3;

    dericheCoefficients(){
    };
    void setCoeff(float s){

        //USE DOUBLE FOR BETTER PRECISION BEFORE PASSING TO FLOAT VARIABLE
        double cos0,cos1,sin0,sin1,exp0,exp1;
        const double da0 = 1.6800;
        const double da1 = 3.7350;
        const double dw0 = 0.6318;
        const double db0 = 1.7830;
        const double dc0 =-0.6803;
        const double dc1 =-0.2598;
        const double dw1 = 1.9970;
        const double db1 = 1.7230;

        double dd1,dd2,dd3,dd4;

        double dn0,dn1,dn2,dn3;
        double dbn1,dbn2,dbn3,dbn4;

        double deta0,deta1,deta2,deta3;

        cos0 = cos(dw0/s);
        cos1 = cos(dw1/s);
        sin0 = sin(dw0/s);
        sin1 = sin(dw1/s);
        exp0 = exp(-db0/s);
        exp1 = exp(-db1/s);

        //forward coefficients
        dn0 = da0 + dc0;
        dn1 = exp1*(dc1*sin1 - (dc0 + 2.f*da0)*cos1) + exp0*(da1*sin0 - (2.f*dc0 + da0)*cos0);
        dn2 = 2.0f*exp0*exp1*((da0+dc0)*cos1*cos0 - cos1*da1*sin0 - cos0*dc1*sin1) + dc0*exp0*exp0 + da0*exp1*exp1;
        dn3 = exp1*exp0*exp0*(dc1*sin1 - cos1*dc0) + exp0*exp1*exp1*(da1*sin0 - cos0*da0);
        dd1 = -2.f*exp1*cos1 - 2.f*exp0*cos0;
        dd2 = 4.f*cos1*cos0*exp0*exp1 + exp1*exp1 + exp0*exp0;
        dd3 = -2.f*cos0*exp0*exp1*exp1 - 2.0f*cos1*exp1*exp0*exp0;
        dd4 = exp0*exp0*exp1*exp1;
        //backward coefficients
        dbn1 = dn1 - dd1*dn0;
        dbn2 = dn2 - dd2*dn0;
        dbn3 = dn3 - dd3*dn0;
        dbn4 = -dd4*dn0;

        deta0 = dn0 - dd1*dd1*dn0 - dd2*dd2*dn0 - dd3*dd3*dn0 - dd4*dd4*dn0 + 2*dd1*dn1 + 2*dd2*dn2 + 2*dd3*dn3;
        deta1 = -dd1*dd2*dn0 - dd2*dd3*dn0 - dd3*dd4*dn0 + dn1 + dd2*dn1 + dd1*dn2 + dd3*dn2 + dd2*dn3 +dd4*dn3;
        deta2 = -dd1*dd3*dn0 - dd2*dd4*dn0 + dd3*dn1 + dn2 + dd4*dn2 + dd1*dn3;
        deta3 = -dd1*dd4*dn0 + dd4*dn1 + dn3;

        d1 = dd1;
        d2 = dd2;
        d3 = dd3;
        d4 = dd4;
        n0 = dn0;
        n1 = dn1;
        n2 = dn2;
        n3 = dn3;
        bn1 = dbn1;
        bn2 = dbn2;
        bn3 = dbn3;
        bn4 = dbn4;
        eta0 = deta0;
        eta1 = deta1;
        eta2 = deta2;
        eta3 = deta3;
        
    }
};



class eigensolveVars{
private:

public:
    std::complex<double> v[4];
    std::complex<double> vi[4];
    std::complex<double> ls[4];
    float psi[7];

    eigensolveVars() {}

    void eigensolve(float d1, float d2, float d3, float d4){
        Matrixd M(4,4);

        for(int i=0;i<4;i++){
            for(int j=0;j<4;j++){
                M(i,j)=0;
            }

        }

        M(0,0) = -d1; M(0,1) = -d2; M(0,2) = -d3; M(0,3) = -d4;
        M(1,0) = 1.0;
        M(2,1) = 1.0;
        M(3,2) = 1.0;

        Eigen::EigenSolver<Matrixd> es(M);

        Eigen::Matrix<std::complex<double>, -1, -1, 1, -1, -1> V = es.eigenvectors();
        Eigen::Matrix<std::complex<double>, -1, -1, 1, -1, -1> Vinv = V.inverse();
        Eigen::Matrix<std::complex<double>, -1, 1> diag = es.eigenvalues();

        ls[0] = diag(0); ls[1] = diag(1); ls[2] = diag(2); ls[3] = diag(3);
        v[0] = V(0,0); v[1] = V(0,1); v[2] = V(0,2); v[3] = V(0,3);
        vi[0] = Vinv(0,0); vi[1] = Vinv(1,0); vi[2] = Vinv(2,0); vi[3] = Vinv(3,0);

        // Precompute psis
        for (int x = 0; x < 7; x++) {
            std::complex<double> res(0,0);
            for (int i = 0; i < 4; i++) {
                for(int j =0; j < 4; j++) {
                    res += v[i]
                        * v[j]
                        * vi[i]
                        * vi[j]
                        * pow(ls[i],float(x)) / (1.0 - ls[i]*ls[j]);
                }
            }
            psi[x] = res.real();
        }
    }
};


//function to obtain index of flattened matrix
static inline int ind(int i, int j,int m){
    return i*m+j;
}



//unbuffered in-place transpose
static inline void transpose(float *M){
    float aux;
    for(int i = 0; i < L; i++){
        for(int j = 0; j < i; j++){
            aux = M[ind(i,j,L)];
            M[ind(i,j,L)] = M[ind(j,i,L)];
            M[ind(j,i,L)] = aux;
        }
    }
}

//swap pointers
void swap(float * &r, float * &s){
   float *temp = &r[0];
   r = &s[0];
   s = &temp[0];
   return;
} 


float* dericheFilter(float* in, float* A, float* Sx, float* Sy, int ng ,int N){

    float *buffer =  (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *out = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *temp1 = (float*)aligned_alloc(alignment,sizeof(float) * L*L);
    float *temp2 = (float*)aligned_alloc(alignment,sizeof(float) * L*L);

    //buffers for boundary conditions
    float *bound1 = (float*)aligned_alloc(alignment,sizeof(float)* 4*N);
    float *bound2 = (float*)aligned_alloc(alignment,sizeof(float)* 4*N);

    int nB = (N + L -1)/L;
    float **block = new float*[nB*nB]; //buffer blocks    
    float **inBlock = new float*[nB*nB]; //input blocks    
    float **outBlock = new float*[nB*nB]; //output blocks    

    float **blockBound1 = new float*[nB]; //boundary terms
    float **blockBound2 = new float*[nB]; //boundary terms

    //set block addresses
    for(int i = 0; i < nB; i++){
        for(int j = 0; j < nB; j++){
            inBlock[ind(i,j,nB)] = &buffer[j*L*L + i*L*L*nB];
            block[ind(i,j,nB)] = &in[j*L*L + i*L*L*nB];
            outBlock[ind(i,j,nB)] = &out[j*L*L + i*L*L*nB];
        }
        blockBound1[i] = &bound1[i*4*L];
        blockBound2[i] = &bound2[i*4*L];
    }

    //read input to blocks
    for(int i = 0; i < nB; i++){
        for(int j = 0; j < nB; j++)
            for(int xx = 0; xx < L; xx++){
                for(int yy = 0; yy < L; yy++){
                    inBlock[ind(i,j,nB)][ind(xx,yy,L)] = in[ind(i*L + xx,j*L + yy,N)];
                }
            }
    }
    swap(in,buffer); //at this point input will be in block format

    dericheCoefficients p;
    int x, y, g;
    float b1,b2,b3,B;
    float q,q2,q3,b0;


    float amp0;
    float amp1;
    float amp2;
    float amp3;
    float d1;
    float d2;
    float d3;
    float d4;

    for(g = 0; g < ng; g++){
        //*-----------------------------------------------------------------------//
        //* Horizontal filter-----------------------------------------------------//
        //*-----------------------------------------------------------------------//
        eigensolveVars boundVars;

        //&Coefficients-----------------------------------------------------------//
        {    
            p.setCoeff(Sx[g]);   
            amp0 = A[g]*p.eta0;
            amp1 = A[g]*p.eta1;
            amp2 = A[g]*p.eta2;
            amp3 = A[g]*p.eta3;

            d1 = p.d1;
            d2 = p.d2;
            d3 = p.d3;
            d4 = p.d4;

            boundVars.eigensolve(d1,d2,d3,d4);
        }
        
        //&Block filtering--------------------------------------------------------//
        {   
            //^filter first row blocks, i=0
            for(int j = 0; j < nB; j++){
                //filter first rows (assume input zero outside boundary)
                #pragma omp simd aligned(block, inBlock:alignment)
                for( int yy = 0; yy < L; yy++){
                    block[ind(0,j,nB)][ind(0,yy,L)] =  
                     amp0*inBlock[ind(0,j,nB)][ind(0,yy,L)]
                    +amp1*inBlock[ind(0,j,nB)][ind(1,yy,L)]
                    +amp2*inBlock[ind(0,j,nB)][ind(2,yy,L)]
                    +amp3*inBlock[ind(0,j,nB)][ind(3,yy,L)];
                    
                    block[ind(0,j,nB)][ind(1,yy,L)] = 
                     amp1*inBlock[ind(0,j,nB)][ind(0,yy,L)]
                    +amp0*inBlock[ind(0,j,nB)][ind(1,yy,L)]
                    +amp1*inBlock[ind(0,j,nB)][ind(2,yy,L)]
                    +amp2*inBlock[ind(0,j,nB)][ind(3,yy,L)]
                    +amp3*inBlock[ind(0,j,nB)][ind(4,yy,L)]
                    ;

                    block[ind(0,j,nB)][ind(2,yy,L)] = 
                     amp2*inBlock[ind(0,j,nB)][ind(0,yy,L)]
                    +amp1*inBlock[ind(0,j,nB)][ind(1,yy,L)]
                    +amp0*inBlock[ind(0,j,nB)][ind(2,yy,L)]
                    +amp1*inBlock[ind(0,j,nB)][ind(3,yy,L)]
                    +amp2*inBlock[ind(0,j,nB)][ind(4,yy,L)]
                    +amp3*inBlock[ind(0,j,nB)][ind(5,yy,L)]
                    ;
                }

                //filter inner rows
                for(int xx = 3; xx < L-3; xx++){
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for( int yy = 0; yy < L; yy++){
                        block[ind(0,j,nB)][ind(xx,yy,L)] =  
                        amp3*inBlock[ind(0,j,nB)][ind(xx-3,yy,L)]
                        +amp2*inBlock[ind(0,j,nB)][ind(xx-2,yy,L)]
                        +amp1*inBlock[ind(0,j,nB)][ind(xx-1,yy,L)]
                        +amp0*inBlock[ind(0,j,nB)][ind(xx,yy,L)]
                        +amp1*inBlock[ind(0,j,nB)][ind(xx+1,yy,L)]
                        +amp2*inBlock[ind(0,j,nB)][ind(xx+2,yy,L)]
                        +amp3*inBlock[ind(0,j,nB)][ind(xx+3,yy,L)]
                        ;
                    }
                }
            
                //filter last rows (fetch from blocks below)
                #pragma omp simd aligned(block, inBlock:alignment)
                for( int yy = 0; yy < L; yy++){
                    block[ind(0,j,nB)][ind(L-1,yy,L)] =  
                    +amp3*inBlock[ind(1,j,nB)][ind(2,yy,L)]
                    +amp2*inBlock[ind(1,j,nB)][ind(1,yy,L)]
                    +amp1*inBlock[ind(1,j,nB)][ind(0,yy,L)]
                    +amp0*inBlock[ind(0,j,nB)][ind(L-1,yy,L)]
                    +amp1*inBlock[ind(0,j,nB)][ind(L-2,yy,L)]
                    +amp2*inBlock[ind(0,j,nB)][ind(L-3,yy,L)]
                    +amp3*inBlock[ind(0,j,nB)][ind(L-4,yy,L)];
                    
                    block[ind(0,j,nB)][ind(L-2,yy,L)] = 
                    +amp3*inBlock[ind(1,j,nB)][ind(1,yy,L)]
                    +amp2*inBlock[ind(1,j,nB)][ind(0,yy,L)]
                    +amp1*inBlock[ind(0,j,nB)][ind(L-1,yy,L)]
                    +amp0*inBlock[ind(0,j,nB)][ind(L-2,yy,L)]
                    +amp1*inBlock[ind(0,j,nB)][ind(L-3,yy,L)]
                    +amp2*inBlock[ind(0,j,nB)][ind(L-4,yy,L)]
                    +amp3*inBlock[ind(0,j,nB)][ind(L-5,yy,L)];

                    block[ind(0,j,nB)][ind(L-3,yy,L)] = 
                    +amp3*inBlock[ind(1,j,nB)][ind(0,yy,L)]
                    +amp2*inBlock[ind(0,j,nB)][ind(L-1,yy,L)]
                    +amp1*inBlock[ind(0,j,nB)][ind(L-2,yy,L)]
                    +amp0*inBlock[ind(0,j,nB)][ind(L-3,yy,L)]
                    +amp1*inBlock[ind(0,j,nB)][ind(L-4,yy,L)]
                    +amp2*inBlock[ind(0,j,nB)][ind(L-5,yy,L)]
                    +amp3*inBlock[ind(0,j,nB)][ind(L-6,yy,L)];
                }

            }

            //^filter inner blocks
            for(int i = 1; i < nB - 1; i++){
                for(int j = 0; j <nB; j++){
                    //apply manually to first rows: fetch from previous block
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for(int yy = 0; yy < L; yy ++){
                        block[ind(i,j,nB)][ind(0,yy,L)] =  
                            +amp3*inBlock[ind(i-1,j,nB)][ind(L-3,yy,L)]
                            +amp2*inBlock[ind(i-1,j,nB)][ind(L-2,yy,L)]
                            +amp1*inBlock[ind(i-1,j,nB)][ind(L-1,yy,L)]
                            +amp0*inBlock[ind(i,j,nB)][ind(0,yy,L)]
                            +amp1*inBlock[ind(i,j,nB)][ind(1,yy,L)]
                            +amp2*inBlock[ind(i,j,nB)][ind(2,yy,L)]
                            +amp3*inBlock[ind(i,j,nB)][ind(3,yy,L)]
                            ;
                        
                        block[ind(i,j,nB)][ind(1,yy,L)] =  
                            +amp3*inBlock[ind(i-1,j,nB)][ind(L-2,yy,L)]
                            +amp2*inBlock[ind(i-1,j,nB)][ind(L-1,yy,L)]
                            +amp1*inBlock[ind(i,j,nB)][ind(0,yy,L)]
                            +amp0*inBlock[ind(i,j,nB)][ind(1,yy,L)]
                            +amp1*inBlock[ind(i,j,nB)][ind(2,yy,L)]
                            +amp2*inBlock[ind(i,j,nB)][ind(3,yy,L)]
                            +amp3*inBlock[ind(i,j,nB)][ind(4,yy,L)]
                            ;

                        block[ind(i,j,nB)][ind(2,yy,L)] =  
                            +amp3*inBlock[ind(i-1,j,nB)][ind(L-1,yy,L)]
                            +amp2*inBlock[ind(i,j,nB)][ind(0,yy,L)]
                            +amp1*inBlock[ind(i,j,nB)][ind(1,yy,L)]
                            +amp0*inBlock[ind(i,j,nB)][ind(2,yy,L)]
                            +amp1*inBlock[ind(i,j,nB)][ind(3,yy,L)]
                            +amp2*inBlock[ind(i,j,nB)][ind(4,yy,L)]
                            +amp3*inBlock[ind(i,j,nB)][ind(5,yy,L)]
                            ;
                    }

                    //filter inner rows
                    for(int xx = 3; xx < L-3; xx++){
                        #pragma omp simd aligned(block, inBlock:alignment)
                        for( int yy = 0; yy < L; yy++){
                            block[ind(i,j,nB)][ind(xx,yy,L)] =  
                            amp3*inBlock[ind(i,j,nB)][ind(xx-3,yy,L)]
                            +amp2*inBlock[ind(i,j,nB)][ind(xx-2,yy,L)]
                            +amp1*inBlock[ind(i,j,nB)][ind(xx-1,yy,L)]
                            +amp0*inBlock[ind(i,j,nB)][ind(xx,yy,L)]
                            +amp1*inBlock[ind(i,j,nB)][ind(xx+1,yy,L)]
                            +amp2*inBlock[ind(i,j,nB)][ind(xx+2,yy,L)]
                            +amp3*inBlock[ind(i,j,nB)][ind(xx+3,yy,L)]
                            ;
                        }
                    }
                
                    //filter last rows (fetch from blocks below)
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for( int yy = 0; yy < L; yy++){
                        block[ind(i,j,nB)][ind(L-1,yy,L)] =  
                        +amp3*inBlock[ind(i+1,j,nB)][ind(2,yy,L)]
                        +amp2*inBlock[ind(i+1,j,nB)][ind(1,yy,L)]
                        +amp1*inBlock[ind(i+1,j,nB)][ind(0,yy,L)]
                        +amp0*inBlock[ind(i,j,nB)][ind(L-1,yy,L)]
                        +amp1*inBlock[ind(i,j,nB)][ind(L-2,yy,L)]
                        +amp2*inBlock[ind(i,j,nB)][ind(L-3,yy,L)]
                        +amp3*inBlock[ind(i,j,nB)][ind(L-4,yy,L)];
                        
                        block[ind(i,j,nB)][ind(L-2,yy,L)] = 
                        +amp3*inBlock[ind(i+1,j,nB)][ind(1,yy,L)]
                        +amp2*inBlock[ind(i+1,j,nB)][ind(0,yy,L)]
                        +amp1*inBlock[ind(i,j,nB)][ind(L-1,yy,L)]
                        +amp0*inBlock[ind(i,j,nB)][ind(L-2,yy,L)]
                        +amp1*inBlock[ind(i,j,nB)][ind(L-3,yy,L)]
                        +amp2*inBlock[ind(i,j,nB)][ind(L-4,yy,L)]
                        +amp3*inBlock[ind(i,j,nB)][ind(L-5,yy,L)];

                        block[ind(i,j,nB)][ind(L-3,yy,L)] = 
                        +amp3*inBlock[ind(i+1,j,nB)][ind(0,yy,L)]
                        +amp2*inBlock[ind(i,j,nB)][ind(L-1,yy,L)]
                        +amp1*inBlock[ind(i,j,nB)][ind(L-2,yy,L)]
                        +amp0*inBlock[ind(i,j,nB)][ind(L-3,yy,L)]
                        +amp1*inBlock[ind(i,j,nB)][ind(L-4,yy,L)]
                        +amp2*inBlock[ind(i,j,nB)][ind(L-5,yy,L)]
                        +amp3*inBlock[ind(i,j,nB)][ind(L-6,yy,L)];
                    }

                }
            }
            
            //^filter last row blocks, i=nB-1                  
            for(int j = 0; j < nB; j++){
                //filter first rows (fetch from previous blocks)
                #pragma omp simd aligned(block, inBlock:alignment)
                for(int yy = 0; yy < L; yy ++){
                    block[ind(nB-1,j,nB)][ind(0,yy,L)] =  
                        +amp3*inBlock[ind(nB-2,j,nB)][ind(L-3,yy,L)]
                        +amp2*inBlock[ind(nB-2,j,nB)][ind(L-2,yy,L)]
                        +amp1*inBlock[ind(nB-2,j,nB)][ind(L-1,yy,L)]
                        +amp0*inBlock[ind(nB-1,j,nB)][ind(0,yy,L)]
                        +amp1*inBlock[ind(nB-1,j,nB)][ind(1,yy,L)]
                        +amp2*inBlock[ind(nB-1,j,nB)][ind(2,yy,L)]
                        +amp3*inBlock[ind(nB-1,j,nB)][ind(3,yy,L)]
                        ;
                    
                    block[ind(nB-1,j,nB)][ind(1,yy,L)] =  
                        +amp3*inBlock[ind(nB-2,j,nB)][ind(L-2,yy,L)]
                        +amp2*inBlock[ind(nB-2,j,nB)][ind(L-1,yy,L)]
                        +amp1*inBlock[ind(nB-1,j,nB)][ind(0,yy,L)]
                        +amp0*inBlock[ind(nB-1,j,nB)][ind(1,yy,L)]
                        +amp1*inBlock[ind(nB-1,j,nB)][ind(2,yy,L)]
                        +amp2*inBlock[ind(nB-1,j,nB)][ind(3,yy,L)]
                        +amp3*inBlock[ind(nB-1,j,nB)][ind(4,yy,L)]
                        ;

                    block[ind(nB-1,j,nB)][ind(2,yy,L)] =  
                        +amp3*inBlock[ind(nB-2,j,nB)][ind(L-1,yy,L)]
                        +amp2*inBlock[ind(nB-1,j,nB)][ind(0,yy,L)]
                        +amp1*inBlock[ind(nB-1,j,nB)][ind(1,yy,L)]
                        +amp0*inBlock[ind(nB-1,j,nB)][ind(2,yy,L)]
                        +amp1*inBlock[ind(nB-1,j,nB)][ind(3,yy,L)]
                        +amp2*inBlock[ind(nB-1,j,nB)][ind(4,yy,L)]
                        +amp3*inBlock[ind(nB-1,j,nB)][ind(5,yy,L)]
                        ;
                }

                //filter inner rows
                for(int xx = 3; xx < L-3; xx++){
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for( int yy = 0; yy < L; yy++){
                        block[ind(nB-1,j,nB)][ind(xx,yy,L)] =  
                         amp3*inBlock[ind(nB-1,j,nB)][ind(xx-3,yy,L)]
                        +amp2*inBlock[ind(nB-1,j,nB)][ind(xx-2,yy,L)]
                        +amp1*inBlock[ind(nB-1,j,nB)][ind(xx-1,yy,L)]
                        +amp0*inBlock[ind(nB-1,j,nB)][ind(xx,yy,L)]
                        +amp1*inBlock[ind(nB-1,j,nB)][ind(xx+1,yy,L)]
                        +amp2*inBlock[ind(nB-1,j,nB)][ind(xx+2,yy,L)]
                        +amp3*inBlock[ind(nB-1,j,nB)][ind(xx+3,yy,L)]
                        ;
                    }
                }
                
                //filter last rows (assume input zero outside boundary)
                #pragma omp simd aligned(block, inBlock:alignment)
                for( int yy = 0; yy < L; yy++){
                    block[ind(nB-1,j,nB)][ind(L-1,yy,L)] =  
                    +amp0*inBlock[ind(nB-1,j,nB)][ind(L-1,yy,L)]
                    +amp1*inBlock[ind(nB-1,j,nB)][ind(L-2,yy,L)]
                    +amp2*inBlock[ind(nB-1,j,nB)][ind(L-3,yy,L)]
                    +amp3*inBlock[ind(nB-1,j,nB)][ind(L-4,yy,L)];
                    
                    block[ind(nB-1,j,nB)][ind(L-2,yy,L)] = 
                    +amp1*inBlock[ind(nB-1,j,nB)][ind(L-1,yy,L)]
                    +amp0*inBlock[ind(nB-1,j,nB)][ind(L-2,yy,L)]
                    +amp1*inBlock[ind(nB-1,j,nB)][ind(L-3,yy,L)]
                    +amp2*inBlock[ind(nB-1,j,nB)][ind(L-4,yy,L)]
                    +amp3*inBlock[ind(nB-1,j,nB)][ind(L-5,yy,L)]
                    ;

                    block[ind(nB-1,j,nB)][ind(L-3,yy,L)] = 
                    +amp2*inBlock[ind(nB-1,j,nB)][ind(L-1,yy,L)]
                    +amp1*inBlock[ind(nB-1,j,nB)][ind(L-2,yy,L)]
                    +amp0*inBlock[ind(nB-1,j,nB)][ind(L-3,yy,L)]
                    +amp1*inBlock[ind(nB-1,j,nB)][ind(L-4,yy,L)]
                    +amp2*inBlock[ind(nB-1,j,nB)][ind(L-5,yy,L)]
                    +amp3*inBlock[ind(nB-1,j,nB)][ind(L-6,yy,L)]
                    ;
                }
     
            }

        
        }

        //&Forward recursive filter-----------------------------------------------//
        {}
        {
            //^Filter first row blocks
            for(int i = 0; i < 1; i++){
                for(int j = 0; j < nB; j++){
                    //Filter outside image
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for( int yy = 0; yy < L; yy++){
                        blockBound1[j][ind(0,yy,L)] = (amp3*inBlock[ind(i,j,nB)][ind(0,yy,L)]); //row -3

                        blockBound1[j][ind(1,yy,L)] = (amp3*inBlock[ind(i,j,nB)][ind(1,yy,L)] //row -2
                                                    + amp2*inBlock[ind(i,j,nB)][ind(0,yy,L)])
                                                    - d1*blockBound1[j][ind(0,yy,L)];

                        blockBound1[j][ind(2,yy,L)] = (amp3*inBlock[ind(i,j,nB)][ind(2,yy,L)] //row -1
                                                    + amp2*inBlock[ind(i,j,nB)][ind(1,yy,L)]
                                                    + amp1*inBlock[ind(i,j,nB)][ind(0,yy,L)])
                                                    - d1*blockBound1[j][ind(1,yy,L)]
                                                    - d2*blockBound1[j][ind(0,yy,L)];
                    }

                    //Filter first rows
                    #pragma omp simd aligned(block:alignment)
                    for(int yy = 0; yy < L; yy++){

                        block[ind(i,j,nB)][ind(0,yy,L)] = block[ind(i,j,nB)][ind(0,yy,L)]
                        -d1*blockBound1[j][ind(2,yy,L)]
                        -d2*blockBound1[j][ind(1,yy,L)]
                        -d3*blockBound1[j][ind(0,yy,L)]
                        ;
                    
                        block[ind(i,j,nB)][ind(1,yy,L)] = block[ind(i,j,nB)][ind(1,yy,L)]
                        -d1*block[ind(i,j,nB)][ind(0,yy,L)]
                        -d2*blockBound1[j][ind(2,yy,L)]
                        -d3*blockBound1[j][ind(1,yy,L)]
                        -d4*blockBound1[j][ind(0,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(2,yy,L)] = block[ind(i,j,nB)][ind(2,yy,L)] 
                        -d1*block[ind(i,j,nB)][ind(1,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(0,yy,L)]
                        -d3*blockBound1[j][ind(2,yy,L)]
                        -d4*blockBound1[j][ind(1,yy,L)]
                        ;
                        
                        block[ind(i,j,nB)][ind(3,yy,L)] = block[ind(i,j,nB)][ind(3,yy,L)] 
                        -d1*block[ind(i,j,nB)][ind(2,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(1,yy,L)]
                        -d3*block[ind(i,j,nB)][ind(0,yy,L)]
                        -d4*blockBound1[j][ind(2,yy,L)]
                        ;

                    }

                    //Filter inner rows
                    for(int xx = 4; xx < L; xx++){
                        #pragma omp simd aligned(block:alignment)
                        for(int yy = 0; yy < L; yy++){
                            block[ind(i,j,nB)][ind(xx,yy,L)] = block[ind(i,j,nB)][ind(xx,yy,L)]
                            -d1*block[ind(i,j,nB)][ind(xx-1,yy,L)]
                            -d2*block[ind(i,j,nB)][ind(xx-2,yy,L)]
                            -d3*block[ind(i,j,nB)][ind(xx-3,yy,L)]
                            -d4*block[ind(i,j,nB)][ind(xx-4,yy,L)]
                            ;
                        }
                    }
                
                }
            }
           
           
            //^Filter inner blocks
            for(int i = 1; i < nB; i++){
                for(int j = 0; j < nB; j++){

                    //Filter first rows
                    #pragma omp simd aligned(block:alignment)
                    for(int yy = 0; yy < L; yy++){
                        block[ind(i,j,nB)][ind(0,yy,L)] = block[ind(i,j,nB)][ind(0,yy,L)]
                        -d1*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                        -d2*block[ind(i-1,j,nB)][ind(L-2,yy,L)]
                        -d3*block[ind(i-1,j,nB)][ind(L-3,yy,L)]
                        -d4*block[ind(i-1,j,nB)][ind(L-4,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(1,yy,L)] = block[ind(i,j,nB)][ind(1,yy,L)]
                        -d1*block[ind(i,j,nB)][ind(0,yy,L)]
                        -d2*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                        -d3*block[ind(i-1,j,nB)][ind(L-2,yy,L)]
                        -d4*block[ind(i-1,j,nB)][ind(L-3,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(2,yy,L)] = block[ind(i,j,nB)][ind(2,yy,L)]
                        -d1*block[ind(i,j,nB)][ind(1,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(0,yy,L)]
                        -d3*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                        -d4*block[ind(i-1,j,nB)][ind(L-2,yy,L)]
                        ;
                        
                        block[ind(i,j,nB)][ind(3,yy,L)] = block[ind(i,j,nB)][ind(3,yy,L)]
                        -d1*block[ind(i,j,nB)][ind(2,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(1,yy,L)]
                        -d3*block[ind(i,j,nB)][ind(0,yy,L)]
                        -d4*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                        ;

                    }

                    //Filter inner rows
                    for(int xx = 4; xx < L; xx++){
                        #pragma omp simd aligned(block:alignment)
                        for(int yy = 0; yy < L; yy++){
                            block[ind(i,j,nB)][ind(xx,yy,L)] = block[ind(i,j,nB)][ind(xx,yy,L)]
                            -d1*block[ind(i,j,nB)][ind(xx-1,yy,L)]
                            -d2*block[ind(i,j,nB)][ind(xx-2,yy,L)]
                            -d3*block[ind(i,j,nB)][ind(xx-3,yy,L)]
                            -d4*block[ind(i,j,nB)][ind(xx-4,yy,L)]
                            ;
                        }
                    }
                }
            }

            //^Filter outside image 
            for(int i = nB-1;  i < nB; i++){
                //uses last blocks
                for(int j = 0; j < nB; j++){
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for( int yy = 0; yy < L; yy++){
                        //row N
                        blockBound1[j][ind(0,yy,L)] = (amp3*inBlock[ind(i,j,nB)][ind(L-3,yy,L)] 
                                                    + amp2*inBlock[ind(i,j,nB)][ind(L-2,yy,L)]
                                                    + amp1*inBlock[ind(i,j,nB)][ind(L-1,yy,L)])
                                                    - d1*block[ind(i,j,nB)][ind(L-1,yy,L)]
                                                    - d2*block[ind(i,j,nB)][ind(L-2,yy,L)]
                                                    - d3*block[ind(i,j,nB)][ind(L-3,yy,L)]
                                                    - d4*block[ind(i,j,nB)][ind(L-4,yy,L)]
                                                    ;
                        //row N+1
                        blockBound1[j][ind(1,yy,L)] = (amp3*inBlock[ind(i,j,nB)][ind(L-2,yy,L)] 
                                                    + amp2*inBlock[ind(i,j,nB)][ind(L-1,yy,L)])
                                                    - d1*blockBound1[j][ind(0,yy,L)]
                                                    - d2*block[ind(i,j,nB)][ind(L-1,yy,L)]
                                                    - d3*block[ind(i,j,nB)][ind(L-2,yy,L)]
                                                    - d4*block[ind(i,j,nB)][ind(L-3,yy,L)]
                                                    ;
                        //row N+2
                        blockBound1[j][ind(2,yy,L)] = (amp3*inBlock[ind(i,j,nB)][ind(L-1,yy,L)])
                                                    - d1*blockBound1[j][ind(1,yy,L)]
                                                    - d2*blockBound1[j][ind(0,yy,L)]
                                                    - d3*block[ind(i,j,nB)][ind(L-1,yy,L)]
                                                    - d4*block[ind(i,j,nB)][ind(L-2,yy,L)]
                                                    ;
                      
                        //Compute h values 
                        //h3
                        blockBound1[j][ind(3,yy,L)] = (blockBound1[j][ind(2,yy,L)]
                                                    + d1*blockBound1[j][ind(1,yy,L)]
                                                    + d2*blockBound1[j][ind(0,yy,L)]
                                                    + d3*block[ind(i,j,nB)][ind(L-1,yy,L)]);
                        //h2
                        blockBound1[j][ind(2,yy,L)] = (blockBound1[j][ind(1,yy,L)]
                                                    + d1*blockBound1[j][ind(0,yy,L)]
                                                    + d2*block[ind(i,j,nB)][ind(L-1,yy,L)]);

                        //h1
                        blockBound1[j][ind(1,yy,L)] = (blockBound1[j][ind(0,yy,L)]
                                                    + d1*block[ind(i,j,nB)][ind(L-1,yy,L)]);

                        //h0
                        blockBound1[j][ind(0,yy,L)] = block[ind(i,j,nB)][ind(L-1,yy,L)];
                    }
                }
            }


        }
        
        
        //&Backward recursive filter----------------------------------------------//
        {}
        {
            //^Filter last row blocks
            for(int i = nB-1; i>nB-2; i--){
                for(int j = 0; j < nB; j++){
                    //Filter last rows
                    #pragma omp simd aligned(block:alignment)
                    for(int yy = 0; yy < L; yy++){
                        //Filter outside image
                        {
                            blockBound2[j][ind(0,yy,L)]= 
                                         boundVars.psi[0]*blockBound1[j][ind(0,yy,L)]
                                        +boundVars.psi[1]*blockBound1[j][ind(1,yy,L)]
                                        +boundVars.psi[2]*blockBound1[j][ind(2,yy,L)]
                                        +boundVars.psi[3]*blockBound1[j][ind(3,yy,L)];

                            blockBound2[j][ind(1,yy,L)]= 
                                         boundVars.psi[1]*blockBound1[j][ind(0,yy,L)]
                                        +boundVars.psi[0]*blockBound1[j][ind(1,yy,L)]
                                        +boundVars.psi[1]*blockBound1[j][ind(2,yy,L)]
                                        +boundVars.psi[2]*blockBound1[j][ind(3,yy,L)];

                            blockBound2[j][ind(2,yy,L)]= 
                                         boundVars.psi[2]*blockBound1[j][ind(0,yy,L)]
                                        +boundVars.psi[1]*blockBound1[j][ind(1,yy,L)]
                                        +boundVars.psi[0]*blockBound1[j][ind(2,yy,L)]
                                        +boundVars.psi[1]*blockBound1[j][ind(3,yy,L)];       
                            
                            blockBound2[j][ind(3,yy,L)]= 
                                        boundVars.psi[3]*blockBound1[j][ind(0,yy,L)]
                                        +boundVars.psi[2]*blockBound1[j][ind(1,yy,L)]
                                        +boundVars.psi[1]*blockBound1[j][ind(2,yy,L)]
                                        +boundVars.psi[0]*blockBound1[j][ind(3,yy,L)];                
                        }
                        //Filter inside image
                        block[ind(i,j,nB)][ind(L-1,yy,L)] = blockBound2[j][ind(0,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(L-2,yy,L)] = block[ind(i,j,nB)][ind(L-2,yy,L)]
                        -d1*blockBound2[j][ind(0,yy,L)]
                        -d2*blockBound2[j][ind(1,yy,L)]
                        -d3*blockBound2[j][ind(2,yy,L)]
                        -d4*blockBound2[j][ind(3,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(L-3,yy,L)] = block[ind(i,j,nB)][ind(L-3,yy,L)]
                        -d1*block[ind(i,j,nB)][ind(L-2,yy,L)]
                        -d2*blockBound2[j][ind(0,yy,L)]
                        -d3*blockBound2[j][ind(1,yy,L)]
                        -d4*blockBound2[j][ind(2,yy,L)]
                        ;
                        
                        block[ind(i,j,nB)][ind(L-4,yy,L)] = block[ind(i,j,nB)][ind(L-4,yy,L)]
                        -d1*block[ind(i,j,nB)][ind(L-3,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(L-2,yy,L)]
                        -d3*blockBound2[j][ind(0,yy,L)]
                        -d4*blockBound2[j][ind(1,yy,L)]
                        ;

                    }

                    //Filter inner rows
                    for(int xx = L-5; xx>= 0; xx--){
                        #pragma omp simd aligned(block:alignment)
                        for(int yy = 0; yy < L; yy++){
                            block[ind(i,j,nB)][ind(xx,yy,L)] = block[ind(i,j,nB)][ind(xx,yy,L)]
                            -d1*block[ind(i,j,nB)][ind(xx+1,yy,L)]
                            -d2*block[ind(i,j,nB)][ind(xx+2,yy,L)]
                            -d3*block[ind(i,j,nB)][ind(xx+3,yy,L)]
                            -d4*block[ind(i,j,nB)][ind(xx+4,yy,L)]
                            ;
                        }
                    }
                }
            }
           
           
            //^Filter inner blocks
            for(int i = nB-2; i >= 0; i--){
                for(int j = 0; j < nB; j++){

                    //Filter last rows
                    #pragma omp simd aligned(block:alignment)
                    for(int yy = 0; yy < L; yy++){
                        block[ind(i,j,nB)][ind(L-1,yy,L)] = block[ind(i,j,nB)][ind(L-1,yy,L)]
                        -d1*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        -d2*block[ind(i+1,j,nB)][ind(1,yy,L)]
                        -d3*block[ind(i+1,j,nB)][ind(2,yy,L)]
                        -d4*block[ind(i+1,j,nB)][ind(3,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(L-2,yy,L)] = block[ind(i,j,nB)][ind(L-2,yy,L)]
                        -d1*block[ind(i,j,nB)][ind(L-1,yy,L)]
                        -d2*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        -d3*block[ind(i+1,j,nB)][ind(1,yy,L)]
                        -d4*block[ind(i+1,j,nB)][ind(2,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(L-3,yy,L)] = block[ind(i,j,nB)][ind(L-3,yy,L)]
                        -d1*block[ind(i,j,nB)][ind(L-2,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(L-1,yy,L)]
                        -d3*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        -d4*block[ind(i+1,j,nB)][ind(1,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(L-4,yy,L)] = block[ind(i,j,nB)][ind(L-4,yy,L)]
                        -d1*block[ind(i,j,nB)][ind(L-3,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(L-2,yy,L)]
                        -d3*block[ind(i,j,nB)][ind(L-1,yy,L)]
                        -d4*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        ;

                      
                    }

                    //Filter inner rows
                    for(int xx = L-5; xx >= 0; xx--){
                        #pragma omp simd aligned(block:alignment)
                        for(int yy = 0; yy < L; yy++){
                            block[ind(i,j,nB)][ind(xx,yy,L)] = block[ind(i,j,nB)][ind(xx,yy,L)]
                            -d1*block[ind(i,j,nB)][ind(xx+1,yy,L)]
                            -d2*block[ind(i,j,nB)][ind(xx+2,yy,L)]
                            -d3*block[ind(i,j,nB)][ind(xx+3,yy,L)]
                            -d4*block[ind(i,j,nB)][ind(xx+4,yy,L)]
                            ;
                        }
                    }
                }
            }
        
        }

        //transpose:
        for(int i = 0; i < nB; i++){
            for(int j = 0; j < i; j++){
                swap(block[ind(i,j,nB)],block[ind(j,i,nB)]);
            }
        }
        for(int i = 0; i < nB; i++){
            for(int j = 0; j < nB; j++){
                transpose(block[ind(i,j,nB)]);
            }  
        }

        //TODO:: FIX BOUNDARY FOR VERTICAL FILTER
        //*-----------------------------------------------------------------------//
        //* Vertical Filter-------------------------------------------------------//
        //*-----------------------------------------------------------------------//

        //&y filter coefficients
        {
            p.setCoeff(Sy[g]);
            amp0 = p.eta0;
            amp1 = p.eta1;
            amp2 = p.eta2;
            amp3 = p.eta3;
            d1 = p.d1;
            d2 = p.d2;
            d3 = p.d3;
            d4 = p.d4;
            boundVars.eigensolve(d1,d2,d3,d4);
        }
        

        //&Forward recursive filter-----------------------------------------------//
        {
            
            //^Filter first row blocks
            for(int i = 0; i < 1; i++){
                for(int j = 0; j < nB; j++){

                    //Filter first rows
                    #pragma omp simd aligned(block:alignment)
                    for(int yy = 0; yy < L; yy++){
                    
                        block[ind(i,j,nB)][ind(1,yy,L)] += 
                        -d1*block[ind(i,j,nB)][ind(0,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(2,yy,L)] += 
                        -d1*block[ind(i,j,nB)][ind(1,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(0,yy,L)]
                        ;
                        
                        block[ind(i,j,nB)][ind(3,yy,L)] += 
                        -d1*block[ind(i,j,nB)][ind(2,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(1,yy,L)]
                        -d3*block[ind(i,j,nB)][ind(0,yy,L)]
                        ;

                    }

                    //Filter inner rows
                    for(int xx = 4; xx < L; xx++){
                        #pragma omp simd aligned(block:alignment)
                        for(int yy = 0; yy < L; yy++){
                            block[ind(i,j,nB)][ind(xx,yy,L)] += 
                            -d1*block[ind(i,j,nB)][ind(xx-1,yy,L)]
                            -d2*block[ind(i,j,nB)][ind(xx-2,yy,L)]
                            -d3*block[ind(i,j,nB)][ind(xx-3,yy,L)]
                            -d4*block[ind(i,j,nB)][ind(xx-4,yy,L)]
                            ;
                        }
                    }
                }
            }
           
           
            //^Filter inner blocks
            for(int i = 1; i < nB; i++){
                for(int j = 0; j < nB; j++){

                    //Filter first rows
                    #pragma omp simd aligned(block:alignment)
                    for(int yy = 0; yy < L; yy++){
                        block[ind(i,j,nB)][ind(0,yy,L)] += 
                        -d1*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                        -d2*block[ind(i-1,j,nB)][ind(L-2,yy,L)]
                        -d3*block[ind(i-1,j,nB)][ind(L-3,yy,L)]
                        -d4*block[ind(i-1,j,nB)][ind(L-4,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(1,yy,L)] += 
                        -d1*block[ind(i,j,nB)][ind(0,yy,L)]
                        -d2*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                        -d3*block[ind(i-1,j,nB)][ind(L-2,yy,L)]
                        -d4*block[ind(i-1,j,nB)][ind(L-3,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(2,yy,L)] += 
                        -d1*block[ind(i,j,nB)][ind(1,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(0,yy,L)]
                        -d3*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                        -d4*block[ind(i-1,j,nB)][ind(L-2,yy,L)]
                        ;
                        
                        block[ind(i,j,nB)][ind(3,yy,L)] += 
                        -d1*block[ind(i,j,nB)][ind(2,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(1,yy,L)]
                        -d3*block[ind(i,j,nB)][ind(0,yy,L)]
                        -d4*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                        ;

                    }

                    //Filter inner rows
                    for(int xx = 4; xx < L; xx++){
                        #pragma omp simd aligned(block:alignment)
                        for(int yy = 0; yy < L; yy++){
                            block[ind(i,j,nB)][ind(xx,yy,L)] += 
                            -d1*block[ind(i,j,nB)][ind(xx-1,yy,L)]
                            -d2*block[ind(i,j,nB)][ind(xx-2,yy,L)]
                            -d3*block[ind(i,j,nB)][ind(xx-3,yy,L)]
                            -d4*block[ind(i,j,nB)][ind(xx-4,yy,L)]
                            ;
                        }
                    }
                }
            }
        
            //^Compute H values
            for(int i = nB-1; i<nB; i++){
                for(int j = 0; j < nB; j++){
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for( int yy = 0; yy < L; yy++){
                        //h3
                        blockBound2[j][ind(3,yy,L)] = (block[ind(i,j,nB)][ind(L-1,yy,L)]
                                                    + d1*block[ind(i,j,nB)][ind(L-2,yy,L)]
                                                    + d2*block[ind(i,j,nB)][ind(L-3,yy,L)]
                                                    + d3*block[ind(i,j,nB)][ind(L-4,yy,L)]);
                        //h2
                        blockBound2[j][ind(2,yy,L)] = (block[ind(i,j,nB)][ind(L-2,yy,L)]
                                                    + d1*block[ind(i,j,nB)][ind(L-3,yy,L)]
                                                    + d2*block[ind(i,j,nB)][ind(L-4,yy,L)]);

                        //h1
                        blockBound2[j][ind(1,yy,L)] = (block[ind(i,j,nB)][ind(L-3,yy,L)]
                                                    + d1*block[ind(i,j,nB)][ind(L-4,yy,L)]);

                        //h0
                        blockBound2[j][ind(0,yy,L)] = block[ind(i,j,nB)][ind(L-4,yy,L)];

                    }
                }
            }

        }


        //&Backward recursive filter----------------------------------------------//
        {}
        {
            //^Filter last row blocks
            for(int i = nB-1; i>nB-2; i--){
                for(int j = 0; j < nB; j++){
                    //Filter last rows
                    #pragma omp simd aligned(block:alignment)
                    for(int yy = 0; yy < L; yy++){
                        
                        block[ind(i,j,nB)][ind(L-1,yy,L)] = 
                             blockBound2[j][ind(0,yy,L)]*boundVars.psi[3]
                            +blockBound2[j][ind(1,yy,L)]*boundVars.psi[2]
                            +blockBound2[j][ind(2,yy,L)]*boundVars.psi[1]
                            +blockBound2[j][ind(3,yy,L)]*boundVars.psi[0];
                        
                        block[ind(i,j,nB)][ind(L-2,yy,L)] = 
                             blockBound2[j][ind(0,yy,L)]*boundVars.psi[2]
                            +blockBound2[j][ind(1,yy,L)]*boundVars.psi[1]
                            +blockBound2[j][ind(2,yy,L)]*boundVars.psi[0]
                            +blockBound2[j][ind(3,yy,L)]*boundVars.psi[1];
                    
                        block[ind(i,j,nB)][ind(L-3,yy,L)] = 
                             blockBound2[j][ind(0,yy,L)]*boundVars.psi[1]
                            +blockBound2[j][ind(1,yy,L)]*boundVars.psi[0]
                            +blockBound2[j][ind(2,yy,L)]*boundVars.psi[1]
                            +blockBound2[j][ind(3,yy,L)]*boundVars.psi[2];
                    
                        
                        block[ind(i,j,nB)][ind(L-4,yy,L)] = 
                             blockBound2[j][ind(0,yy,L)]*boundVars.psi[0]
                            +blockBound2[j][ind(1,yy,L)]*boundVars.psi[1]
                            +blockBound2[j][ind(2,yy,L)]*boundVars.psi[2]
                            +blockBound2[j][ind(3,yy,L)]*boundVars.psi[3];

                        //compute conditions outside image

                        //row N
                        blockBound1[j][ind(0,yy,L)] = 
                             blockBound2[j][ind(0,yy,L)]*boundVars.psi[4]
                            +blockBound2[j][ind(1,yy,L)]*boundVars.psi[3]
                            +blockBound2[j][ind(2,yy,L)]*boundVars.psi[2]
                            +blockBound2[j][ind(3,yy,L)]*boundVars.psi[1];
                        
                        //row N+1
                        blockBound1[j][ind(1,yy,L)] = 
                             blockBound2[j][ind(0,yy,L)]*boundVars.psi[5]
                            +blockBound2[j][ind(1,yy,L)]*boundVars.psi[4]
                            +blockBound2[j][ind(2,yy,L)]*boundVars.psi[3]
                            +blockBound2[j][ind(3,yy,L)]*boundVars.psi[2];

                        //row N+2
                        blockBound1[j][ind(2,yy,L)] = 
                             blockBound2[j][ind(0,yy,L)]*boundVars.psi[6]
                            +blockBound2[j][ind(1,yy,L)]*boundVars.psi[5]
                            +blockBound2[j][ind(2,yy,L)]*boundVars.psi[4]
                            +blockBound2[j][ind(3,yy,L)]*boundVars.psi[3];
                    
                    }
                           
                    //Filter inner rows
                    for(int xx = L-5; xx>= 0; xx--){
                        #pragma omp simd aligned(block:alignment)
                        for(int yy = 0; yy < L; yy++){
                            block[ind(i,j,nB)][ind(xx,yy,L)] += 
                            -d1*block[ind(i,j,nB)][ind(xx+1,yy,L)]
                            -d2*block[ind(i,j,nB)][ind(xx+2,yy,L)]
                            -d3*block[ind(i,j,nB)][ind(xx+3,yy,L)]
                            -d4*block[ind(i,j,nB)][ind(xx+4,yy,L)]
                            ;
                        }
                    }
                }
            }
           
           
            //^Filter inner blocks
            for(int i = nB-2; i >= 0; i--){
                for(int j = 0; j < nB; j++){

                    //Filter last rows
                    #pragma omp simd aligned(block:alignment)
                    for(int yy = 0; yy < L; yy++){
                        block[ind(i,j,nB)][ind(L-1,yy,L)] += 
                        -d1*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        -d2*block[ind(i+1,j,nB)][ind(1,yy,L)]
                        -d3*block[ind(i+1,j,nB)][ind(2,yy,L)]
                        -d4*block[ind(i+1,j,nB)][ind(3,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(L-2,yy,L)] += 
                        -d1*block[ind(i,j,nB)][ind(L-1,yy,L)]
                        -d2*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        -d3*block[ind(i+1,j,nB)][ind(1,yy,L)]
                        -d4*block[ind(i+1,j,nB)][ind(2,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(L-3,yy,L)] += 
                        -d1*block[ind(i,j,nB)][ind(L-2,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(L-1,yy,L)]
                        -d3*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        -d4*block[ind(i+1,j,nB)][ind(1,yy,L)]
                        ;

                        block[ind(i,j,nB)][ind(L-4,yy,L)] += 
                        -d1*block[ind(i,j,nB)][ind(L-3,yy,L)]
                        -d2*block[ind(i,j,nB)][ind(L-2,yy,L)]
                        -d3*block[ind(i,j,nB)][ind(L-1,yy,L)]
                        -d4*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        ;

                      
                    }

                    //Filter inner rows
                    for(int xx = L-5; xx >= 0; xx--){
                        #pragma omp simd aligned(block:alignment)
                        for(int yy = 0; yy < L; yy++){
                            block[ind(i,j,nB)][ind(xx,yy,L)] += 
                            -d1*block[ind(i,j,nB)][ind(xx+1,yy,L)]
                            -d2*block[ind(i,j,nB)][ind(xx+2,yy,L)]
                            -d3*block[ind(i,j,nB)][ind(xx+3,yy,L)]
                            -d4*block[ind(i,j,nB)][ind(xx+4,yy,L)]
                            ;
                        }
                    }
                }
            }

            //^Filter outside image
            for(int i = 0; i <1; i++){
                for(int j = 0; j < nB; j++){
                    for(int yy=0; yy<L; yy++){
                        //row -1
                        blockBound2[j][ind(0,yy,L)] = 
                                    -d1*block[ind(i,j,nB)][ind(0,yy,L)]
                                    -d2*block[ind(i,j,nB)][ind(1,yy,L)]
                                    -d3*block[ind(i,j,nB)][ind(2,yy,L)]
                                    -d4*block[ind(i,j,nB)][ind(3,yy,L)];

                        //row -2
                        blockBound2[j][ind(1,yy,L)] = 
                                    -d1*blockBound2[j][ind(0,yy,L)]
                                    -d2*block[ind(i,j,nB)][ind(0,yy,L)]
                                    -d3*block[ind(i,j,nB)][ind(1,yy,L)]
                                    -d4*block[ind(i,j,nB)][ind(2,yy,L)]
                                    ;
                        //row -3
                        blockBound2[j][ind(2,yy,L)] = 
                                    -d1*blockBound2[j][ind(1,yy,L)]
                                    -d2*blockBound2[j][ind(0,yy,L)]
                                    -d3*block[ind(i,j,nB)][ind(0,yy,L)]
                                    -d4*block[ind(i,j,nB)][ind(1,yy,L)]
                                    ;
                    }
                }
            }
        }

        /*at this point: blockBound1 -> N,N+1,N+2
                         blockBound2 -> -3,-2,-1
        */

        //&Block filtering--------------------------------------------------------//
        if(g==0)
        {   
            //^filter first row blocks, i=0
            for(int j = 0; j < nB; j++){
                //filter first rows (assume input zero outside boundary)
                #pragma omp simd aligned(block, inBlock:alignment)
                for( int yy = 0; yy < L; yy++){
                    outBlock[ind(0,j,nB)][ind(0,yy,L)] =  
                    +amp3*blockBound2[j][ind(2,yy,L)]
                    +amp2*blockBound2[j][ind(1,yy,L)]
                    +amp1*blockBound2[j][ind(0,yy,L)]
                    +amp0*block[ind(0,j,nB)][ind(0,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(1,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(2,yy,L)]
                    +amp3*block[ind(0,j,nB)][ind(3,yy,L)];
                    
                    outBlock[ind(0,j,nB)][ind(1,yy,L)] = 
                    +amp3*blockBound2[j][ind(1,yy,L)]
                    +amp2*blockBound2[j][ind(0,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(0,yy,L)]
                    +amp0*block[ind(0,j,nB)][ind(1,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(2,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(3,yy,L)]
                    +amp3*block[ind(0,j,nB)][ind(4,yy,L)]
                    ;

                    outBlock[ind(0,j,nB)][ind(2,yy,L)] = 
                    +amp3*blockBound2[j][ind(0,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(0,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(1,yy,L)]
                    +amp0*block[ind(0,j,nB)][ind(2,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(3,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(4,yy,L)]
                    +amp3*block[ind(0,j,nB)][ind(5,yy,L)]
                    ;
                }

                //filter inner rows
                for(int xx = 3; xx < L-3; xx++){
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for( int yy = 0; yy < L; yy++){
                        outBlock[ind(0,j,nB)][ind(xx,yy,L)] =  
                         amp3*block[ind(0,j,nB)][ind(xx-3,yy,L)]
                        +amp2*block[ind(0,j,nB)][ind(xx-2,yy,L)]
                        +amp1*block[ind(0,j,nB)][ind(xx-1,yy,L)]
                        +amp0*block[ind(0,j,nB)][ind(xx,yy,L)]
                        +amp1*block[ind(0,j,nB)][ind(xx+1,yy,L)]
                        +amp2*block[ind(0,j,nB)][ind(xx+2,yy,L)]
                        +amp3*block[ind(0,j,nB)][ind(xx+3,yy,L)]
                        ;
                    }
                }
            
                //filter last rows (fetch from blocks below)
                #pragma omp simd aligned(block, inBlock:alignment)
                for( int yy = 0; yy < L; yy++){
                    outBlock[ind(0,j,nB)][ind(L-1,yy,L)] =  
                    +amp3*block[ind(1,j,nB)][ind(2,yy,L)]
                    +amp2*block[ind(1,j,nB)][ind(1,yy,L)]
                    +amp1*block[ind(1,j,nB)][ind(0,yy,L)]
                    +amp0*block[ind(0,j,nB)][ind(L-1,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(L-2,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(L-3,yy,L)]
                    +amp3*block[ind(0,j,nB)][ind(L-4,yy,L)];
                    
                    outBlock[ind(0,j,nB)][ind(L-2,yy,L)] = 
                    +amp3*block[ind(1,j,nB)][ind(1,yy,L)]
                    +amp2*block[ind(1,j,nB)][ind(0,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(L-1,yy,L)]
                    +amp0*block[ind(0,j,nB)][ind(L-2,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(L-3,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(L-4,yy,L)]
                    +amp3*block[ind(0,j,nB)][ind(L-5,yy,L)];

                    outBlock[ind(0,j,nB)][ind(L-3,yy,L)] = 
                    +amp3*block[ind(1,j,nB)][ind(0,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(L-1,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(L-2,yy,L)]
                    +amp0*block[ind(0,j,nB)][ind(L-3,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(L-4,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(L-5,yy,L)]
                    +amp3*block[ind(0,j,nB)][ind(L-6,yy,L)];
                }

            }

            //^filter inner blocks
            for(int i = 1; i < nB - 1; i++){
                for(int j = 0; j <nB; j++){
                    //apply manually to first rows: fetch from previous block
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for(int yy = 0; yy < L; yy ++){
                        outBlock[ind(i,j,nB)][ind(0,yy,L)] =  
                            +amp3*block[ind(i-1,j,nB)][ind(L-3,yy,L)]
                            +amp2*block[ind(i-1,j,nB)][ind(L-2,yy,L)]
                            +amp1*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                            +amp0*block[ind(i,j,nB)][ind(0,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(1,yy,L)]
                            +amp2*block[ind(i,j,nB)][ind(2,yy,L)]
                            +amp3*block[ind(i,j,nB)][ind(3,yy,L)]
                            ;
                        
                        outBlock[ind(i,j,nB)][ind(1,yy,L)] =  
                            +amp3*block[ind(i-1,j,nB)][ind(L-2,yy,L)]
                            +amp2*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(0,yy,L)]
                            +amp0*block[ind(i,j,nB)][ind(1,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(2,yy,L)]
                            +amp2*block[ind(i,j,nB)][ind(3,yy,L)]
                            +amp3*block[ind(i,j,nB)][ind(4,yy,L)]
                            ;

                        outBlock[ind(i,j,nB)][ind(2,yy,L)] =  
                            +amp3*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                            +amp2*block[ind(i,j,nB)][ind(0,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(1,yy,L)]
                            +amp0*block[ind(i,j,nB)][ind(2,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(3,yy,L)]
                            +amp2*block[ind(i,j,nB)][ind(4,yy,L)]
                            +amp3*block[ind(i,j,nB)][ind(5,yy,L)]
                            ;
                    }

                    //filter inner rows
                    for(int xx = 3; xx < L-3; xx++){
                        #pragma omp simd aligned(block, inBlock:alignment)
                        for( int yy = 0; yy < L; yy++){
                            outBlock[ind(i,j,nB)][ind(xx,yy,L)] =  
                             amp3*block[ind(i,j,nB)][ind(xx-3,yy,L)]
                            +amp2*block[ind(i,j,nB)][ind(xx-2,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(xx-1,yy,L)]
                            +amp0*block[ind(i,j,nB)][ind(xx,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(xx+1,yy,L)]
                            +amp2*block[ind(i,j,nB)][ind(xx+2,yy,L)]
                            +amp3*block[ind(i,j,nB)][ind(xx+3,yy,L)]
                            ;
                        }
                    }
                
                    //filter last rows (fetch from blocks below)
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for( int yy = 0; yy < L; yy++){
                        outBlock[ind(i,j,nB)][ind(L-1,yy,L)] =  
                        +amp3*block[ind(i+1,j,nB)][ind(2,yy,L)]
                        +amp2*block[ind(i+1,j,nB)][ind(1,yy,L)]
                        +amp1*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        +amp0*block[ind(i,j,nB)][ind(L-1,yy,L)]
                        +amp1*block[ind(i,j,nB)][ind(L-2,yy,L)]
                        +amp2*block[ind(i,j,nB)][ind(L-3,yy,L)]
                        +amp3*block[ind(i,j,nB)][ind(L-4,yy,L)];
                        
                        outBlock[ind(i,j,nB)][ind(L-2,yy,L)] = 
                        +amp3*block[ind(i+1,j,nB)][ind(1,yy,L)]
                        +amp2*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        +amp1*block[ind(i,j,nB)][ind(L-1,yy,L)]
                        +amp0*block[ind(i,j,nB)][ind(L-2,yy,L)]
                        +amp1*block[ind(i,j,nB)][ind(L-3,yy,L)]
                        +amp2*block[ind(i,j,nB)][ind(L-4,yy,L)]
                        +amp3*block[ind(i,j,nB)][ind(L-5,yy,L)];

                        outBlock[ind(i,j,nB)][ind(L-3,yy,L)] = 
                        +amp3*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        +amp2*block[ind(i,j,nB)][ind(L-1,yy,L)]
                        +amp1*block[ind(i,j,nB)][ind(L-2,yy,L)]
                        +amp0*block[ind(i,j,nB)][ind(L-3,yy,L)]
                        +amp1*block[ind(i,j,nB)][ind(L-4,yy,L)]
                        +amp2*block[ind(i,j,nB)][ind(L-5,yy,L)]
                        +amp3*block[ind(i,j,nB)][ind(L-6,yy,L)];
                    }

                }
            }
            
            //^filter last row blocks, i=nB-1                  
            for(int j = 0; j < nB; j++){
                //filter first rows (fetch from previous blocks)
                #pragma omp simd aligned(block, inBlock:alignment)
                for(int yy = 0; yy < L; yy ++){
                    outBlock[ind(nB-1,j,nB)][ind(0,yy,L)] =  
                        +amp3*block[ind(nB-2,j,nB)][ind(L-3,yy,L)]
                        +amp2*block[ind(nB-2,j,nB)][ind(L-2,yy,L)]
                        +amp1*block[ind(nB-2,j,nB)][ind(L-1,yy,L)]
                        +amp0*block[ind(nB-1,j,nB)][ind(0,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(1,yy,L)]
                        +amp2*block[ind(nB-1,j,nB)][ind(2,yy,L)]
                        +amp3*block[ind(nB-1,j,nB)][ind(3,yy,L)]
                        ;
                    
                    outBlock[ind(nB-1,j,nB)][ind(1,yy,L)] =  
                        +amp3*block[ind(nB-2,j,nB)][ind(L-2,yy,L)]
                        +amp2*block[ind(nB-2,j,nB)][ind(L-1,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(0,yy,L)]
                        +amp0*block[ind(nB-1,j,nB)][ind(1,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(2,yy,L)]
                        +amp2*block[ind(nB-1,j,nB)][ind(3,yy,L)]
                        +amp3*block[ind(nB-1,j,nB)][ind(4,yy,L)]
                        ;

                    outBlock[ind(nB-1,j,nB)][ind(2,yy,L)] =  
                        +amp3*block[ind(nB-2,j,nB)][ind(L-1,yy,L)]
                        +amp2*block[ind(nB-1,j,nB)][ind(0,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(1,yy,L)]
                        +amp0*block[ind(nB-1,j,nB)][ind(2,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(3,yy,L)]
                        +amp2*block[ind(nB-1,j,nB)][ind(4,yy,L)]
                        +amp3*block[ind(nB-1,j,nB)][ind(5,yy,L)]
                        ;
                }

                //filter inner rows
                for(int xx = 3; xx < L-3; xx++){
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for( int yy = 0; yy < L; yy++){
                        outBlock[ind(nB-1,j,nB)][ind(xx,yy,L)] =  
                         amp3*block[ind(nB-1,j,nB)][ind(xx-3,yy,L)]
                        +amp2*block[ind(nB-1,j,nB)][ind(xx-2,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(xx-1,yy,L)]
                        +amp0*block[ind(nB-1,j,nB)][ind(xx,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(xx+1,yy,L)]
                        +amp2*block[ind(nB-1,j,nB)][ind(xx+2,yy,L)]
                        +amp3*block[ind(nB-1,j,nB)][ind(xx+3,yy,L)]
                        ;
                    }
                }
                
                //filter last rows (assume input zero outside boundary)
                #pragma omp simd aligned(block, inBlock:alignment)
                for( int yy = 0; yy < L; yy++){
                    outBlock[ind(nB-1,j,nB)][ind(L-1,yy,L)] = 
                    +amp3*blockBound1[j][ind(2,yy,L)]
                    +amp2*blockBound1[j][ind(1,yy,L)]
                    +amp1*blockBound1[j][ind(0,yy,L)]
                    +amp0*block[ind(nB-1,j,nB)][ind(L-1,yy,L)]
                    +amp1*block[ind(nB-1,j,nB)][ind(L-2,yy,L)]
                    +amp2*block[ind(nB-1,j,nB)][ind(L-3,yy,L)]
                    +amp3*block[ind(nB-1,j,nB)][ind(L-4,yy,L)];
                    
                    outBlock[ind(nB-1,j,nB)][ind(L-2,yy,L)] = 
                    +amp3*blockBound1[j][ind(1,yy,L)]
                    +amp2*blockBound1[j][ind(0,yy,L)]
                    +amp1*block[ind(nB-1,j,nB)][ind(L-1,yy,L)]
                    +amp0*block[ind(nB-1,j,nB)][ind(L-2,yy,L)]
                    +amp1*block[ind(nB-1,j,nB)][ind(L-3,yy,L)]
                    +amp2*block[ind(nB-1,j,nB)][ind(L-4,yy,L)]
                    +amp3*block[ind(nB-1,j,nB)][ind(L-5,yy,L)]
                    ;

                    outBlock[ind(nB-1,j,nB)][ind(L-3,yy,L)] = 
                    +amp3*blockBound1[j][ind(0,yy,L)]
                    +amp2*block[ind(nB-1,j,nB)][ind(L-1,yy,L)]
                    +amp1*block[ind(nB-1,j,nB)][ind(L-2,yy,L)]
                    +amp0*block[ind(nB-1,j,nB)][ind(L-3,yy,L)]
                    +amp1*block[ind(nB-1,j,nB)][ind(L-4,yy,L)]
                    +amp2*block[ind(nB-1,j,nB)][ind(L-5,yy,L)]
                    +amp3*block[ind(nB-1,j,nB)][ind(L-6,yy,L)]
                    ;
                }
     
            }

        
        }
        else{  
              //^filter first row blocks, i=0
            for(int j = 0; j < nB; j++){
                //filter first rows (assume input zero outside boundary)
                #pragma omp simd aligned(block, inBlock:alignment)
                for( int yy = 0; yy < L; yy++){
                    outBlock[ind(0,j,nB)][ind(0,yy,L)] +=  
                    +amp3*blockBound2[j][ind(2,yy,L)]
                    +amp2*blockBound2[j][ind(1,yy,L)]
                    +amp1*blockBound2[j][ind(0,yy,L)]
                    +amp0*block[ind(0,j,nB)][ind(0,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(1,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(2,yy,L)]
                    +amp3*block[ind(0,j,nB)][ind(3,yy,L)];
                    
                    outBlock[ind(0,j,nB)][ind(1,yy,L)] += 
                    +amp3*blockBound2[j][ind(1,yy,L)]
                    +amp2*blockBound2[j][ind(0,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(0,yy,L)]
                    +amp0*block[ind(0,j,nB)][ind(1,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(2,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(3,yy,L)]
                    +amp3*block[ind(0,j,nB)][ind(4,yy,L)]
                    ;

                    outBlock[ind(0,j,nB)][ind(2,yy,L)] += 
                    +amp3*blockBound2[j][ind(0,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(0,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(1,yy,L)]
                    +amp0*block[ind(0,j,nB)][ind(2,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(3,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(4,yy,L)]
                    +amp3*block[ind(0,j,nB)][ind(5,yy,L)]
                    ;
                }

                //filter inner rows
                for(int xx = 3; xx < L-3; xx++){
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for( int yy = 0; yy < L; yy++){
                        outBlock[ind(0,j,nB)][ind(xx,yy,L)] +=  
                         amp3*block[ind(0,j,nB)][ind(xx-3,yy,L)]
                        +amp2*block[ind(0,j,nB)][ind(xx-2,yy,L)]
                        +amp1*block[ind(0,j,nB)][ind(xx-1,yy,L)]
                        +amp0*block[ind(0,j,nB)][ind(xx,yy,L)]
                        +amp1*block[ind(0,j,nB)][ind(xx+1,yy,L)]
                        +amp2*block[ind(0,j,nB)][ind(xx+2,yy,L)]
                        +amp3*block[ind(0,j,nB)][ind(xx+3,yy,L)]
                        ;
                    }
                }
            
                //filter last rows (fetch from blocks below)
                #pragma omp simd aligned(block, inBlock:alignment)
                for( int yy = 0; yy < L; yy++){
                    outBlock[ind(0,j,nB)][ind(L-1,yy,L)] +=  
                    +amp3*block[ind(1,j,nB)][ind(2,yy,L)]
                    +amp2*block[ind(1,j,nB)][ind(1,yy,L)]
                    +amp1*block[ind(1,j,nB)][ind(0,yy,L)]
                    +amp0*block[ind(0,j,nB)][ind(L-1,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(L-2,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(L-3,yy,L)]
                    +amp3*block[ind(0,j,nB)][ind(L-4,yy,L)];
                    
                    outBlock[ind(0,j,nB)][ind(L-2,yy,L)] += 
                    +amp3*block[ind(1,j,nB)][ind(1,yy,L)]
                    +amp2*block[ind(1,j,nB)][ind(0,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(L-1,yy,L)]
                    +amp0*block[ind(0,j,nB)][ind(L-2,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(L-3,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(L-4,yy,L)]
                    +amp3*block[ind(0,j,nB)][ind(L-5,yy,L)];

                    outBlock[ind(0,j,nB)][ind(L-3,yy,L)] += 
                    +amp3*block[ind(1,j,nB)][ind(0,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(L-1,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(L-2,yy,L)]
                    +amp0*block[ind(0,j,nB)][ind(L-3,yy,L)]
                    +amp1*block[ind(0,j,nB)][ind(L-4,yy,L)]
                    +amp2*block[ind(0,j,nB)][ind(L-5,yy,L)]
                    +amp3*block[ind(0,j,nB)][ind(L-6,yy,L)];
                }

            }

            //^filter inner blocks
            for(int i = 1; i < nB - 1; i++){
                for(int j = 0; j <nB; j++){
                    //apply manually to first rows: fetch from previous block
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for(int yy = 0; yy < L; yy ++){
                        outBlock[ind(i,j,nB)][ind(0,yy,L)] +=  
                            +amp3*block[ind(i-1,j,nB)][ind(L-3,yy,L)]
                            +amp2*block[ind(i-1,j,nB)][ind(L-2,yy,L)]
                            +amp1*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                            +amp0*block[ind(i,j,nB)][ind(0,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(1,yy,L)]
                            +amp2*block[ind(i,j,nB)][ind(2,yy,L)]
                            +amp3*block[ind(i,j,nB)][ind(3,yy,L)]
                            ;
                        
                        outBlock[ind(i,j,nB)][ind(1,yy,L)] +=  
                            +amp3*block[ind(i-1,j,nB)][ind(L-2,yy,L)]
                            +amp2*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(0,yy,L)]
                            +amp0*block[ind(i,j,nB)][ind(1,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(2,yy,L)]
                            +amp2*block[ind(i,j,nB)][ind(3,yy,L)]
                            +amp3*block[ind(i,j,nB)][ind(4,yy,L)]
                            ;

                        outBlock[ind(i,j,nB)][ind(2,yy,L)] +=  
                            +amp3*block[ind(i-1,j,nB)][ind(L-1,yy,L)]
                            +amp2*block[ind(i,j,nB)][ind(0,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(1,yy,L)]
                            +amp0*block[ind(i,j,nB)][ind(2,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(3,yy,L)]
                            +amp2*block[ind(i,j,nB)][ind(4,yy,L)]
                            +amp3*block[ind(i,j,nB)][ind(5,yy,L)]
                            ;
                    }

                    //filter inner rows
                    for(int xx = 3; xx < L-3; xx++){
                        #pragma omp simd aligned(block, inBlock:alignment)
                        for( int yy = 0; yy < L; yy++){
                            outBlock[ind(i,j,nB)][ind(xx,yy,L)] +=  
                             amp3*block[ind(i,j,nB)][ind(xx-3,yy,L)]
                            +amp2*block[ind(i,j,nB)][ind(xx-2,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(xx-1,yy,L)]
                            +amp0*block[ind(i,j,nB)][ind(xx,yy,L)]
                            +amp1*block[ind(i,j,nB)][ind(xx+1,yy,L)]
                            +amp2*block[ind(i,j,nB)][ind(xx+2,yy,L)]
                            +amp3*block[ind(i,j,nB)][ind(xx+3,yy,L)]
                            ;
                        }
                    }
                
                    //filter last rows (fetch from blocks below)
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for( int yy = 0; yy < L; yy++){
                        outBlock[ind(i,j,nB)][ind(L-1,yy,L)] +=  
                        +amp3*block[ind(i+1,j,nB)][ind(2,yy,L)]
                        +amp2*block[ind(i+1,j,nB)][ind(1,yy,L)]
                        +amp1*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        +amp0*block[ind(i,j,nB)][ind(L-1,yy,L)]
                        +amp1*block[ind(i,j,nB)][ind(L-2,yy,L)]
                        +amp2*block[ind(i,j,nB)][ind(L-3,yy,L)]
                        +amp3*block[ind(i,j,nB)][ind(L-4,yy,L)];
                        
                        outBlock[ind(i,j,nB)][ind(L-2,yy,L)] += 
                        +amp3*block[ind(i+1,j,nB)][ind(1,yy,L)]
                        +amp2*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        +amp1*block[ind(i,j,nB)][ind(L-1,yy,L)]
                        +amp0*block[ind(i,j,nB)][ind(L-2,yy,L)]
                        +amp1*block[ind(i,j,nB)][ind(L-3,yy,L)]
                        +amp2*block[ind(i,j,nB)][ind(L-4,yy,L)]
                        +amp3*block[ind(i,j,nB)][ind(L-5,yy,L)];

                        outBlock[ind(i,j,nB)][ind(L-3,yy,L)] += 
                        +amp3*block[ind(i+1,j,nB)][ind(0,yy,L)]
                        +amp2*block[ind(i,j,nB)][ind(L-1,yy,L)]
                        +amp1*block[ind(i,j,nB)][ind(L-2,yy,L)]
                        +amp0*block[ind(i,j,nB)][ind(L-3,yy,L)]
                        +amp1*block[ind(i,j,nB)][ind(L-4,yy,L)]
                        +amp2*block[ind(i,j,nB)][ind(L-5,yy,L)]
                        +amp3*block[ind(i,j,nB)][ind(L-6,yy,L)];
                    }

                }
            }
            
            //^filter last row blocks, i=nB-1                  
            for(int j = 0; j < nB; j++){
                //filter first rows (fetch from previous blocks)
                #pragma omp simd aligned(block, inBlock:alignment)
                for(int yy = 0; yy < L; yy ++){
                    outBlock[ind(nB-1,j,nB)][ind(0,yy,L)] +=  
                        +amp3*block[ind(nB-2,j,nB)][ind(L-3,yy,L)]
                        +amp2*block[ind(nB-2,j,nB)][ind(L-2,yy,L)]
                        +amp1*block[ind(nB-2,j,nB)][ind(L-1,yy,L)]
                        +amp0*block[ind(nB-1,j,nB)][ind(0,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(1,yy,L)]
                        +amp2*block[ind(nB-1,j,nB)][ind(2,yy,L)]
                        +amp3*block[ind(nB-1,j,nB)][ind(3,yy,L)]
                        ;
                    
                    outBlock[ind(nB-1,j,nB)][ind(1,yy,L)] +=  
                        +amp3*block[ind(nB-2,j,nB)][ind(L-2,yy,L)]
                        +amp2*block[ind(nB-2,j,nB)][ind(L-1,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(0,yy,L)]
                        +amp0*block[ind(nB-1,j,nB)][ind(1,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(2,yy,L)]
                        +amp2*block[ind(nB-1,j,nB)][ind(3,yy,L)]
                        +amp3*block[ind(nB-1,j,nB)][ind(4,yy,L)]
                        ;

                    outBlock[ind(nB-1,j,nB)][ind(2,yy,L)] +=  
                        +amp3*block[ind(nB-2,j,nB)][ind(L-1,yy,L)]
                        +amp2*block[ind(nB-1,j,nB)][ind(0,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(1,yy,L)]
                        +amp0*block[ind(nB-1,j,nB)][ind(2,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(3,yy,L)]
                        +amp2*block[ind(nB-1,j,nB)][ind(4,yy,L)]
                        +amp3*block[ind(nB-1,j,nB)][ind(5,yy,L)]
                        ;
                }

                //filter inner rows
                for(int xx = 3; xx < L-3; xx++){
                    #pragma omp simd aligned(block, inBlock:alignment)
                    for( int yy = 0; yy < L; yy++){
                        outBlock[ind(nB-1,j,nB)][ind(xx,yy,L)] +=  
                         amp3*block[ind(nB-1,j,nB)][ind(xx-3,yy,L)]
                        +amp2*block[ind(nB-1,j,nB)][ind(xx-2,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(xx-1,yy,L)]
                        +amp0*block[ind(nB-1,j,nB)][ind(xx,yy,L)]
                        +amp1*block[ind(nB-1,j,nB)][ind(xx+1,yy,L)]
                        +amp2*block[ind(nB-1,j,nB)][ind(xx+2,yy,L)]
                        +amp3*block[ind(nB-1,j,nB)][ind(xx+3,yy,L)]
                        ;
                    }
                }
                
                //filter last rows (assume input zero outside boundary)
                #pragma omp simd aligned(block, inBlock:alignment)
                for( int yy = 0; yy < L; yy++){
                    outBlock[ind(nB-1,j,nB)][ind(L-1,yy,L)] +=  
                    +amp3*blockBound1[j][ind(2,yy,L)]
                    +amp2*blockBound1[j][ind(1,yy,L)]
                    +amp1*blockBound1[j][ind(0,yy,L)]
                    +amp0*block[ind(nB-1,j,nB)][ind(L-1,yy,L)]
                    +amp1*block[ind(nB-1,j,nB)][ind(L-2,yy,L)]
                    +amp2*block[ind(nB-1,j,nB)][ind(L-3,yy,L)]
                    +amp3*block[ind(nB-1,j,nB)][ind(L-4,yy,L)];
                    
                    outBlock[ind(nB-1,j,nB)][ind(L-2,yy,L)] += 
                    +amp3*blockBound1[j][ind(1,yy,L)]
                    +amp2*blockBound1[j][ind(0,yy,L)]
                    +amp1*block[ind(nB-1,j,nB)][ind(L-1,yy,L)]
                    +amp0*block[ind(nB-1,j,nB)][ind(L-2,yy,L)]
                    +amp1*block[ind(nB-1,j,nB)][ind(L-3,yy,L)]
                    +amp2*block[ind(nB-1,j,nB)][ind(L-4,yy,L)]
                    +amp3*block[ind(nB-1,j,nB)][ind(L-5,yy,L)]
                    ;

                    outBlock[ind(nB-1,j,nB)][ind(L-3,yy,L)] += 
                    +amp3*blockBound1[j][ind(0,yy,L)]
                    +amp2*block[ind(nB-1,j,nB)][ind(L-1,yy,L)]
                    +amp1*block[ind(nB-1,j,nB)][ind(L-2,yy,L)]
                    +amp0*block[ind(nB-1,j,nB)][ind(L-3,yy,L)]
                    +amp1*block[ind(nB-1,j,nB)][ind(L-4,yy,L)]
                    +amp2*block[ind(nB-1,j,nB)][ind(L-5,yy,L)]
                    +amp3*block[ind(nB-1,j,nB)][ind(L-6,yy,L)]
                    ;
                }
     
            }


        }
     
    }

    //convert output to desired format-------------------------

    //transpose:
    for(int i = 0; i < nB; i++){
        for(int j = 0; j < i; j++){
            swap(outBlock[ind(i,j,nB)],outBlock[ind(j,i,nB)]);
        }
    }

    for(int i = 0; i < nB; i++){
        for(int j = 0; j < nB; j++){
            transpose(outBlock[ind(i,j,nB)]);
        }
    }

    
    //copy
    for(int i = 0; i < nB; i++){
        for(int j = 0; j < nB; j++){
            for(int xx = 0; xx < L; xx++){
                for(int yy = 0; yy < L; yy++){
                    buffer[ind(i*L + xx, j*L + yy, N)] = outBlock[ind(i,j,nB)][ind(xx,yy,L)];
                }
            }
        }
    }
    
    swap(out,buffer);

    return out;
};



int main(int argc, char **argv){
    int N = argc < 2 ? 4096 : atoi(argv[1]);
    int ng= 7;
    int i,j,x,y;

    float A[ng] = {0.0007529296964615122,0.0008367529266638894,0.0006450617864747959,
    0.0005213929956621048,0.0003981768137922748,0.00032428002360665784,0.0001996767865228412};
    float Sx[ng] = {0.9256926546479348,1.66591945428758,3.241463860181495,5.347072133914192,
    8.752160665039751,12.895919059006964,24.433838861919735};
    float Sy[ng] = {0.6545635533961315,1.1779829430373405,2.2920610765054583,
    3.7809509653843483,6.1887121562837715,9.118791816256666,17.27733314968284};
    
    //ng = 5; //using only 5 Gaussian terms -> below 1% norm2 error

    /*Initialize Image */
    
    float *I = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *O;
    for(i=1;i<N*N;i++){
        //I[i] = rand();
        I[i] = 0;
    }
    //all impulses
    I[ind(N/2,N/2,N)] = 1.0;
    I[ind(0,0,N)] = 1.0;
    I[ind(N-1,N-1,N)] = 1.0;

    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    int it = 1; //CAUTION: ITERATIONS CHANGE INPUT ORDERING TO BLOCK ORDERING

    float derichet =1e10;

    for(i = 0; i<it ; i++){
        start = high_resolution_clock::now();
        O = dericheFilter(I, A, Sx, Sy, ng, N);
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        derichet = min(derichet,(duration.count())/1000000.f);
        //cout << "Deriche1 filter: " << float(duration.count())/1000000 << " seconds" << endl;
    }

    printf("%f",derichet);

    FILE *f;
    f = fopen("kernel.csv", "w");
    for(int x=0;x<N;x++){
        for(int y=0;y<N-1;y++)
            fprintf(f,"%f,",O[ind(x,y,N)]);
        fprintf(f,"%f\n",O[ind(x,N-1,N)]);
    }
    
    return 0;
}