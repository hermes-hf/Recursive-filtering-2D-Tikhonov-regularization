#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <omp.h>
#include <Eigen/Sparse>


#define ind(x,y,n) ((x)*(n) + (y))
#define indt(x,y,n) ((y)*(n) + (x))

#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(a,b) (((a)>(b))?(a):(b))
#define alignment 64 //number of bytes for SIMD alignment

#define L 64

float eps = 0.0;

using namespace std;
using namespace std::chrono;

//TIME MACROS----------------------------------------------
#define REPORT_TIME

#ifdef REPORT_TIME
#define CLOCK_VAR() __clock_start
#define INIT_CLOCK() auto __clock_start = high_resolution_clock::now();
#define START_CLOCK() __clock_start = high_resolution_clock::now()
#define STOP_CLOCK_AND_REPORT_TIME(DESCRIPTION)                         \
    printf("%40s :: % 6.2f ms\n",                                       \
           DESCRIPTION,                                                 \
           duration_cast<microseconds>(high_resolution_clock::now() - __clock_start).count() / 1000.0f)
#define STOP_CLOCK_AND_GET_TIME(DESCRIPTION) (duration_cast<microseconds>(high_resolution_clock::now() - __clock_start).count() / 1000.0f)

#define TIMER_PRINTF(...) printf(__VA_ARGS__)
#else
#define CLOCK_VAR()
#define INIT_CLOCK() (void)0
#define START_CLOCK() (void)0
#define STOP_CLOCK_AND_REPORT_TIME(DESCRIPTION) (void)0
#define TIMER_PRINTF(...) (void)0
#endif


//---------------------------------------------------------

class tikhVariables{
public:
    float *pB;
    float *pF;
    float *pBh;
    float *pFh;
    float *B;
    float *F;
    float *normX;
    float *pD;
    float *pU;
    float *pDh;
    float *pUh; 
    float *D;
    float *U;
    float *normY;
    float *H;
    float *i_TygH;
    float rate;
    void freevars(){
        free(pB);
        free(pF);
        free(B);
        free(F);
        free(normX);
        //free(pD); //causes numerical error?
        free(pU);
        free(D);
        free(U);
        free(normY);
        free(H);
        free(i_TygH);
    }
};

static inline float weightfun(float i0, float i1, float mu, float lambda){
    //return lambda; //TODO:::
    return eps + lambda*powf(abs(1.00 - abs(i0 - i1)),mu);
}

//in-place matrix transpose (buffered)
void buf_transpose(float* M, float* buf1, float* buf2, int N) {
    for (int i = 0; i < N; i += L) {
        for (int j = 0; j <= i; j += L) {

            int maxii = min(L, N - i);
            int maxjj = min(L, N - j);

            for (int ii = 0; ii < maxii; ii++) {
                for (int jj = 0; jj < maxjj; jj++) {
                    buf1[ind(jj, ii, L)] = M[ind(i + ii, j + jj, N)];
                }
            }

            for (int jj = 0; jj < maxjj; jj++) {
                for (int ii = 0; ii < maxii; ii++) {
                    buf2[ind(ii, jj, L)] = M[ind(j + jj, i + ii, N)];
                    M[ind(j + jj, i + ii, N)] = buf1[ind(jj, ii, L)];
                }
            }

            for (int ii = 0; ii < maxii; ii++) {
                for (int jj = 0; jj < maxjj; jj++) {
                    M[ind(i + ii, j + jj, N)] = buf2[ind(ii, jj, L)];
                }
            }

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


void computeWx_Wy(float *in, int N, float* temp1, float* temp2, float *Wx, float *Wy, float mu, float lambdax, float lambday){
    float *inT =  (float*)aligned_alloc(alignment,sizeof(float) * N*N); //transpose of input (non-blocked)

    //Compute Wx,Wyt
    for(int i=0; i<N*N; i++)
        inT[i] = in[i];

    buf_transpose(inT, temp1, temp2, N);

    for(int y=0; y<N; y++){
            Wx[ind(0,y,N)]  = eps;
            Wy[ind(0,y,N)] = eps;
    }

    //LONGEST EXECUTION TIME
    for(int x=1; x<N; x++){
        #pragma omp simd aligned(in,Wx,Wy,inT:alignment)
        for(int y=0; y<N; y++){
            Wx[ind(x,y,N)]  = weightfun( in[ind(x,y,N)], in[ind(x-1,y,N)],mu, lambdax);
            Wy[ind(x,y,N)] = weightfun(inT[ind(x,y,N)],inT[ind(x-1,y,N)],mu, lambday);
        }
    }

    //untranspose Wy
    buf_transpose(Wy, temp1, temp2, N);

    free(inT);

}


tikhVariables tihkonovVariables(float* in, int N, float mu, float lambdax, float lambday, float *temp1, float *temp2, float *Wx, float *Wy){

    //------------------------------------------------------------------------------------
    //Pre-compute buffers ----------------------------------------------------------------
    //------------------------------------------------------------------------------------
   
    //compute H
    float *H =  (float*)aligned_alloc(alignment,sizeof(float) * N*N);

    
    for(int x=0; x<N; x++){
        #pragma omp simd aligned(Wx,Wy,H:alignment)
        for(int y=0; y<N; y++){
            H[ind(x,y,N)]  =  1 +  Wx[ind(x,y,N)]+ Wy[ind(x,y,N)];
        }
    }


    for(int x=0; x<N-1; x++){
        #pragma omp simd aligned(Wx,H:alignment)
        for(int y=0; y<N; y++){
            H[ind(x,y,N)]  += Wx[ind(x+1,y,N)];
        }
    }

    for(int x=0; x<N; x++){
        #pragma omp simd aligned(Wx,Wy,H:alignment)
        for(int y=0; y<N-1; y++){
            H[ind(x,y,N)]  += Wy[ind(x,y+1,N)];
        }
    }


    //compute probabilities X
    float *pF =  (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *pB =  (float*)aligned_alloc(alignment,sizeof(float) * N*N);

    
    for(int x=0; x<N; x++){
        #pragma omp simd aligned(Wx,pB,H:alignment)
        for(int y=0; y<N; y++){
           pB[ind(x,y,N)]  = Wx[ind(x,y,N)]/H[ind(x,y,N)];
        }
        
    }


    for(int x=0; x<N-1; x++){
        #pragma omp simd aligned(Wx,pF,H:alignment)
        for(int y=0; y<N; y++){
           pF[ind(x,y,N)]  = Wx[ind(x+1,y,N)]/H[ind(x,y,N)];
        } 
    }
    
    
    for(int y=0; y<N; y++){
           pF[ind(N-1,y,N)]  = 0;
    } 


    //compute normalization X
    float *normX = (float*)aligned_alloc(alignment,sizeof(float) * N*N);

    for(int y=0; y<N; y++){
            normX[ind(N-1,y,N)] = 1;
    }

    for(int x=N-2; x>=0; x=x-1){
        #pragma omp simd aligned(normX,pB,pF:alignment)
        for(int y=0; y<N; y++){
            normX[ind(x,y,N)] = 1 - pB[ind(x+1,y,N)]*pF[ind(x,y,N)]/normX[ind(x+1,y,N)];
        }
    }

    //compute F
    float *F = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    for(int y=0; y<N; y++)
        F[ind(0,y,N)] = 0;
    for(int x=1; x<N; x++){
        #pragma omp simd aligned(F,pB,normX:alignment)
        for(int y=0; y<N; y++){
            F[ind(x,y,N)] = pB[ind(x,y,N)]/normX[ind(x-1,y,N)];
        }
    }

    //compute B
    float *B = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    for(int y=0; y<N; y++)
        B[ind(N-1,y,N)] = 0;
    for(int x=0; x<N-1; x++){
        #pragma omp simd aligned(normX,B,pF:alignment)
        for(int y=0; y<N; y++){
            B[ind(x,y,N)] = pF[ind(x,y,N)]/normX[ind(x+1,y,N)];
        }
    }

    //compute probabilities Y
    float *pU =  (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *pD =  (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    
    for(int x=0; x<N; x++){
        #pragma omp simd aligned(Wy,pD,H:alignment)
        for(int y=0; y<N; y++){
           pD[ind(x,y,N)]  = Wy[ind(x,y,N)]/H[ind(x,y,N)];
        }
    }

    for(int x=0; x<N; x++){
        pU[ind(x,N-1,N)]  = 0;
        #pragma omp simd aligned(Wx,pU,H:alignment)
        for(int y=0; y<N-1; y++){
            pU[ind(x,y,N)]  = Wy[ind(x,y+1,N)]/H[ind(x,y,N)];
        }
    }

    //compute Normalization Y
    float *normY = (float*)aligned_alloc(alignment,sizeof(float) * N*N);

    for(int x=0; x<N; x++){
            normY[ind(N-1,x,N)] = 1;
    }

    buf_transpose(pU, temp1, temp2, N);
    buf_transpose(pD, temp1, temp2, N);

    for(int x=N-2; x>=0; x=x-1){
        #pragma omp simd aligned(normY,pU,pD:alignment) //-gera problemas
        for(int y=0; y<N; y++){
            normY[ind(x,y,N)] = 1 - pD[ind(x+1,y,N)]*pU[ind(x,y,N)]/normY[ind(x+1,y,N)];
        }
    }


    //compute U
    float *U = (float*)aligned_alloc(alignment,sizeof(float) * N*N);

    U[0] = 0;
    for(int y=1; y<N; y++){
        U[indt(y,0,N)] = 0;
        #pragma omp simd aligned(U,pD,normY:alignment)
        for(int x=0; x<N; x++){
            U[indt(x,y,N)] = pD[indt(x,y,N)]/normY[indt(x,y-1,N)];
        }
    }


    //compute D
    float *D = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    for(int y=0; y<N-1; y++){
        #pragma omp simd aligned(D,pU,normY:alignment)
        for(int x=0; x<N; x++){
            D[indt(x,y,N)] = pU[indt(x,y,N)]/normY[indt(x,y+1,N)];
        }
        D[indt(y,N-1,N)] = 0;
    }
    D[indt(N-1,N-1,N)] = 0;


    //all required buffers are pre-computed at this point

    tikhVariables output;
    output.pB = pB;
    output.pF = pF;
    output.B = B;
    output.F = F;
    output.normX = normX;
    output.pD = pD;
    output.pU = pU;
    output.D = D;
    output.U = U;
    output.normY = normY;
    output.H = H;

    return output;
}



tikhVariables tihkonovVariablesDirect(float* in, int N, float mu, float lambdax, float lambday, float *temp1, float *temp2, float *Wx, float *Wy){

    //------------------------------------------------------------------------------------
    //Pre-compute buffers ----------------------------------------------------------------
    //------------------------------------------------------------------------------------
   
    //compute H
    float *H =  (float*)aligned_alloc(alignment,sizeof(float) * N*N);

    
    for(int x=0; x<N; x++){
        #pragma omp simd aligned(Wx,Wy,H:alignment)
        for(int y=0; y<N; y++){
            H[ind(x,y,N)]  =  1 +  Wx[ind(x,y,N)]+ Wy[ind(x,y,N)];
        }
    }


    for(int x=0; x<N-1; x++){
        #pragma omp simd aligned(Wx,H:alignment)
        for(int y=0; y<N; y++){
            H[ind(x,y,N)]  += Wx[ind(x+1,y,N)];
        }
    }

    for(int x=0; x<N; x++){
        #pragma omp simd aligned(Wx,Wy,H:alignment)
        for(int y=0; y<N-1; y++){
            H[ind(x,y,N)]  += Wy[ind(x,y+1,N)];
        }
    }


    //compute probabilities X
    float *pF =  (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *pB =  (float*)aligned_alloc(alignment,sizeof(float) * N*N);

    
    for(int x=0; x<N; x++){
        #pragma omp simd aligned(Wx,pB,H:alignment)
        for(int y=0; y<N; y++){
           pB[ind(x,y,N)]  = Wx[ind(x,y,N)]/H[ind(x,y,N)];
        }
        
    }


    for(int x=0; x<N-1; x++){
        #pragma omp simd aligned(Wx,pF,H:alignment)
        for(int y=0; y<N; y++){
           pF[ind(x,y,N)]  = Wx[ind(x+1,y,N)]/H[ind(x,y,N)];
        } 
    }
    
    
    for(int y=0; y<N; y++){
           pF[ind(N-1,y,N)]  = 0;
    } 

    //compute probabilities Y
    float *pU =  (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *pD =  (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    
    for(int x=0; x<N; x++){
        #pragma omp simd aligned(Wy,pD,H:alignment)
        for(int y=0; y<N; y++){
           pD[ind(x,y,N)]  = Wy[ind(x,y,N)]/H[ind(x,y,N)];
        }
    }

    for(int x=0; x<N; x++){
        pU[ind(x,N-1,N)]  = 0;
        #pragma omp simd aligned(Wx,pU,H:alignment)
        for(int y=0; y<N-1; y++){
            pU[ind(x,y,N)]  = Wy[ind(x,y+1,N)]/H[ind(x,y,N)];
        }
    }


   
    tikhVariables output;
    output.pB = pB;
    output.pF = pF;
    output.pD = pD;
    output.pU = pU;
    output.H = H;

    return output;
}


void tikhonovXAxis(float *input, float *output, int N, tikhVariables tV){
    float *pF = tV.pF;
    float *pB= tV.pB;
    float *F= tV.F;
    float *B= tV.B; 
    float *normX= tV.normX;

    //^Normalize
    {
        for(int x=0; x<1; x++){
            #pragma omp simd aligned(output,pF,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pF[ind(x,y,N)]*input[ind(x+1,y,N)];
            }
        }

        for(int x=1; x<N-1; x++){
            #pragma omp simd aligned(output,pF,pB,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pB[ind(x,y,N)]*input[ind(x-1,y,N)] + pF[ind(x,y,N)]*input[ind(x+1,y,N)];
            }
        }

        for(int x=N-1; x<N; x++){
            #pragma omp simd aligned(output,pB,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pB[ind(x,y,N)]*input[ind(x-1,y,N)];
            }
        }

    }
    
    //^Backward filter
    for(int x=N-2; x >= 0; x= x-1){
        #pragma omp simd aligned(output,B:alignment)
        for(int y=0;y<N;y++){
            output[ind(x,y,N)] = output[ind(x,y,N)] + B[ind(x,y,N)] * output[ind(x+1,y,N)];
        }
    }

    //^Forward filter
    for(int x=1;x<N;x++){
        #pragma omp simd aligned(output,F:alignment)
        for(int y=0; y<N;y++){
            output[ind(x,y,N)] = output[ind(x,y,N)] + F[ind(x,y,N)]*output[ind(x-1,y,N)];
        }
    }

    //^Normalize
    for(int x=0;x<N;x++){
        #pragma omp simd aligned(output,normX:alignment)
        for(int y=0; y<N;y++){
            output[ind(x,y,N)] = output[ind(x,y,N)]/normX[ind(x,y,N)];
        }
    }
    
    return;
}

//assumes that input is transposed

void tikhonovYAxis(float *input, float *output, int N, tikhVariables tV){
    float *pU = tV.pU;
    float *pD= tV.pD;
    float *U= tV.U;
    float *D= tV.D; 
    float *normY= tV.normY;

    //^Normalize
    {
        for(int x=0; x<1; x++){
            #pragma omp simd aligned(output,pU,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pU[ind(x,y,N)]*input[ind(x+1,y,N)];
            }
        }

        for(int x=1; x<N-1; x++){
            #pragma omp simd aligned(output,pU,pD,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pD[ind(x,y,N)]*input[ind(x-1,y,N)] + pU[ind(x,y,N)]*input[ind(x+1,y,N)];
            }
        }

        for(int x=N-1; x<N; x++){
            #pragma omp simd aligned(output,pD,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pD[ind(x,y,N)]*input[ind(x-1,y,N)];
            }
        }

    }
    
    //^Downward filter
    for(int x=N-2; x >= 0; x= x-1){
        #pragma omp simd aligned(output,D:alignment)
        for(int y=0;y<N;y++){
            output[ind(x,y,N)] = output[ind(x,y,N)] + D[ind(x,y,N)] * output[ind(x+1,y,N)];
        }
    }

    //^Upward filter
    for(int x=1;x<N;x++){
        #pragma omp simd aligned(output,U:alignment)
        for(int y=0; y<N;y++){
            output[ind(x,y,N)] = output[ind(x,y,N)] + U[ind(x,y,N)]*output[ind(x-1,y,N)];
        }
    }

    //^Normalize
    for(int x=0;x<N;x++){
        #pragma omp simd aligned(output,normY:alignment)
        for(int y=0; y<N;y++){
            output[ind(x,y,N)] = output[ind(x,y,N)]/normY[ind(x,y,N)];
        }
    }
    
    return;
}

void mXFilter(float *input, float *output, int N, tikhVariables tV){
    float *pF = tV.pF;
    float *pB= tV.pB;

    //^Normalize
    {
        for(int x=0; x<1; x++){
            #pragma omp simd aligned(output,pF,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pF[ind(x,y,N)]*input[ind(x+1,y,N)];
            }
        }

        for(int x=1; x<N-1; x++){
            #pragma omp simd aligned(output,pF,pB,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pB[ind(x,y,N)]*input[ind(x-1,y,N)] + pF[ind(x,y,N)]*input[ind(x+1,y,N)];
            }
        }

        for(int x=N-1; x<N; x++){
            #pragma omp simd aligned(output,pB,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pB[ind(x,y,N)]*input[ind(x-1,y,N)];
            }
        }

    }
    return;
}


void mXFilterH(float *input, float *output, int N, tikhVariables tV){
    float *pF = tV.pFh;
    float *pB= tV.pBh;

    //^Normalize
    {
        for(int x=0; x<1; x++){
            #pragma omp simd aligned(output,pF,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pF[ind(x,y,N)]*input[ind(x+1,y,N)];
            }
        }

        for(int x=1; x<N-1; x++){
            #pragma omp simd aligned(output,pF,pB,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pB[ind(x,y,N)]*input[ind(x-1,y,N)] + pF[ind(x,y,N)]*input[ind(x+1,y,N)];
            }
        }

        for(int x=N-1; x<N; x++){
            #pragma omp simd aligned(output,pB,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pB[ind(x,y,N)]*input[ind(x-1,y,N)];
            }
        }

    }
    return;
}

void mYFilter(float *input, float *output, int N, tikhVariables tV){
    float *pU = tV.pU;
    float *pD= tV.pD;

    //^Normalize
    {
        for(int x=0; x<N; x++){
            for(int y=0; y<1; y++){
                output[ind(x,y,N)] = pU[ind(x,y,N)]*input[ind(x,y+1,N)];
            }

            #pragma omp simd aligned(output,pU,pD,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pD[ind(x,y,N)]*input[ind(x,y-1,N)] + pU[ind(x,y,N)]*input[ind(x,y+1,N)];
            }

            #pragma omp simd aligned(output,pD,input:alignment)
            for(int y=N-1; y<N; y++){
                output[ind(x,y,N)] = pD[ind(x,y,N)]*input[ind(x,y-1,N)];
            }
        }

    }
    return;
}


void mYFilterH(float *input, float *output, int N, tikhVariables tV){
    float *pU = tV.pUh;
    float *pD= tV.pDh;

    //^Normalize
    {
        for(int x=0; x<N; x++){
            for(int y=0; y<1; y++){
                output[ind(x,y,N)] = pU[ind(x,y,N)]*input[ind(x,y+1,N)];
            }

            #pragma omp simd aligned(output,pU,pD,input:alignment)
            for(int y=0; y<N; y++){
                output[ind(x,y,N)] = pD[ind(x,y,N)]*input[ind(x,y-1,N)] + pU[ind(x,y,N)]*input[ind(x,y+1,N)];
            }

            #pragma omp simd aligned(output,pD,input:alignment)
            for(int y=N-1; y<N; y++){
                output[ind(x,y,N)] = pD[ind(x,y,N)]*input[ind(x,y-1,N)];
            }
        }

    }
    return;
}



void computeResidue(float *in, float *buffer ,float *out, int N, tikhVariables tV, float *temp1, float *temp2){
    //Residue is R = (I+Ty)gH - (I-TyTx)f
    tikhonovXAxis(in, buffer, N, tV);
    buf_transpose(buffer, temp1, temp2, N);
    tikhonovYAxis(buffer,out,N,tV);
    buf_transpose(out, temp1, temp2, N);
    for(int i =0;i<N*N;i++)
        out[i] = tV.i_TygH[i] - (in[i] - out[i]);

    return;
}



void computeResidue2(float *in, float *buffer ,float *out, int N, tikhVariables tV, float *epsilon , float *temp1, float *temp2){
    //Residue is R = epsilon - (I-TyTx)f
    tikhonovXAxis(in, buffer, N, tV);
    buf_transpose(buffer, temp1, temp2, N);
    tikhonovYAxis(buffer,out,N,tV);
    buf_transpose(out, temp1, temp2, N);
    for(int i =0;i<N*N;i++)
        out[i] = epsilon[i] - (in[i] - out[i]);

    return;
}


void computeResidue_2(float *in, float *buffer , float *buffer2,float *out, int N, tikhVariables tV, float *temp1, float *temp2){
    //Residue is R = g - Acg f

    mXFilterH(in, buffer,N,tV);
    for(int i=0;i<N*N;i++)
        out[i] = tV.H[i]*in[i] - buffer[i]; //out = H-Mx 
    for(int i =0; i< N*N;i++)
        buffer2[i] = in[i];
    mYFilterH(buffer2, buffer, N, tV);
    for(int i=0;i<N*N;i++)
        out[i] = out[i] - buffer[i]; //out = H-Mx - My


    for(int i =0;i<N*N;i++)
        out[i] = in[i] - out[i];

    return;
}


void getiTygH(float* in, int N, tikhVariables &tV, float *buffer1,float*buffer2, float *temp1, float *temp2){
    float *i_TygH = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    //^Compute (I+Ty)gH
    {
        //normalize input and transpose
        for(int i=0;i<N*N;i++)
            buffer1[i] = in[i]/tV.H[i];

        buf_transpose(buffer1,temp1,temp2,N);

        tikhonovYAxis(buffer1,buffer2,N,tV);

        //cumulate to I+Ty
        for(int i=0;i<N*N;i++)
            buffer1[i] += buffer2[i];

        buf_transpose(buffer1,temp1,temp2,N);

        //at this point block1 = (I+Ty)gH, transfer to i_TygH
        for(int i=0;i<N*N;i++)
            i_TygH[i] = buffer1[i];
        
    }
    tV.i_TygH = i_TygH;
    return;
}

void edgeAwareTikhonovFilterPrecalc(float *g,float* in, int N, int num_it, tikhVariables tV, float *buffer1,float*buffer2, float *temp1, float *temp2){

    for(int i=0; i<N*N;i++)
        buffer1[i] = in[i];

    //^Iterate  filter
    for(int it =0; it<num_it; it++){

        tikhonovXAxis(buffer1,buffer2,N,tV);
        
        buf_transpose(buffer2,temp1,temp2,N);

        tikhonovYAxis(buffer2,buffer1,N,tV);
    
        buf_transpose(buffer1,temp1,temp2,N);

        //accumulate
        for(int i=0;i<N*N;i++)
            buffer1[i] += g[i]; //g = i_TygH if we are solving the fine problem
                                //g = residue if we are solving the coarse problem
        
    }

    return;
}



int edgeAwareTikhonovFilterPrecalcConjugateGradientsRecursive(tikhVariables tV, float* x_k, float *r_k , 
float* p_k, float *Ap_k , float *buffer , float *temp1, float *temp2, int N){
    //A = (I-TyTx)
    float rkt_rk=0;
    float p_ktApk=0;
    //compute A p_k

    tikhonovXAxis(p_k, buffer,N,tV);
    buf_transpose(buffer, temp1,temp2,N);
    tikhonovYAxis(buffer, Ap_k, N,tV);
    buf_transpose(Ap_k, temp1,temp2,N);
    for(int i=0;i<N*N;i++)
        Ap_k[i] = p_k[i] - Ap_k[i];
    //
    for(int i=0; i<N*N; i++)
        p_ktApk += p_k[i]*Ap_k[i];

   
    for(int i=0;i<N*N;i++)
        rkt_rk += r_k[i]*r_k[i];

    float alpha_k = rkt_rk/p_ktApk;

    //compute x_k
    for(int i=0; i<N*N;i++)
        x_k[i] = x_k[i] + alpha_k*p_k[i];

    for(int i=0; i<N*N;i++){
        r_k[i] = r_k[i] - alpha_k*Ap_k[i];
    }

    float beta_k = 0;
    for(int i=0;i<N*N;i++)
        beta_k += r_k[i]*r_k[i];

    //if((beta_k>= rkt_rk)){
    if(sqrtf(beta_k/rkt_rk) > tV.rate){
        //result diverging from optimal solution
        return 1;
    }
    else
        beta_k = beta_k/rkt_rk;
    
    for(int i=0;i<N*N;i++)
        p_k[i] = r_k[i] + beta_k*p_k[i];

    

    return 0;
}

int edgeAwareTikhonovFilterPrecalcConjugateGradientsDirectV2(tikhVariables tV, float* x_k, float *r_k , 
float* p_k, float *Ap_k , float *buffer , float *buffer2, float *temp1, float *temp2, int N){

    //A = (I-TyTx)
    float rkt_rk=0;
    float p_ktApk=0;

    //compute A p_k----------------------------------
    mXFilter(p_k, buffer,N,tV);
    for(int i=0;i<N*N;i++)
        Ap_k[i] = tV.H[i]*p_k[i] - buffer[i]; //Apk = H-Mx 
    for(int i =0; i< N*N;i++)
        buffer2[i] = p_k[i];
    mYFilter(buffer2, buffer, N, tV);
    for(int i=0;i<N*N;i++)
        Ap_k[i] = Ap_k[i] - buffer[i]; //Apk = H-Mx - My
    //------------------------------------------------
    for(int i=0; i<N*N; i++)
        p_ktApk += p_k[i]*Ap_k[i];

   
    for(int i=0;i<N*N;i++)
        rkt_rk += r_k[i]*r_k[i];

    float alpha_k = rkt_rk/p_ktApk;

    //compute x_k
    for(int i=0; i<N*N;i++)
        x_k[i] = x_k[i] + alpha_k*p_k[i];

    for(int i=0; i<N*N;i++){
        r_k[i] = r_k[i] - alpha_k*Ap_k[i];
    }

    float beta_k = 0;
    for(int i=0;i<N*N;i++)
        beta_k += r_k[i]*r_k[i];

    int diverge_flag = 0;

    beta_k = beta_k/rkt_rk;
    
    for(int i=0;i<N*N;i++)
        p_k[i] = r_k[i] + beta_k*p_k[i];

    

    return diverge_flag;
}


float** edgeAwareTikhonovFilterConjugateGradientRecursiveHybridV2(float* in, int N, int num_it){
    INIT_CLOCK();

    float *exectime = (float*)malloc(sizeof(float));
    float **outputs = (float**)malloc(2*sizeof(float*));

    float mu;
    float lambdax;
    float lambday;
    int S;
    FILE  *fparams = fopen("input_params.txt","r");
    S= fscanf(fparams, "%f,%f,%f,%i", &lambdax, &lambday, &mu , &S); //S=N, unused
    fclose(fparams);
    


    float* temp1 = (float*)aligned_alloc(alignment,sizeof(float) * L*L);
    float* temp2 = (float*)aligned_alloc(alignment,sizeof(float) * L*L);

    float *buffer0 = (float*)aligned_alloc(alignment,sizeof(float) * N*N); //accumulate fine
    float *buffer1 = (float*)aligned_alloc(alignment,sizeof(float) * N*N); 
    float *buffer2 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *buffer3 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *buffer4 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *buffer5 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);


    float *Wx = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *Wy = (float*)aligned_alloc(alignment,sizeof(float) * N*N);

    computeWx_Wy(in, N, temp1, temp2, Wx, Wy, mu, lambdax, lambday);

    START_CLOCK();
    
    tikhVariables tV = tihkonovVariables(in, N, mu, lambdax, lambday, temp1, temp2, Wx, Wy);
    getiTygH(in, N, tV, buffer1,buffer2,temp1,temp2);

    float kappa = 1 + 4*lambdax + 4*lambday;
    float dir_conv = (sqrtf(kappa) -1)/(sqrtf(kappa) +1);
    tV.rate = dir_conv;

    //STOP_CLOCK_AND_REPORT_TIME("Setup time");
   
    //^Compute initial iterate
    {
        #pragma omp simd aligned(in,buffer0:alignment)
        for(int i=0;i<N*N;i++)
            buffer0[i] = in[i];

        mXFilter(buffer0,buffer1,N,tV);

        #pragma omp simd aligned(buffer0,buffer1:alignment)
        for(int i=0; i<N*N;i++)
            buffer0[i] = buffer0[i] - buffer1[i]; //buffer 0 is Omega0
    }
    
    //Apply conjugate gradients
    computeResidue(buffer0, buffer2, buffer1, N, tV,temp1,temp2);

    for(int i=0;i<N*N;i++)
        buffer2[i] = buffer1[i];

    //SOLVE  (I-TyTx)Omega = (I+Ty)gH with conjugate gradients
    int it;
    int flag=0;
    for(it =0; (it < num_it)&&(flag <1); it++){
        flag += edgeAwareTikhonovFilterPrecalcConjugateGradientsRecursive(tV, buffer0, buffer1, buffer2, buffer3, buffer4, temp1, temp2,N);
    }
    //^Compute (I+Tx)
    {
        tikhonovXAxis(buffer0,buffer1,N,tV);

        //accumulate
        for(int i =0;i<N*N;i++)
            buffer0[i] += buffer1[i];
    }

    if(it < num_it){
        //^Iterate (I-Mx-My)f = gH
    
        //unnormalize probabilities
        buf_transpose(tV.pU, temp1, temp2, N);
        buf_transpose(tV.pD, temp1, temp2, N);
        for(int i=0;i<N*N;i++){
            tV.pB[i] *= tV.H[i];
            
            tV.pF[i] *= tV.H[i];
            
            tV.pU[i] *= tV.H[i];
            
            tV.pD[i] *= tV.H[i];
        }
        //compute new initial x0
        
        //computeResidue(buffer0, buffer2, buffer1, N, tV,temp1,temp2);
        mXFilter(buffer0, buffer1, N, tV);
        for(int i =0; i<N*N;i++)
            buffer2[i] = tV.H[i]*buffer0[i] - buffer1[i]; // (H-Mx)f
        for(int i =0; i<N*N;i++)
            buffer1[i] = buffer0[i];
        mYFilter(buffer1, buffer3, N, tV);
        for(int i=0;i<N*N;i++)
            buffer2[i] = buffer2[i] - buffer3[i]; //(H-Mx-My)f

        for(int i=0; i<N*N;i++)
            buffer1[i] = in[i] - buffer2[i]; // g - (H-Mx-My)f

        for(int i=0;i<N*N;i++)
            buffer2[i] = buffer1[i];

        //complete remaining iterations
        for(it=it; it<num_it;it++)
            flag = edgeAwareTikhonovFilterPrecalcConjugateGradientsDirectV2(tV, 
            buffer0, buffer1, buffer2, buffer3, buffer4, buffer5 ,temp1, temp2,N);
        
    }
  

    exectime[0] = STOP_CLOCK_AND_GET_TIME();
    outputs[0] = buffer0;
    outputs[1] = exectime;
    free(Wx);
    free(Wy);
    tV.freevars();
    return outputs;
}



float** edgeAwareTikhonovFilterConjugateGradientDirectEigen(float* in, int N, int num_it){
    INIT_CLOCK();

    
    float *exectime = (float*)malloc(sizeof(float));
    float **outputs = (float**)malloc(2*sizeof(float*));
    float mu;
    float lambdax;
    float lambday;
    int S;
    FILE  *fparams = fopen("input_params.txt","r");
    S= fscanf(fparams, "%f,%f,%f,%i", &lambdax, &lambday, &mu , &S); //S=N, unused
    fclose(fparams);

    float* temp1 = (float*)aligned_alloc(alignment,sizeof(float) * L*L);
    float* temp2 = (float*)aligned_alloc(alignment,sizeof(float) * L*L);

    float *Wx = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *Wy = (float*)aligned_alloc(alignment,sizeof(float) * N*N);

    computeWx_Wy(in, N, temp1, temp2, Wx, Wy, mu, lambdax, lambday);

    
    
    tikhVariables tV = tihkonovVariablesDirect(in, N, mu, lambdax, lambday, temp1, temp2, Wx, Wy);
    for(int x=0; x<N; x++)
    for(int y=0; y<N; y++){
        tV.pU[ind(x,y,N)]= -tV.pU[ind(x,y,N)]*tV.H[ind(x,y,N)];
        tV.pD[ind(x,y,N)]= -tV.pD[ind(x,y,N)]*tV.H[ind(x,y,N)];
        tV.pB[ind(x,y,N)]= -tV.pB[ind(x,y,N)]*tV.H[ind(x,y,N)];
        tV.pF[ind(x,y,N)]= -tV.pF[ind(x,y,N)]*tV.H[ind(x,y,N)];
    }

    START_CLOCK(); //Begin eigen
    Eigen::SparseMatrix<float> A(N*N,N*N);
    Eigen::VectorXf X(N*N), b(N*N), X0(N*N);
    float *buffer0 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);

    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower| Eigen::Upper, Eigen::DiagonalPreconditioner<float>> cg;
    //Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower| Eigen::Upper, Eigen::IncompleteLUT<float>> cg;
    //Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower| Eigen::Upper> cg;
    //Eigen::BiCGSTAB<Eigen::SparseMatrix<float>> cg;

    A.reserve(Eigen::VectorXf::Constant(N*N,5));
    //Fill matrix-------------------------------------------
    //diagonal
    int i,j;
    for(int x=0; x<N;x++){
        for(int y=0; y<N;y++){
            i = x*N + y;
                A.insert(i,i) = 1.0*tV.H[ind(x,y,N)];
        }
    }

    
    for(int x=0; x<N;x++){
        for(int y=0; y<N-1;y++){
            i = x*N + y;
            j = i + 1;
                A.insert(i,j) = tV.pU[ind(x,y,N)];
        }
    }

    for(int x=0; x<N;x++){
        for(int y=1; y<N;y++){
            i = x*N + y;
            j = i - 1;
                A.insert(i,j) = tV.pD[ind(x,y,N)];
        }
    }

    for(int x=1; x<N;x++){
        for(int y=0; y<N;y++){
            i = x*N + y;
            j = i - N;
                A.insert(i,j) = tV.pB[ind(x,y,N)];
        }
    }

    for(int x=0; x<N-1;x++){
        for(int y=0; y<N;y++){
            i = x*N + y;
            j = i + N;
                A.insert(i,j) = tV.pF[ind(x,y,N)];
        }
    }


    for(int x=0; x<N;x++){
        for(int y=0; y<N;y++){
            i = x*N + y;
            b(i) = in[ind(x,y,N)];
            X0(i) = in[ind(x,y,N)];
        }
    }
    

    //------------------------------------------------------
    cg.compute(A);
    cg.setMaxIterations(num_it);
    X = cg.solveWithGuess(b,X0);

    exectime[0] = STOP_CLOCK_AND_GET_TIME();
    for(int x=0; x<N;x++){
        for(int y=0; y<N;y++){
            i = (x)*N + y;
            buffer0[ind(x,y,N)] = X(i);
        }
    }
    outputs[0] = buffer0;
    outputs[1] = exectime;
    free(Wx);
    free(Wy);
    //tV.freevars();
    free(tV.H);
    free(tV.pB);
    free(tV.pF);
    free(tV.pU);
    free(tV.pD);

    return outputs;
}


void edgeAwareTikhonovFilterPrecalcBICGSTAB(tikhVariables tV, float *xk, float &rhok, float *rk, float *r0, float &ak, float &wk, float *vk, 
float *pk, float *h, float *s , float *t, float *pk1, float *buffer ,float *temp1, float *temp2, int N){


    float rhok1=0;
    for(int i=0;i<N*N;i++){
        rhok1 += r0[i]*rk[i]; 
    }

    float beta =(rhok1/rhok) * (ak/wk);
    
    for(int i=0; i <N*N;i++)
        pk[i] = rk[i] + beta*(pk[i] - wk*vk[i]);

    //vi = A pk1 where A = (I-TyTx)
    tikhonovXAxis(pk,buffer,N,tV);
    buf_transpose(buffer,temp1,temp2,N);
    tikhonovYAxis(buffer,vk,N,tV);
    buf_transpose(vk,temp1,temp2,N);
    for(int i =0; i< N*N ; i++)
        vk[i] = pk[i] - vk[i]; 

    ak = 0;
    for(int i =0; i<N*N;i++)
        ak = ak + r0[i]*vk[i];

    ak = rhok1/ak; // ak = rhok/<r0,vk>
   
    for(int i=0; i<N*N;i++)
        h[i] = xk[i] + ak*pk[i];
    
    for(int i=0; i<N*N; i++)
        s[i] = rk[i] - ak*vk[i];

    tikhonovXAxis(s,buffer,N,tV);
    buf_transpose(buffer,temp1,temp2,N);
    tikhonovYAxis(buffer,t,N,tV);
    buf_transpose(t,temp1,temp2,N);
    for(int i=0;i<N*N;i++)
        t[i] = s[i] - t[i]; //t = A*s, A=(I-TyTx)

    float num=0;
    float den=0;
    for(int i=0; i<N*N;i++){
        num += t[i]*s[i];
        den += t[i]*t[i];
    }
    float wk1 = num/den;

    for(int i=0; i<N*N;i++)
        xk[i] = h[i] + wk1*s[i];

    for(int i=0; i<N*N;i++)
        rk[i] = s[i] - wk1*t[i];

    rhok = rhok1;
    wk = wk1;
    //swap(vk,vk1); //same vector...
    return;
}



float** edgeAwareTikhonovFilterBICGSTAB(float* in, int N, int num_it){
    INIT_CLOCK();

    float *exectime = (float*)malloc(sizeof(float));
    float **outputs = (float**)malloc(2*sizeof(float*));

    float mu;
    float lambdax;
    float lambday;
    int S;
    FILE  *fparams = fopen("input_params.txt","r");
    S= fscanf(fparams, "%f,%f,%f,%i", &lambdax, &lambday, &mu , &S); //S=N, unused
    fclose(fparams);
    


    float* temp1 = (float*)aligned_alloc(alignment,sizeof(float) * L*L);
    float* temp2 = (float*)aligned_alloc(alignment,sizeof(float) * L*L);

    float *buffer0 = (float*)aligned_alloc(alignment,sizeof(float) * N*N); //accumulate fine
    float *buffer1 = (float*)aligned_alloc(alignment,sizeof(float) * N*N); 
    float *buffer2 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *buffer3 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *buffer4 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *buffer5 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *buffer6 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *buffer7 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *buffer8 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *buffer9 = (float*)aligned_alloc(alignment,sizeof(float) * N*N);


    float *Wx = (float*)aligned_alloc(alignment,sizeof(float) * N*N);
    float *Wy = (float*)aligned_alloc(alignment,sizeof(float) * N*N);

    computeWx_Wy(in, N, temp1, temp2, Wx, Wy, mu, lambdax, lambday);

    START_CLOCK();
    
    tikhVariables tV = tihkonovVariables(in, N, mu, lambdax, lambday, temp1, temp2, Wx, Wy);
    getiTygH(in, N, tV, buffer1,buffer2,temp1,temp2);

    //STOP_CLOCK_AND_REPORT_TIME("Setup time");
   
    //^Compute initial iterate
    {
        #pragma omp simd aligned(in,buffer0:alignment)
        for(int i=0;i<N*N;i++)
            buffer0[i] = in[i];

        mXFilter(buffer0,buffer1,N,tV);

        #pragma omp simd aligned(buffer0,buffer1:alignment)
        for(int i=0; i<N*N;i++)
            buffer0[i] = buffer0[i] - buffer1[i]; //buffer 0 is Omega0
    }

    //Apply conjugate gradients
    computeResidue(buffer0, buffer2, buffer1, N, tV,temp1,temp2);

    float *xk = &buffer0[0];
    float *r0 = &buffer1[0];
    float *vk = &buffer2[0];
    float *rk = &buffer3[0];
    float *pk = &buffer4[0];
    float *h = &buffer5[0];
    float *s = &buffer6[0];
    float *pk1 = &buffer7[0];
    float *t = &buffer8[0];
    float *buffer = &buffer9[0];

    float rhok = 1;
    float ak = 1;
    float wk = 1;
    
    for(int i=0; i<N*N;i++ ){
        vk[i] = 0;
        pk[i] = 0;
        rk[i] = r0[i];
    }

    //SOLVE  (I-TyTx)Omega = (I+Ty)gH with conjugate gradients
    for(int it =0; it<num_it; it++){
        edgeAwareTikhonovFilterPrecalcBICGSTAB(tV, xk, rhok, rk, r0, ak, wk,
         vk, pk, h, s, t, pk1, buffer, temp1, temp2, N);
    }
    //^Compute (I+Tx)
    {
        tikhonovXAxis(buffer0,buffer1,N,tV);

        //accumulate
        for(int i =0;i<N*N;i++)
            buffer0[i] += buffer1[i];
    }

    exectime[0] = STOP_CLOCK_AND_GET_TIME();
    outputs[0] = buffer0;
    outputs[1] = exectime;
    free(Wx);
    free(Wy);
    tV.freevars();
    return outputs;
}

int main(int argc, char *argv[]){
    INIT_CLOCK();
    int i,j,x,y;
    int N;
    int num_it;
    int aux=0;
    float **outs;
    int method = 0;
    int maxits = 0;

    if (argc >= 3){
        N = stoi(argv[1]);
        num_it = stoi(argv[2]);  
        method = stoi(argv[3]);
    }else{ 
        printf("type N and num_it\n");
        aux = scanf("%i %i", &N, &num_it);
    }

    /*Initialize Image */
    float *I = (float*)aligned_alloc(alignment,sizeof(float) * N*N);

    //read exact solution as double
    double *esolv = new double[N*N];
    double *img = new double[N*N];
    float *O;

    FILE *fptr = fopen("input.bin", "rb");
    aux = fread(img,sizeof(double),N*N,fptr);
    fclose(fptr);

    fptr = fopen("esolv.bin", "rb");
    aux = fread(esolv,sizeof(double),N*N,fptr);
    fclose(fptr);

    float *exectimes = new float[num_it];
    float *errs = new float[num_it];
    int it_stride = 5;
    int it;
    for(it=0; it<=num_it; it += max(1,(sqrtf(it+1)))){
        for(int i=0; i<N*N; i++)
            I[i] = img[i]; //reset image
        if(method == 0)
            outs = edgeAwareTikhonovFilterConjugateGradientDirectEigen(I, N, it);
        else if(method==1)
            outs =edgeAwareTikhonovFilterBICGSTAB(I, N, it);

        O = outs[0];
        exectimes[it] = outs[1][0];
        //compute errors
        double err=0;
        double norm=0;
        for(int i=0; i<N*N; i++){
            err += (O[i] - float(esolv[i]))*(O[i] - float(esolv[i]));
            norm += float(esolv[i])*float(esolv[i]);
        }
        err = sqrtf(err/norm);
        errs[it] = err;
        printf("Time = %f, REL error = %g\n",exectimes[it],err );
    }

    int write_out =0;

    FILE *resultf = fopen("logfile.txt","w");
    for(it=0;it<=num_it; it += max(1,int(sqrtf(it+1)))){
        fprintf(resultf, "%f,%f\n", exectimes[it],errs[it]);
    }
    fclose(resultf);
    if(write_out==1){
        FILE *arq = fopen("output.csv", "w");
        for(int i=0; i<N*N;i++){
            fprintf(arq,"%f\n",abs(O[i]));
        }
        fclose(arq);
    }

    return 0;
}