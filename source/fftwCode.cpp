#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <omp.h>
#include <complex>
#include <cstring>
#include <fftw3.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

//#define REPORT_TIME

#ifdef REPORT_TIME
#define CLOCK_VAR() __clock_start
#define INIT_CLOCK() auto __clock_start = high_resolution_clock::now();
#define START_CLOCK() __clock_start = high_resolution_clock::now()
#define STOP_CLOCK_AND_REPORT_TIME(DESCRIPTION)                         \
    printf("%40s :: % 6.2f ms\n",                                       \
           DESCRIPTION,                                                 \
           duration_cast<microseconds>(high_resolution_clock::now() - __clock_start).count() / 1000.0f)
#define TIMER_PRINTF(...) printf(__VA_ARGS__)
#else
#define CLOCK_VAR()
#define INIT_CLOCK() (void)0
#define START_CLOCK() (void)0
#define STOP_CLOCK_AND_REPORT_TIME(DESCRIPTION) (void)0
#define TIMER_PRINTF(...) (void)0
#endif


//function to obtain index of flattened matrix
static inline int ind(int i, int j,int m){
    return i*m+j;
}

int main(int argc, char **argv){
    INIT_CLOCK();
    //Initialize---------------------------------------------//
    int N = argc < 2 ? 4096 : atoi(argv[1]);
    int fftw_precomp = argc < 3 ? 1 : atoi(argv[2]);
    float *I = new float[N*N];
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            I[ind(i,j,N)] = rand();
        }
    }

    float lx = 200;
    float ly = 100;

    auto start1 = high_resolution_clock::now();

    auto start2 = high_resolution_clock::now();

    auto stop = high_resolution_clock::now();

    //FFTW---------------------------------------------------//
    
    //In-place or out of place transform??
    fftwf_complex *O = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*(N/2+1)) ;
    
    // unsigned flags = FFTW_ESTIMATE; // choose FFTW_ESTIMATE/FFTW_MEASURE/FFTW_PATIENT 
    unsigned flags;
    switch (fftw_precomp)
    {
    case 0:
    case 1:
        //printf("FFTW_ESTIMATE\n");
        flags = FFTW_ESTIMATE;
        break;
    case 2:
        //printf("FFTW_MEASURE\n");
        flags = FFTW_MEASURE;
        break;
    case 3:
        //printf("FFTW_PATIENT\n");
        flags = FFTW_PATIENT;
        break;
    }

    
    START_CLOCK();

    start1 = high_resolution_clock::now();

    //Set plans
    fftwf_plan planf = fftwf_plan_dft_r2c_2d(N, N, I, O, flags);
    fftwf_plan planb = fftwf_plan_dft_c2r_2d(N, N ,O, I, flags);
    STOP_CLOCK_AND_REPORT_TIME("FFTW Plans");
    START_CLOCK();
    
    //Execute forward plan
    start2 = high_resolution_clock::now();

    fftwf_execute(planf);
    STOP_CLOCK_AND_REPORT_TIME("FFTW Forward");
    START_CLOCK();
    

    //perform computations-----------------------------------//
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N/2+1; j++){
                O[ind(i,j,N/2 + 1)][0] *=1.0f/(1.f+2.f*lx*(1.f-cos(2.f*M_PI*i/float(N))) + 2.f*ly*(1.f-cos(2.f*M_PI*j/float(N))))/float(N*N);
            }
    }

    STOP_CLOCK_AND_REPORT_TIME("Filter");
    START_CLOCK();

    //Execute backward plan----------------------------------//
    fftwf_execute(planb);
    STOP_CLOCK_AND_REPORT_TIME("FFTW Inverse");

    stop = high_resolution_clock::now();
    //cout << "Duration: " << float(duration.count())/1000000 << " seconds" << endl;
    auto duration1 = duration_cast<microseconds>(stop - start1);
    auto duration2 = duration_cast<microseconds>(stop - start2);

    printf("%f %f",float(duration1.count())/1000000,float(duration2.count())/1000000);
    //Write Output-------------------------------------------//
    
    FILE *f;
    /*f = fopen("kernel.csv", "w");
    int y;
    for(int x=0;x<N;x++){
        for(y=0;y<N-1;y++)
            fprintf(f,"%f,",I[ind(x,y,N)]);
        fprintf(f,"%f\n",I[ind(x,y,N)]);
    }
    */
    
    //Finalize-----------------------------------------------//
    fftwf_destroy_plan(planf);
    fftwf_destroy_plan(planb);
    free(I); 
    fftwf_free(O);

    return 0;
}
