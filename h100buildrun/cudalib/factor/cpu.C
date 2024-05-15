#include <iostream>
// Based on https://dynamithead.wordpress.com/2012/06/30/introduction-to-how-to-call-lapack-from-a-cc-program-example-solving-a-system-of-linear-equations/

template <typename T> void print_matrix_l(const int &m, const int &n, const T *A, const int &lda); 
template <> void print_matrix_l(const int &m, const int &n, const double *A, const int &lda) {     
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%8.3f ", A[j * lda + i]);                                                
        }
        std::printf("\n");
    }
}
    
template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &lda);   
template <> void print_matrix(const int &m, const int &n, const float *A, const int &lda) {        
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);                                                 
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}   
    

#ifndef pmax
int pmax=10;
#endif

#ifdef MINE
    const int m = MINE;
#else
    const int m = 3;
#endif


#ifdef __INTEL_LLVM_COMPILER
#include "mkl_lapack.h"
#else
#include "lapacke.h"
#endif

#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sys/time.h>
#include <omp.h>

 
using namespace std;


double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

 
int main()
{
    // note, to understand this part take a look in the MAN pages, at section of parameters.
    char    TRANS = 'N';
    int     INFO=3;
    int     LDA = m;
    int     LDB = m;
    int     N = m;
    int     NRHS = 1;
    int     IPIV[m] ;
    double inv1,inv2;
 
    std::vector<double> A(m*m, 0.2); 
    int one;
    for (one=0 ; one < m*m; one=one+(m+1)) {
	    A[one]=1;
    }

    std::vector<double> B(m, 1.0); 
    
    
    printf("Matrix size %d\n",m);
    if (m <= pmax) {
    	printf("Printing %d elements\n",m);
    	print_matrix_l(m,m,& *A.begin(),m);
    }
#pragma omp parallel
    {
#pragma omp single
    printf("threads %d\n",omp_get_num_threads());
    }


 
// end of declarations
 
    cout << "compute the LU factorization..." << endl << endl;
    //void LAPACK_dgetrf( lapack_int* m, lapack_int* n, double* a, lapack_int* lda, lapack_int* ipiv, lapack_int *info );
    //LAPACK_dgetrf(&N,&N,A,&LDA,IPIV,&INFO);
    inv1=mysecond();
    dgetrf_(&N,&N,& *A.begin(),&LDA,IPIV,&INFO);
    
 
    // checks INFO, if INFO != 0 something goes wrong, for more information see the MAN page of dgetrf.
    if(INFO)
    {
        cout << "an error occured : "<< INFO << endl << endl;
    }else{
        cout << "solving the system..."<< endl << endl;
        // void LAPACK_dgetrs( char* trans, lapack_int* n, lapack_int* nrhs, const double* a, lapack_int* lda, const lapack_int* ipiv,double* b, lapack_int* ldb, lapack_int *info );
#ifdef EXTRA
        dgetrs_(&TRANS,&N,&NRHS,& *A.begin(),&LDA,IPIV,& *B.begin(),&LDB,&INFO,0);
#else
        dgetrs_(&TRANS,&N,&NRHS,& *A.begin(),&LDA,IPIV,& *B.begin(),&LDB,&INFO);
#endif
        inv2=mysecond();
        if(INFO)
        {
            // checks INFO, if INFO != 0 something goes wrong, for more information see the MAN page of dgetrs.
            cout << "an error occured : "<< INFO << endl << endl;
        }else{
            cout << "print the result : {";
            int i;
            for (i=0;i<min(N,pmax);i++)
            {
                cout << B[i] << " ";
            }
            cout << "}" << endl << endl;
        }
    }
    printf("TIMINGS (seconds)\n");
	    printf("  CPU total %g\n",inv2-inv1);
 
    return 0;
}
