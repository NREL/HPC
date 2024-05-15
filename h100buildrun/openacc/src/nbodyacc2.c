/*
 * Copyright (c) 2017, NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void
check(int* res, int* exp, int n)
{
    int i;
    int tests_passed = 0;
    int tests_failed = 0;

    for (i = 0; i < n; i++) {
        if (exp[i] == res[i]) {
	    tests_passed ++;
        } else {
            tests_failed ++;
	    if( tests_failed < 50 )
            printf(
	    "test number %d FAILED. res %d(%08x)  exp %d(%08x)\n",
	     i+1,res[i], res[i], exp[i], exp[i] );
        }
    }
    if (tests_failed == 0) {
	printf("%3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    } else {
	printf("%3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
    }
}

void
check_(int* res, int* exp, int* np)
{
    check(res, exp, *np);
}

void
checkf(float* res, float* exp, int n)
{
    int i;
    int tests_passed = 0;
    int tests_failed = 0;

    for (i = 0; i < n; i++) {
        if (exp[i] == res[i]) {
	    tests_passed ++;
        } else {
            tests_failed ++;
	    if( tests_failed < 50 )
            printf(
	    "test number %d FAILED. res %g  exp %g\n",
	     i+1,res[i], exp[i]);
        }
    }
    if (tests_failed == 0) {
	printf("%3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    } else {
	printf("%3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
    }
}

void
checkf_(float* res, float* exp, int* np)
{
    checkf(res, exp, *np);
}

void
checkftol(float* res, float* exp, int n)
{
    int i;
    int tests_passed = 0;
    int tests_failed = 0;
    float tol = 0.000001;

    for (i = 0; i < n; i++) {
        if (exp[i] == res[i]) {
	    tests_passed ++;
	}else if( exp[i] != 0.0 && fabsf((exp[i]-res[i])/exp[i]) <= tol ){
	    tests_passed ++;
	}else if( exp[i] == 0.0 && res[i] <= tol ){
	    tests_passed ++;
        } else {
            tests_failed ++;
	    if( tests_failed < 50 )
            printf(
	    "test number %d FAILED. res %f  exp %f\n",
	     i+1,res[i], exp[i]);
        }
    }
    if (tests_failed == 0) {
	printf("%3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    } else {
	printf("%3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
    }
}

void
checkftol_(float* res, float* exp, int* np)
{
    checkftol(res, exp, *np);
}

void
checkftol5(float* res, float* exp, int n)
{
    int i;
    int tests_passed = 0;
    int tests_failed = 0;
    float tol = 0.00002;

    for (i = 0; i < n; i++) {
        if (exp[i] == res[i]) {
	    tests_passed ++;
	}else if( exp[i] != 0.0 && fabsf((exp[i]-res[i])/exp[i]) <= tol ){
	    tests_passed ++;
	}else if( exp[i] == 0.0 && res[i] <= tol ){
	    tests_passed ++;
        } else {
            tests_failed ++;
	    if( tests_failed < 50 )
            printf(
	    "test number %d FAILED. res %f  exp %f\n",
	     i+1,res[i], exp[i]);
        }
    }
    if (tests_failed == 0) {
	printf("%3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    } else {
	printf("%3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
    }
}


void
checkll(long long *res, long long *exp, int n)
{
    int i;
    int tests_passed = 0;
    int tests_failed = 0;

    for (i = 0; i < n; i++) {
        if (exp[i] == res[i]) {
	    tests_passed ++;
        } else {
             tests_failed ++;
	    if( tests_failed < 50 )
             printf( "test number %d FAILED. res %lld(%0llx)  exp %lld(%0llx)\n",
	     i+1,res[i], res[i], exp[i], exp[i] );
        }
    }
    if (tests_failed == 0) {
	printf("%3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    } else {
	printf("%3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
    }
}

void
checkll_(long long *res, long long *exp, int *np)
{
    checkll(res, exp, *np);
}

void
checkd(double* res, double* exp, int n)
{
    int i;
    int tests_passed = 0;
    int tests_failed = 0;

    for (i = 0; i < n; i++) {
        if (exp[i] == res[i])
	    tests_passed ++;
        else {
            tests_failed ++;
	    if( tests_failed < 50 )
            printf("test number %d FAILED. res %lg  exp %lg\n",
                      i+1, res[i], exp[i] );
        }
    }
    if (tests_failed == 0) {
	printf("%3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    } else {
	printf("%3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
    }
}

void
checkd_(double* res, double* exp, int* np)
{
    checkd(res, exp, *np);
}

void
checkdtol(double* res, double* exp, int n)
{
    int i;
    int tests_passed = 0;
    int tests_failed = 0;
    double tol = 0.00000000002;

    for (i = 0; i < n; i++) {
        if (exp[i] == res[i]){
	    tests_passed ++;
	}else if( exp[i] != 0.0 && ((exp[i]-res[i])/exp[i]) <= tol ){
	    tests_passed ++;
	}else if( exp[i] == 0.0 && res[i] <= tol ){
	    tests_passed ++;
        }else{
	    tests_failed ++;
	    if( tests_failed < 50 )
            printf(
	    "test number %d FAILED. res %lg  exp %lg\n",
	     i+1,res[i], exp[i]);
        }
    }
    if (tests_failed == 0) {
	printf("%3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    } else {
	printf("%3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
    }
}

void
checkdtol_(double* res, double* exp, int* np)
{
    checkdtol(res, exp, *np);
}

void
fcpyf_(float *r, float f)
{
    *r = f;
}

void
fcpyf(float *r, float f)
{
    fcpyf_(r, f);
}

void
fcpyi_(int *r, int f)
{
    *r = f;
}

void
fcpyi(int *r, int f)
{
    fcpyi_(r, f);
}

#if defined(WINNT) || defined(WIN32)
void
__stdcall CHECK(int* res, int* exp, int* np)
{
    check_(res, exp, np);
}

void
__stdcall CHECKD( double* res, double* exp, int* np)
{
    checkd_(res, exp, np);
}

void
__stdcall CHECKF( double* res, double* exp, int* np)
{
    checkf_(res, exp, np);
}

void
__stdcall CHECKFTOL( double* res, double* exp, int* np)
{
    checkftol_(res, exp, np);
}

void
__stdcall CHECKDTOL( double* res, double* exp, int* np)
{
    checkdtol_(res, exp, np);
}

void
__stdcall CHECKLL(long long *res, long long *exp, int *np)
{
    checkll_(res, exp, np);
}

void
__stdcall FCPYF(float *r, float f)
{
    fcpyf_(r, f);
}

void
__stdcall FCPYI(int *r, int f)
{
    fcpyi_(r, f);
}
#endif
#ifdef __cplusplus
}
#endif

/*
 * Copyright (c) 2017, NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#endif

#ifndef FP32
#ifndef FP64
#define FP64
#endif
#endif

#ifdef FP64
typedef double real;
int flopsPerInteraction = 30;
const real SOFTENING_SQUARED = 0.01;
#define RSQRT(x) 1.0 / sqrt((x))
#else
typedef float real;
int flopsPerInteraction = 20;
const real SOFTENING_SQUARED = 0.01f;
#define RSQRT(x) 1.0f / sqrtf((x))
#endif

typedef struct { real x, y, z; } real3;
typedef struct { real x, y, z, w; } real4;

real3 bodyBodyInteraction(real iPosx, real iPosy, real iPosz, 
                          real jPosx, real jPosy, real jPosz, real jMass)
{
    real rx, ry, rz;

    // r_01  [3 FLOPS]
    rx = jPosx - iPosx;
    ry = jPosy - iPosy;
    rz = jPosz - iPosz;

    // d^2 + e^2 [6 FLOPS]
    real distSqr = rx*rx+ry*ry+rz*rz;;
    distSqr += SOFTENING_SQUARED;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    real invDist = RSQRT(distSqr);
    real invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    real s = jMass * invDistCube;

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    real3 f;
    f.x = rx * s;
    f.y = ry * s;
    f.z = rz * s;

    return f;
}

void seqintegrate(real4 * restrict out,
               real4 * restrict in,
               real3 * restrict vel,
               real3 * restrict force,
               real dt, 
               int n)
{

        for (int i = 0; i < n; i++)
        {
            real fx, fy, fz;
            fx = fy = fz = 0;

            real iPosx = in[i].x;
            real iPosy = in[i].y;
            real iPosz = in[i].z;

            for (int j = 0; j < n; j++)
            {
                real3 ff = bodyBodyInteraction(iPosx, iPosy, iPosz,
                                               in[j].x, in[j].y, in[j].z, in[j].w);
                fx += ff.x;
                fy += ff.y;
                fz += ff.z;
            }

            force[i].x = fx;
            force[i].y = fy;
            force[i].z = fz;
        }

        for (int i = 0; i < n; i++)
        {
            real fx = force[i].x;
            real fy = force[i].y;
            real fz = force[i].z;

            real px = in[i].x;
            real py = in[i].y;
            real pz = in[i].z;
            real invMass = in[i].w;

            real vx = vel[i].x;
            real vy = vel[i].y;
            real vz = vel[i].z;

            // acceleration = force / mass; 
            // new velocity = old velocity + acceleration * deltaTime
            vx += (fx * invMass) * dt;
            vy += (fy * invMass) * dt;
            vz += (fz * invMass) * dt;

            // new position = old position + velocity * deltaTime
            px += vx * dt;
            py += vy * dt;
            pz += vz * dt;

            out[i].x = px;
            out[i].y = py;
            out[i].z = pz;
            out[i].w = invMass;

            vel[i].x = vx;
            vel[i].y = vy;
            vel[i].z = vz;
        }
}

void integrate(real4 * restrict out,
               real4 * restrict in,
               real3 * restrict vel,
               real3 * restrict force,
               real dt, 
               int n)
{

#pragma acc data pcopyin(in[0:n]) pcopyout(out[0:n]) pcopy(force[0:n], vel[0:n])
    {
#pragma acc parallel loop
        for (int i = 0; i < n; i++)
        {
            real fx, fy, fz;
            fx = fy = fz = 0;

            real iPosx = in[i].x;
            real iPosy = in[i].y;
            real iPosz = in[i].z;

            for (int j = 0; j < n; j++)
            {
                real3 ff = bodyBodyInteraction(iPosx, iPosy, iPosz,
                                               in[j].x, in[j].y, in[j].z, in[j].w);
                fx += ff.x;
                fy += ff.y;
                fz += ff.z;
            }

            force[i].x = fx;
            force[i].y = fy;
            force[i].z = fz;
        }

#pragma acc parallel loop
        for (int i = 0; i < n; i++)
        {
            real fx = force[i].x;
            real fy = force[i].y;
            real fz = force[i].z;

            real px = in[i].x;
            real py = in[i].y;
            real pz = in[i].z;
            real invMass = in[i].w;

            real vx = vel[i].x;
            real vy = vel[i].y;
            real vz = vel[i].z;

            // acceleration = force / mass; 
            // new velocity = old velocity + acceleration * deltaTime
            vx += (fx * invMass) * dt;
            vy += (fy * invMass) * dt;
            vz += (fz * invMass) * dt;

            // new position = old position + velocity * deltaTime
            px += vx * dt;
            py += vy * dt;
            pz += vz * dt;

            out[i].x = px;
            out[i].y = py;
            out[i].z = pz;
            out[i].w = invMass;

            vel[i].x = vx;
            vel[i].y = vy;
            vel[i].z = vz;
        }
    }
}

real 
dot(real v0[3], real v1[3])
{
    return v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2];
}

real
normalize(real vector[3])
{
    float dist = sqrt(dot(vector, vector));
    if (dist > 1e-6)
    {
        vector[0] /= dist;
        vector[1] /= dist;
        vector[2] /= dist;
    }
    return dist;
}

void cross(real out[3], real v0[3], real v1[3])
{
    out[0] = v0[1]*v1[2]-v0[2]*v1[1];
    out[1] = v0[2]*v1[0]-v0[0]*v1[2];
    out[2] = v0[0]*v1[1]-v0[1]*v1[0];
}

void randomizeBodies(real4* pos, 
                     real3* vel, 
                     float clusterScale, 
                     float velocityScale, 
                     int   n)
{
    srand(42);
    float scale = clusterScale;
    float vscale = scale * velocityScale;
    float inner = 2.5f * scale;
    float outer = 4.0f * scale;

    // int p = 0, v=0;
    int i = 0;
    while (i < n)
    {
        real x, y, z;
        x = rand() / (float) RAND_MAX * 2 - 1;
        y = rand() / (float) RAND_MAX * 2 - 1;
        z = rand() / (float) RAND_MAX * 2 - 1;

        real point[3] = {x, y, z};
        real len = normalize(point);
        if (len > 1)
            continue;

        pos[i].x =  point[0] * (inner + (outer - inner) * rand() / (real) RAND_MAX);
        pos[i].y =  point[1] * (inner + (outer - inner) * rand() / (real) RAND_MAX);
        pos[i].z =  point[2] * (inner + (outer - inner) * rand() / (real) RAND_MAX);
        pos[i].w = 1.0f;

        x = 0.0f; 
        y = 0.0f; 
        z = 1.0f; 
        real axis[3] = {x, y, z};
        normalize(axis);

        if (1 - dot(point, axis) < 1e-6)
        {
            axis[0] = point[1];
            axis[1] = point[0];
            normalize(axis);
        }
        //if (point.y < 0) axis = scalevec(axis, -1);
        real vv[3] = {(real)pos[i].x, (real)pos[i].y, (real)pos[i].z};
        cross(vv, vv, axis);
        vel[i].x = vv[0] * vscale;
        vel[i].y = vv[1] * vscale;
        vel[i].z = vv[2] * vscale;

        i++;
    }
}

#ifdef _WIN32
double PCFreq = 0.0;
__int64 timerStart = 0;
#else
struct timeval timerStart;
#endif

void StartTimer()
{
#ifdef _WIN32
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
        printf("QueryPerformanceFrequency failed!\n");

    PCFreq = (double)li.QuadPart/1000.0;

    QueryPerformanceCounter(&li);
    timerStart = li.QuadPart;
#else
    gettimeofday(&timerStart, NULL);
#endif
}

// time elapsed in ms
double GetTimer()
{
#ifdef _WIN32
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (double)(li.QuadPart-timerStart)/PCFreq;
#else
    struct timeval timerStop, timerElapsed;
    gettimeofday(&timerStop, NULL);
    timersub(&timerStop, &timerStart, &timerElapsed);
    return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
#endif
}

// run one iteration and compare to non-accelerated CPU version
void checkCorrectness(real4 *restrict pin,
                      real4 *restrict pout,
                      real3 *restrict v,
                      real dt,
                      int n)
{
    real4 *pin_ref  = (real4*)malloc(n * sizeof(real4));
    real4 *pout_ref = (real4*)malloc(n * sizeof(real4));
    real3 *v_ref    = (real3*)malloc(n * sizeof(real3));
    real3 *f        = (real3*)malloc(n * sizeof(real3));
    real3 *f_ref    = (real3*)malloc(n * sizeof(real3));

    randomizeBodies(pin_ref, v_ref,  1.54f, 8.0f, n);
    memcpy( pin, pin_ref, sizeof(real4)*n);
    memcpy( v, v_ref, sizeof(real3)*n);
    seqintegrate(pout_ref, pin_ref, v_ref, f_ref, dt, n);
    integrate(pout, pin, v, f, dt, n);

#ifdef FP64
    checkdtol( (real*)pout, (real*)pout_ref, 4*n );
#else
    checkftol5( pout, pout_ref, 4*n );
#endif

    free(pin_ref);
    free(pout_ref);
    free(v_ref);
}

double computePerfStats(float milliseconds, int iterations, int n)
{
    // double precision uses intrinsic operation followed by refinement,
    // resulting in higher operation count per interaction.
    // (Note Astrophysicists use 38 flops per interaction no matter what,
    // based on "historical precedent", but they are using FLOP/s as a 
    // measure of "science throughput". We are using it as a measure of 
    // hardware throughput.  They should really use interactions/s...
    // const int flopsPerInteraction = fp64 ? 30 : 20; 
    double interactionsPerSecond = (double)n * (double)n;
    interactionsPerSecond *= 1e-9 * iterations * 1000 / milliseconds;
    return interactionsPerSecond * (double)flopsPerInteraction;
}

int main(int argc, char** argv)
{
    int n = 1024;
    int iterations = 20;
    real dt = 0.01667;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) iterations = atoi(argv[2]);

    real4 *pin  = (real4*)malloc(n * sizeof(real4));
    real4 *pout = (real4*)malloc(n * sizeof(real4));
    real3 *v    = (real3*)malloc(n * sizeof(real3));
    real3 *f    = (real3*)malloc(n * sizeof(real3));

    randomizeBodies(pin, v,  1.54f, 8.0f, n);

    integrate(pout, pin, v, f, dt, n);
    checkCorrectness(pin, pout, v, dt, n);

    StartTimer();
    #pragma acc data pcopy(pin[0:n], pout[0:n], f[0:n], v[0:n])
    for (int i = 0; i < iterations; i++)
    {
        integrate(pout, pin, v, f, dt, n);
        real4 *t = pout;
        pout = pin; 
        pin = t;
    }
    double ms = GetTimer();

    StartTimer();
    for (int i = 0; i < iterations; i++)
    {
        seqintegrate(pout, pin, v, f, dt, n);
        real4 *t = pout;
        pout = pin; 
        pin = t;
    }
    double msUnaccelerated = GetTimer();

    double gf = computePerfStats(ms, iterations, n);
    double gfUnaccelerated = computePerfStats(msUnaccelerated, iterations, n);

    printf("n=%d bodies for %d iterations\n", n, iterations);
#ifdef _OPENACC
    printf("OpenACC:       %f ms: %f GFLOP/s\n", ms, gf);
#else
    printf("OpenMP:        %f ms: %f GFLOP/s\n", ms, gf);
#endif
    printf("Sequential:    %f ms: %f GFLOP/s\n", msUnaccelerated, gfUnaccelerated);

    free(pin);
    free(pout);
    free(v);

    return 0;
}
