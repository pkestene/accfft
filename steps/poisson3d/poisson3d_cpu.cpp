/*
 * File: poisson3d_cpu.cpp
 * Project: AccFFT
 * Created by Amir Gholami on 12/23/2014
 * Contact: contact@accfft.org
 * Copyright (c) 2014-2015
 *
 * A very simple test for solving Laplacian(\phi) = rho using FFT in 3D.
 *
 * Laplacian operator can be considered as a low-pass filter.
 * Here we implement 2 types of filters :
 *
 * method 0 : see Numerical recipes in C, section 19.4
 * method 1 : just divide right hand side by -(kx^2+ky^2+kz^2) in Fourier
 *
 * Test case 0:  rho(x,y,z) = sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*sin(2*pi*z/Lz)
 * Test case 1:  rho(x,y,z) = 4*alpha*(alpha*(x^2+y^2+z^2)-1)*exp(-alpha*(x^2+y^2+z^2))
 * Test case 2:  rho(x,y,z) = ( r=sqrt(x^2+y^2+z^2) < R ) ? 1 : 0 
 *
 * Example of use:
 * ./poisson3d_cpu --nx 64 --ny 64 --nz 64 --method 1 --testcase 2

 * \author Pierre Kestener
 * \date June 1st, 2015
 */

#include <stdlib.h>
#include <math.h> // for M_PI
#include <mpi.h>

#include "GetPot.h" // for command line arguments

#include <accfft.h>

#define SQR(x) ((x)*(x))

enum TESTCASE {
  TESTCASE_SINE=0,
  TESTCASE_GAUSSIAN=1,
  TESTCASE_UNIFORM_BALL=2
};


// =======================================================
// =======================================================
/*
 * testcase sine: eigenfunctions of Laplacian
 */
double testcase_sine(double x,double y, double z){

  return sin(2*M_PI*x) * sin(2*M_PI*y) * sin(2*M_PI*z);

} // testcase_sine

// =======================================================
// =======================================================
/*
 * testcase gaussian
 */
double testcase_gaussian(double x,double y, double z, double alpha){

  return 4*alpha*(alpha*(x*x+y*y+z*z)-1)*exp(-alpha*(x*x+y*y+z*z));

} // testcase_gaussian

// =======================================================
// =======================================================
/*
 * testcase uniform ball
 */
double testcase_uniform_ball(double x,  double y,  double z,
			     double xC, double yC, double zC,
			     double R) {

  double r = sqrt( (x-xC)*(x-xC) + (y-yC)*(y-yC) + (z-zC)*(z-zC) );

  double res = r < R ? 1.0 : 0.0;
  return res;

} // testcase_uniform_ball


// =======================================================
// =======================================================
template<const TESTCASE testcase_id>
void initialize(double *a, int *n, MPI_Comm c_comm, GetPot &params)
{
  double pi=M_PI;
  int n_tuples=n[2];
  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm);

  /*
   * testcase gaussian parameters
   */
  double alpha=1.0;
  if (testcase_id == TESTCASE_GAUSSIAN)
    alpha = params.follow(1.0,    "--alpha");

  /*
   * testcase uniform ball parameters
   */
  // uniform ball function center
  double xC = params.follow((double) 0.0, "--xC");
  double yC = params.follow((double) 0.0, "--yC");
  double zC = params.follow((double) 0.0, "--zC");
	  
  // uniform ball radius
  double R = params.follow(0.02, "--radius");


#pragma omp parallel
  {
    double X,Y,Z;
    long int ptr;
#pragma omp for
    for (int i=0; i<isize[0]; i++){
      for (int j=0; j<isize[1]; j++){
        for (int k=0; k<isize[2]; k++){
          X=2*pi/n[0]*(i+istart[0]);
          Y=2*pi/n[1]*(j+istart[1]);
          Z=2*pi/n[2]*k;
          ptr=i*isize[1]*n_tuples+j*n_tuples+k;

	  if (testcase_id == TESTCASE_SINE) {
	    a[ptr]=testcase_sine(X,Y,Z);
	  } else if (testcase_id == TESTCASE_GAUSSIAN) {
	    a[ptr]=testcase_gaussian(X,Y,Z,alpha);
	  } else if (testcase_id == TESTCASE_UNIFORM_BALL) {
	    a[ptr]=testcase_uniform_ball(X,Y,Z,xC,yC,zC,R);
	  }

        }
      }
    }
  }
  return;
} // end initialize

// =======================================================
// =======================================================
/*
 * FFT-based poisson solver.
 *
 * \param[in] n global domain sizes
 * \param[in] testCaseNb testcase number for initialization
 * \param[in] nThreads number of threads
 * \param[in] params parameters parsed from the command line arguments
 */
void poisson_solve(int *n, TESTCASE testCaseNb, int nthreads, GetPot &params) {

  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* Create Cartesian Communicator */
  int c_dims[2];
  MPI_Comm c_comm;
  accfft_create_comm(MPI_COMM_WORLD,c_dims,&c_comm);

  printf("[mpi rank %d] c_dims = %d %d\n", procid, c_dims[0], c_dims[1]);

  double *data;
  Complex *data_hat;
  double f_time=0*MPI_Wtime(),i_time=0, setup_time=0;
  int alloc_max=0;

  int isize[3],osize[3],istart[3],ostart[3];
  /* Get the local pencil size and the allocation size */
  alloc_max=accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm);

  printf("[mpi rank %d] isize %d %d %d osize %d %d %d\n", procid,
	 isize[0],isize[1],isize[2],
	 osize[0],osize[1],osize[2]
	 );

  printf("[mpi rank %d] istart %d %d %d ostart %d %d %d\n", procid,
	 istart[0],istart[1],istart[2],
	 ostart[0],ostart[1],ostart[2]
	 );

  data=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));
  data_hat=(Complex*)accfft_alloc(alloc_max);

  accfft_init(nthreads);
  setup_time=-MPI_Wtime();
  /* Create FFT plan */
  accfft_plan * plan = accfft_plan_dft_3d_r2c(n,
					      data, (double*)data_hat,
					      c_comm, ACCFFT_MEASURE);
  setup_time+=MPI_Wtime();

  /*  Initialize data */
  switch(testCaseNb) {
  case TESTCASE_SINE:
    initialize<TESTCASE_SINE>(data, n, c_comm, params); 
    break;
  case TESTCASE_GAUSSIAN:
    initialize<TESTCASE_GAUSSIAN>(data, n, c_comm, params); 
    break;
  case TESTCASE_UNIFORM_BALL:
    initialize<TESTCASE_UNIFORM_BALL>(data, n, c_comm, params); 
    break;
  }
  MPI_Barrier(c_comm);

  /* 
   * Perform forward FFT 
   */
  f_time-=MPI_Wtime();
  accfft_execute_r2c(plan,data,data_hat);
  f_time+=MPI_Wtime();

  MPI_Barrier(c_comm);

  /* 
   * here perform fourier filter associated to poisson ... TODO 
   */


  /* 
   * Perform backward FFT 
   */
  double * data2=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));
  i_time-=MPI_Wtime();
  accfft_execute_c2r(plan,data_hat,data2);
  i_time+=MPI_Wtime();

  /* Check Error */
  double err=0,g_err=0;
  double norm=0,g_norm=0;
  for (int i=0;i<isize[0]*isize[1]*isize[2];++i){
    err+=data2[i]/n[0]/n[1]/n[2]-data[i];
    norm+=data2[i]/n[0]/n[1]/n[2];
  }
  MPI_Reduce(&err,&g_err,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&norm,&g_norm,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);

  PCOUT<<"\n Error is "<<g_err<<std::endl;
  PCOUT<<"Relative Error is "<<g_err/g_norm<<std::endl;

  /* Compute some timings statistics */
  double g_f_time, g_i_time, g_setup_time;
  MPI_Reduce(&f_time,&g_f_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&i_time,&g_i_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&setup_time,&g_setup_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);

  PCOUT<<"Timing for FFT of size "<<n[0]<<"*"<<n[1]<<"*"<<n[2]<<std::endl;
  PCOUT<<"Setup \t"<<g_setup_time<<std::endl;
  PCOUT<<"FFT \t"<<g_f_time<<std::endl;
  PCOUT<<"IFFT \t"<<g_i_time<<std::endl;

  accfft_free(data);
  accfft_free(data_hat);
  accfft_free(data2);
  accfft_destroy_plan(plan);
  accfft_cleanup();
  MPI_Comm_free(&c_comm);
  return ;

} // end poisson_solve

// =======================================================
// =======================================================
// void check_err(double* a,int*n,MPI_Comm c_comm){
//   int nprocs, procid;
//   MPI_Comm_rank(c_comm, &procid);
//   MPI_Comm_size(c_comm,&nprocs);
//   long long int size=n[0];
//   size*=n[1]; size*=n[2];
//   double pi=4*atan(1.0);

//   int n_tuples=n[2];
//   int istart[3], isize[3], osize[3],ostart[3];
//   accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm);

//   double err,norm;

//   double X,Y,Z,numerical_r;
//   long int ptr;
//   int thid=omp_get_thread_num();
//   for (int i=0; i<isize[0]; i++){
//     for (int j=0; j<isize[1]; j++){
//       for (int k=0; k<isize[2]; k++){
//         X=2*pi/n[0]*(i+istart[0]);
//         Y=2*pi/n[1]*(j+istart[1]);
//         Z=2*pi/n[2]*k;
//         ptr=i*isize[1]*n_tuples+j*n_tuples+k;
//         numerical_r=a[ptr]/size; if(numerical_r!=numerical_r) numerical_r=0;
//         err+=std::abs(numerical_r-testcase(X,Y,Z));
//         norm+=std::abs(testcase(X,Y,Z));

//         //PCOUT<<"("<<i<<","<<j<<","<<k<<")  "<<numerical<<'\t'<<testcase(X,Y,Z)<<std::endl;
//       }
//     }
//   }

//   double gerr=0,gnorm=0;
//   MPI_Reduce(&err,&gerr,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
//   MPI_Reduce(&norm,&gnorm,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
//   PCOUT<<"The L1 error between iFF(a)-a is = "<<gerr<<std::endl;
//   PCOUT<<"The Rel. L1 error between iFF(a)-a is = "<<gerr/gnorm<<std::endl;
//   if(gerr/gnorm>1e-10){
//     PCOUT<<"\033[1;31m ERROR!!! FFT not computed correctly!\033[0m"<<std::endl;
//   }
//   else{
//     PCOUT<<"\033[1;36m FFT computed correctly!\033[0m"<<std::endl;
//   }

// } // end check_err

// =======================================================
// =======================================================
/*
 * Poisson fourier filter.
 * Divide fourier coefficients by -(kx^2+ky^2+kz^2).
 */
void poisson_fourier_filter(Complex *data_hat, 
			    int N[3],
			    int isize[3], int istart[3],
			    int testcase) {
  /* TODO */

} // fourier_filter

/******************************************************/
/******************************************************/
/******************************************************/
int main(int argc, char **argv)
{

  int NX,NY,NZ;
  MPI_Init (&argc, &argv);
  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* parse command line arguments */
  GetPot cl(argc, argv);

  NX = cl.follow(128,    "--nx");
  NY = cl.follow(128,    "--ny");
  NZ = cl.follow(128,    "--nz");

  int N[3]={NX,NY,NZ};

  // which method ? variant of FFT-based Poisson solver : 0 or 1
  const int methodNb   = cl.follow(0, "--method");

  // test case number
  const int testCaseNb = cl.follow(TESTCASE_SINE, "--testcase");
  if (testCaseNb < 0 || testCaseNb > 2) {
    if (procid == 0) {
      std::cerr << "Wrong test case. Must be integer < 2 !!!\n";
    }
  } else {
    if (procid == 0) {
      std::cout << "Using test case number : " << testCaseNb << std::endl;
    }
  }


  int nthreads=1;
  poisson_solve(N, TESTCASE(testCaseNb), nthreads, cl);

  MPI_Finalize();
  return 0;
} // end main
