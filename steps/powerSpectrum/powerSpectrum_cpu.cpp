/*
 * File: powerSpectrum_cpu.cpp
 * Project: AccFFT
 * Created by Amir Gholami on 12/23/2014
 * Contact: contact@accfft.org
 * Copyright (c) 2014-2015
 *
 * A very simple example: compute Fourier power spectrum using a netcdf input
 * file and in output the spectrum dumped into a numpy  file.
 *
 * Example of use:
 * ./powerSpectrum_cpu --nx 64 --ny 64 --nz 64 --input data.nc --output spectrum.npy
 *
 * \author Pierre Kestener
 * \date June 26, 2015
 */

#include <stdlib.h>
#include <math.h> // for M_PI
#include <mpi.h>

#include "GetPot.h" // for command line arguments

#include <accfft.h>

#include <string>
#include <cnpy.h>

#ifdef USE_PNETCDF
#include "pnetcdf/pnetcdf_io.h"
#endif // USE_PNETCDF

#define SQR(x) ((x)*(x))

// =======================================================
// =======================================================
/*
 * Fourier power spectrum.
 * TODO.
 */
void compute_fourier_power_spectrum(Complex *data_hat, 
				    int N[3],
				    int isize[3], 
				    int istart[3],
				    int methodNb) {

  double NX = N[0];
  double NY = N[1];
  double NZ = N[2];
  
  double Lx = 1.0;
  double Ly = 1.0;
  double Lz = 1.0;

  double dx = Lx/NX;
  double dy = Ly/NY;
  double dz = Lz/NZ;

  for (int i=0; i < isize[0]; i++) {
    for (int j=0; j < isize[1]; j++) {
      for (int k=0; k < isize[2]; k++) {
	
	double kx = istart[0]+i;
	double ky = istart[1]+j;
	double kz = istart[2]+k;

	double kkx = (double) kx;
	double kky = (double) ky;
	double kkz = (double) kz;

	if (kx>NX/2)
	  kkx -= NX;
	if (ky>NY/2)
	  kky -= NY;
	if (kz>NZ/2)
	  kkz -= NZ;

	int index = i*isize[1]*isize[2]+j*isize[2]+k;

      } // end for k
    } // end for j
  } // end for i


} // compute_fourier_power_spectrum


// =======================================================
// =======================================================
/*
 * Perform FFT before computing power-spectrum.
 *
 * \param[in] n global domain sizes
 * \param[in] testCaseNb testcase number for initialization
 * \param[in] nThreads number of threads
 * \param[in] params parameters parsed from the command line arguments
 *
 * TODO
 */
void compute_fft(int *n, TESTCASE testCaseNb, int nthreads, GetPot &params) {

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

  printf("[mpi rank %d] isize  %3d %3d %3d osize  %3d %3d %3d\n", procid,
	 isize[0],isize[1],isize[2],
	 osize[0],osize[1],osize[2]
	 );

  printf("[mpi rank %d] istart %3d %3d %3d ostart %3d %3d %3d\n", procid,
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

  // optional : save input data
#ifdef USE_PNETCDF
  {
    std::string filename = "data_in.nc";
    MPI_Offset istart_mpi[3] = { istart[0], istart[1], istart[2] }; 
    MPI_Offset isize_mpi[3]  = { isize[0],  isize[1],  isize[2] }; 
    write_pnetcdf(filename,
		  istart_mpi,
		  isize_mpi,
		  n,
		  data);
  }
#else
  {
    std::ostringstream mpiRankString;
    mpiRankString.width(5);
    mpiRankString.fill('0');
    mpiRankString << procid ;
    std::string filename = "data_in_" + mpiRankString.str() + ".npz";
    save_cnpy(data, n, isize, istart, filename.c_str());
  }
#endif // USE_PNETCDF

  /* 
   * Perform forward FFT 
   */
  f_time-=MPI_Wtime();
  accfft_execute_r2c(plan,data,data_hat);
  f_time+=MPI_Wtime();

  MPI_Barrier(c_comm);

  /* 
   * here perform fourier filter associated to poisson ...
   */
  //poisson_fourier_filter(data_hat, n, plan->osize_2, plan->ostart_2, methodNb);



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

  return;

} // end compute_fft

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
  //poisson_solve(N, TESTCASE(testCaseNb), nthreads, cl);

  MPI_Finalize();
  return 0;

} // end main
