/*
 * \file: fBm_cpu.cpp
 *
 * Example application:
 * generate a 3D fraction Brownian process
 * http://en.wikipedia.org/wiki/Fractional_Brownian_motion
 *
 * Data are dump using collective IO (all MPI process write in a single file)
 * based on PNetCDF parallel IO library.
 *
 * Here we are using the most simple spectral method : 
 * just set a power-law Fourier spectrum and then apply 
 * inverse FFT (using the distributed implementation provided by accfft).
 *
 * Use C++11 for random number generator.
 *
 * \author Pierre Kestener
 * \date  June 16, 2015
 */

#include <stdlib.h>
#include <math.h> // M_PI, pow
#include <mpi.h>

#include <random> // c++11

#include "GetPot.h" // for command line arguments

#include <accfft.h>

#include <string>

#ifdef USE_PNETCDF
#include "pnetcdf/pnetcdf_io.h"
#endif // USE_PNETCDF

#define SQR(x) ((x)*(x))

// =======================================================
// =======================================================
/*
 * fBm fourier spectrum.
 *
 * \param[in,out] data_hat the fourier coefficient array
 * \param[in]     N        global domain sizes
 * \param[in]     isize    local domain sizes in Fourier (taking into account FFTW)
 * \param[in]     istart   local domain offsets
 * \param[in]     h        Hurst exponent
 *
 */
void fBm_fourier_spectrum(Complex *data_hat, 
			  int N[3],
			  int isize[3], 
			  int istart[3],
			  double h) {
  
  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // init random number generator
  unsigned int r_seed = (12+procid);
  std::default_random_engine generator(r_seed);

  // random distribution
  std::normal_distribution<double>       normal_dist(0.0,1.0); // amplitude
  std::uniform_real_distribution<double> unif_dist(0.0,1.0);   // phase

  double NX = N[0];
  double NY = N[1];
  double NZ = N[2];
  
  double Lx = 1.0;
  double Ly = 1.0;
  double Lz = 1.0;

  for (int i=0; i < isize[0]; i++) {
    for (int j=0; j < isize[1]; j++) {
      for (int k=0; k < isize[2]; k++) { 
	
	// compute global Fourier wave number
	double kx = istart[0]+i;
	double ky = istart[1]+j;
	double kz = istart[2]+k;

	// centerred wave number
	double kkx = (double) kx;
	double kky = (double) ky;
	double kkz = (double) kz;

	if (kx>NX/2)
	  kkx -= NX;
	if (ky>NY/2)
	  kky -= NY;
	if (kz>NZ/2)
	  kkz -= NZ;

	// local array index
	int index = i*isize[1]*isize[2]+j*isize[2]+k;

	// set fourier spectrum of a fBm
	double kSquare = kkx*kkx + kky*kky + kkz*kkz;
	double radius, phase;

	if (kSquare > 0) {
	  radius = pow(kSquare, -(2*h+3)/4) * normal_dist(generator);
	  phase = 2 * M_PI * unif_dist(generator);
	} else {  // enforce mean value is zero
	  radius = 1.0;
	  phase  = 0.0;
	}

	// make sure that Fourier coef at mid frequency plane is real
	if (kz == N[2]/2+1) {
	  radius = 1.0;
	  phase  = 0.0;
	}	  

	data_hat[index][0] = radius * cos(phase);
	data_hat[index][1] = radius * sin(phase);

      } // end for k
    } // end for j
  } // end for i

} // fBm_fourier_spectrum

/*****************************************************/
/* generate_fBm                                      */
/*****************************************************/
void generate_fBm(int *n, int nthreads, GetPot &params) 
{

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

  // printf("[mpi rank %d] isize  %3d %3d %3d osize  %3d %3d %3d\n", procid,
  // 	 isize[0],isize[1],isize[2],
  // 	 osize[0],osize[1],osize[2]
  // 	 );

  // printf("[mpi rank %d] istart %3d %3d %3d ostart %3d %3d %3d\n", procid,
  // 	 istart[0],istart[1],istart[2],
  // 	 ostart[0],ostart[1],ostart[2]
  // 	 );

  data=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));
  data_hat=(Complex*)accfft_alloc(alloc_max);

  accfft_init(nthreads);
  setup_time=-MPI_Wtime();
  /* Create FFT plan */
  accfft_plan * plan=accfft_plan_dft_3d_r2c(n,
					    data,(double*)data_hat,
					    c_comm,ACCFFT_MEASURE);
  setup_time+=MPI_Wtime();

  /* 
   * compute Fourier spectrum of a fBm
   */
  double h = params.follow(0.5,    "--h");
  fBm_fourier_spectrum(data_hat, n, osize, ostart, h);


  /* Perform backward FFT */
  i_time-=MPI_Wtime();
  accfft_execute_c2r(plan,data_hat,data);
  i_time+=MPI_Wtime();

  /* optional : save results */
#ifdef USE_PNETCDF
  {
    std::string filename = "fbm_out.nc";
    MPI_Offset istart_mpi[3] = { istart[0], istart[1], istart[2] }; 
    MPI_Offset isize_mpi[3]  = { isize[0],  isize[1],  isize[2] }; 
    write_pnetcdf(filename,
		  istart_mpi,
		  isize_mpi,
		  n,
		  data);
  }
#endif // USE_PNETCDF

  /* Compute some timings statistics */
  double g_f_time, g_i_time, g_setup_time;
  MPI_Reduce(&i_time,&g_i_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&setup_time,&g_setup_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);

  PCOUT<<"Timing for FFT of size "<<n[0]<<"*"<<n[1]<<"*"<<n[2]<<std::endl;
  PCOUT<<"Setup \t"<<g_setup_time<<std::endl;
  PCOUT<<"IFFT \t"<<g_i_time<<std::endl;

  accfft_free(data);
  accfft_free(data_hat);
  accfft_destroy_plan(plan);
  accfft_cleanup();
  MPI_Comm_free(&c_comm);
  return ;

} // end generate_fBm



/*****************************************************/
/*****************************************************/
/* main                                              */
/*****************************************************/
/*****************************************************/
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


  int nthreads=1;
  generate_fBm(N, nthreads, cl);

  MPI_Finalize();
  return 0;

} // end main



