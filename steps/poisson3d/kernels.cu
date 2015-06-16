/*
 * File: poisson3d_gpu.cpp
 * Project: AccFFT
 * Created by Amir Gholami on 12/23/2014
 * Contact: contact@accfft.org
 * Copyright (c) 2014-2015
 */
/*
 * Poisson solver CUDA kernel.
 * 
 * \author Pierre Kestener
 * \date June 15, 2015
 */

typedef double Complex[2];


// =======================================================
// =======================================================
/*
 * Poisson fourier filter (CUDA kernel).
 * Divide fourier coefficients by -(kx^2+ky^2+kz^2).
 */
__global__ 
void poisson_fourier_filter_kernel(Complex *data_hat, 
				   int N[3],      // global sizes
				   int isize[3],  // local  sizes
				   int istart[3],
				   int methodNb) 
{
  double NX = N[0];
  double NY = N[1];
  double NZ = N[2];
  
  double Lx = 1.0;
  double Ly = 1.0;
  double Lz = 1.0;

  double dx = Lx/NX;
  double dy = Ly/NY;
  double dz = Lz/NZ;

  // take care (direction reverse order for cuda)
  // cuda X dir maps k
  // cuda Y dir maps j
  unsigned int k = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
  //unsigned int i = blockDim.z * blockIdx.z + threadIdx.z;

  if (k < isize[2] and j < isize[1]) {

    double ky = istart[1]+j;
    double kz = istart[2]+k;

    double kky = (double) ky;
    double kkz = (double) kz;

    if (ky>NY/2)
      kky -= NY;
    if (kz>NZ/2)
      kkz -= NZ;
    
    for (int i=0, index=j*isize[2]+k; 
	 i < isize[0]; 
	 i++, index += isize[1]*isize[2]) {
      
      double kx = istart[0]+i;  
      double kkx = (double) kx;
      
      if (kx>NX/2)
	kkx -= NX;
      
      double scaleFactor = 0.0;
      
      if (methodNb==0) {
	
	/*
	 * method 0 (from Numerical recipes)
	 */
	
	scaleFactor=2*( 
		       (cos(1.0*2*M_PI*kx/NX) - 1)/(dx*dx) + 
		       (cos(1.0*2*M_PI*ky/NY) - 1)/(dy*dy) + 
		       (cos(1.0*2*M_PI*kz/NZ) - 1)/(dz*dz) )*(NX*NY*NZ);
	
	
      } else if (methodNb==1) {
	
	/*
	 * method 1 (just from Continuous Fourier transform of 
	 * Poisson equation)
	 */
	scaleFactor=-4*M_PI*M_PI*(kkx*kkx + kky*kky + kkz*kkz)*NX*NY*NZ;
	
      }
      
      
      if (kx!=0 or ky!=0 or kz!=0) {
	data_hat[index][0] /= scaleFactor;
	data_hat[index][1] /= scaleFactor;
      } else { // enforce mean value is zero
	data_hat[index][0] = 0.0;
	data_hat[index][1] = 0.0;
      }
      
    } // end for i

  } // end if
  
} // poisson_fourier_filter_kernel

// =======================================================
// =======================================================
/*
 * Poisson fourier filter.
 * Divide fourier coefficients by -(kx^2+ky^2+kz^2).
 */
#define POISSON_FILTER_DIMX 16
#define POISSON_FILTER_DIMY 16
void poisson_fourier_filter_gpu(Complex *data_hat, 
				int N[3],
				int isize[3],
				int istart[3],
				int methodNb) 
{

  // take care of direction order reversed :
  // CUDA X dir maps isize[2]
  // CUDA Y dir maps isize[1]
  // isize[0] is sweeped inside kernel
  int blocksInX = (isize[2]+POISSON_FILTER_DIMX-1)/POISSON_FILTER_DIMX;
  int blocksInY = (isize[1]+POISSON_FILTER_DIMY-1)/POISSON_FILTER_DIMY;

  dim3 DimGrid(blocksInX, blocksInY, 1);
  dim3 DimBlock(POISSON_FILTER_DIMX, POISSON_FILTER_DIMY, 1);
  poisson_fourier_filter_kernel<<<DimGrid, DimBlock>>>(data_hat,
						       N,
						       isize,
						       istart,
						       methodNb);

} // poisson_fourier_filter_gpu
