/*
 * File: kernels.cu
 *
 * Fractionnal Brownian motion field generator CUDA kernel.
 * 
 * Random number generator is directly adapted from Nvidia curand documentation.
 * 
 * \author Pierre Kestener
 * \date June 17, 2015
 */

#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>

typedef double Complex[2];

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {	\
      printf("Error at %s:%d\n",__FILE__,__LINE__);	\
      return EXIT_FAILURE;}} while(0)


__global__ 
void setup_random_generator_kernel(curandState *state1, 
				   curandState *state2, 
				   dim3 N,
				   dim3 isize)
{
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  int id = j*isize.z+k;

  /* Each thread gets same seed, a different sequence 
     number, no offset */
  if (k < isize.z and j < isize.y) {
    curand_init(1234, id, 0, &state1[id]);
    curand_init(1234, id, 0, &state2[id]);
  }

} // setup_random_generator_kernel

// =======================================================
// =======================================================
/*
 * Poisson fourier filter (CUDA kernel).
 * Divide fourier coefficients by -(kx^2+ky^2+kz^2).
 */
__global__ 
void fBm_fourier_spectrum_kernel(Complex *data_hat, 
				 dim3 N,      // global sizes
				 dim3 isize,  // local  sizes
				 dim3 istart,
				 double h,
				 curandState *randomState1,
				 curandState *randomState2)
{
  double NX = (double) N.x;
  double NY = (double) N.y;
  double NZ = (double) N.z;

  // take care (direction reverse order for cuda)
  // cuda X dir maps k
  // cuda Y dir maps j
  unsigned int k = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
  //unsigned int i = blockDim.z * blockIdx.z + threadIdx.z;
 
  int id = j*isize.z+k;

  if (k < isize.z and j < isize.y) {
    
    /* Copy random state to local memory for efficiency */
    curandState localState1 = randomState1[id];
    curandState localState2 = randomState2[id];

    double ky = istart.y+j;
    double kz = istart.z+k;
    
    double kky = (double) ky;
    double kkz = (double) kz;
    
    if (ky>NY/2)
      kky -= NY;
    if (kz>NZ/2)
      kkz -= NZ;

    for (int i=0, index=j*isize.z+k;
	 i < isize.x;
	 i++, index += isize.y*isize.z) {
      
      // compute global Fourier wave number
      double kx = istart.x+i;
    
      // centerred wave number
      double kkx = (double) kx;
    
      if (kx>NX/2)
	kkx -= NX;
    
      // set fourier spectrum of a fBm
      double kSquare = kkx*kkx + kky*kky + kkz*kkz;
      double radius, phase;
      
      if (kSquare > 0) {
	radius = pow(kSquare, -(2*h+3)/4) * curand_normal(&localState1);
	phase  = 2 * M_PI * curand_uniform(&localState2);
      } else {  // enforce mean value is zero
	radius = 1.0;
	phase  = 0.0;
      }
    
      // make sure that Fourier coef at mid frequency plane is real
      if (kz == NZ/2+1) {
	radius = 1.0;
	phase  = 0.0;
      }
      
      data_hat[index][0] = radius * cos(phase);
      data_hat[index][1] = radius * sin(phase);
        
    } // end for i 

    // save randomState
    randomState1[id] = localState1;
    randomState2[id] = localState2;
    
  } // end if

} // fBm_fourier_spectrum_kernel

// =======================================================
// =======================================================
/*
 * Generate fBm fourier spectrum.
 *
 */
#define FBM_KERNEL_DIMX 16
#define FBM_KERNEL_DIMY 16
void fBm_fourier_spectrum_gpu(Complex *data_hat, 
			      int N[3],
			      int isize[3],
			      int istart[3],
			      double h) 
{

  dim3 NN     (N[0]     , N[1]     , N[2]);
  dim3 iisize (isize[0] , isize[1] , isize[2]);
  dim3 iistart(istart[0], istart[1], istart[2]);

  // take care of direction order reversed :
  // CUDA X dir maps isize[2]
  // CUDA Y dir maps isize[1]
  // isize[0] is sweeped inside kernel
  int blocksInX = (isize[2]+FBM_KERNEL_DIMX-1)/FBM_KERNEL_DIMX;
  int blocksInY = (isize[1]+FBM_KERNEL_DIMY-1)/FBM_KERNEL_DIMY;

  dim3 DimGrid(blocksInX, blocksInY, 1);
  dim3 DimBlock(FBM_KERNEL_DIMX, FBM_KERNEL_DIMY, 1);

  // random number generator states on device / one per thread
  curandState *devStates1, *devStates2;
  cudaMalloc((void **)&devStates1, isize[2] * isize[1] * sizeof(curandState));
  cudaMalloc((void **)&devStates2, isize[2] * isize[1] * sizeof(curandState));
  
  setup_random_generator_kernel<<<DimGrid, DimBlock>>>(devStates1,
						       devStates2,
						       NN,
						       iisize);

  // compute fBm
  fBm_fourier_spectrum_kernel<<<DimGrid, DimBlock>>>(data_hat,
  						     NN,
  						     iisize,
  						     iistart,
  						     h,
  						     devStates1,
  						     devStates2);

  cudaFree(devStates1);
  cudaFree(devStates2);
  
} // fBm_fourier_spectrum_gpu
