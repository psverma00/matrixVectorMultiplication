#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include "cuda_utils.h"
#include "mkl.h"

void CreatMatVec(double *A, double *C, double *D, long int ROW, long int COLUMN)
{
  for(int i=0; i<ROW; i++)
  {
    C[i] = 1;
    //C[i] = drand48();
    D[i] = 0;    
    for(int j=0; j<COLUMN; j++)
    {	  
      A[i*COLUMN+j] = i*COLUMN+j;
      //A[i*COLUMN+j] = drand48();
    }
  }

}	

void matVecMulti(double *A, double *C, double *D, long int ROW, long int COLUMN)
{
  for(int i=0; i<ROW; i++)
  {
    for(int j=0; j<COLUMN; j++)
    {	  
      D[i] += A[i*COLUMN+j] * C[j];
    }
    //printf(" D[%d] : %lf \n", i, D[i]); 
  }

}	

__global__ void matVecMultiCudaKernel(double *A, double *C, double *D, int ROW, int COLUMN)
{
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  
  for(int j = 0; j < COLUMN; j += 1)
  {  
    for(int i = gtid; i < ROW; i = i + gridDim.x*blockDim.x)  // this loop is required when blocksize < COLUMN or ROW
    {
      D[i] += A[i *COLUMN  + j] * C[j]; 
    }
  }

}


int main()
{
  srand48(time(NULL));     	
  long int memMat, memVec;
  long int row, column;
  int diff; 
 
  row = 32768; //100000;  
  column = 32768; //100000;
  
  int blocksize = 256;
  int gridsize = 64; 

  memMat = row*column*sizeof(double);
  memVec = row*sizeof(double);

  double *a = (double*)malloc(memMat);
  double *c = (double*)malloc(memVec);
  double *d = (double*)malloc(memVec);
  double *dBlas = (double*)malloc(memVec);

  clock_t cpu_start0, cpu_end0;
  clock_t cpu_start1, cpu_end1;
  clock_t cpu_start2, cpu_end2;
  clock_t cpu_start3, cpu_end3;
  clock_t cpu_start4, cpu_end4;
  clock_t cpu_start5, cpu_end5;
  clock_t cpu_start6, cpu_end6;

  printf("\nCreating Matrix and Vector\n");
  cpu_start0 = clock();
  CreatMatVec(a, c, d, row, column);
  cpu_end0 = clock();

  printf("\nTime spent in creating vector and matrix   : %4.6f sec\n", (double)((double)(cpu_end0 - cpu_start0)/CLOCKS_PER_SEC));

  for(int i=0; i<column; i++)
  {
    d[i] = 0.0; 
    dBlas[i] = 0.0; 
  }

  printf("\n");
  printf("\nCalculating Matrix-Vector Multiplication using CPU function\n");
  cpu_start1 = clock();
  matVecMulti(a, c, d, row, column);
  cpu_end1 = clock();

  printf("\n");
  printf("\nTime spent in matrix vector multiplication : %4.6f sec\n", (double)((double)(cpu_end1 - cpu_start1)/CLOCKS_PER_SEC));

 // for(int i=0; i<column; i++)
 // {
 //  printf("d[%d] : %lf\n",i,d[i]);
 // }
 
  printf("\n");

  printf("\nCalculating Matrix-Vector Multiplication using BLAS\n");
  cpu_start2 = clock();
  cblas_dgemv(CblasRowMajor, CblasNoTrans, row, column, 1, a, column, c, 1, 1.0f, dBlas, 1);
  cpu_end2 = clock();

  printf("\nTime spent in matrix vector multiplication using blas : %4.6f sec\n", (double)((double)(cpu_end2 - cpu_start2)/CLOCKS_PER_SEC));
 
  //for(int i=0; i<column; i++)
  //{
  // printf("dBlas[%d] : %lf\n",i,dBlas[i]);
  //}


  for(int i=0; i<column; i++) 
  {
   diff +=  dBlas[i] - d[i];
  }
  if(diff == 0) 
    printf("\nResults from CPU function and BLAS funtion are same.\n");  
  else 
    printf("\nResults are different\n");  
  printf("\n");

  double *d_cpu = (double *)malloc(memVec);

  //Defining variabls for cuda kernel

  double *a_gpu;
  double *c_gpu;
  double *d_gpu; 

  //Allocating memory for the cuda kernel

  printf("\nAllocating memory for cuda kernel\n");
  cpu_start3 = clock();
  cudaMalloc(&a_gpu,memMat); 
  cudaMalloc(&c_gpu,memVec); 
  cudaMalloc(&d_gpu,memVec); 
  cpu_end3 = clock();
  printf("\nTime spent in memory allocation on GPU : %4.6f sec\n", (double)((double)(cpu_end3 - cpu_start3)/CLOCKS_PER_SEC));

  cpu_start4 = clock();
  cudaMemcpy(a_gpu,a,memMat,cudaMemcpyHostToDevice);
  cudaMemcpy(c_gpu,c,memVec,cudaMemcpyHostToDevice);
  cpu_end4 = clock();
  
  printf("\nTime spent in copying data from CPU to GPU : %4.6f sec\n", (double)((double)(cpu_end4 - cpu_start4)/CLOCKS_PER_SEC));

  printf("\nCalculating Matrix-Vector Multiplication using GPU Kernel\n");

  cpu_start5 = clock();
  matVecMultiCudaKernel<<<gridsize,blocksize>>>(a_gpu,c_gpu,d_gpu,row,column);
  cpu_end5 = clock();
  
  cudaDeviceSynchronize();

  printf("\nTime spent in matrix vector multiplication using cuda kernel : %4.6f sec\n", (double)((double)(cpu_end5 - cpu_start5)/CLOCKS_PER_SEC));
  
  
  cpu_start6 = clock();
  cudaMemcpy(d_cpu,d_gpu,memVec,cudaMemcpyDeviceToHost);
  cpu_end6 = clock(); 

  printf("\nTime spent in copying data from GPU to CPU : %4.6f sec\n", (double)((double)(cpu_end6 - cpu_start6)/CLOCKS_PER_SEC));

  printf("\nTotal time spent in matrix vector multiplication using GPU : %4.6f sec\n", (double)((double)(cpu_end6 - cpu_start4)/CLOCKS_PER_SEC));

  diff = 0;
  for(int i=0; i<column; i++) 
  {
   diff +=  d[i] - d_cpu[i];
  }
  if(diff == 0) 
    printf("\nResults from CPU function and GPU Kernel are same\n");  
  else 
    printf("\nResults from CPU function and GPU Kernel do not match\n");  
  printf("\n");

  free(a);
  free(c);
  free(d);
  free(d_cpu);
  free(dBlas);
  cudaFree(a_gpu);
  cudaFree(c_gpu);
  cudaFree(d_gpu);
  return 0;	
}

