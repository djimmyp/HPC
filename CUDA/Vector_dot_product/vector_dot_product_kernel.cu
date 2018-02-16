#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_

#define BLOCK_SIZE 32
#define GRID_SIZE 1280

texture<float> A_on_tex;
texture<float> B_on_tex;

__global__ void vector_dot_product_kernel(float* P, const float* A, const float* B, int num_elements)
{
		__shared__ float thread_sums[ BLOCK_SIZE ];
		float local_sum = 0;
		unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int pitch = blockDim.x * gridDim.x;
		int i;
		for(int z = tid; z < num_elements; z+= pitch)
		{
			float A_element = tex1Dfetch(A_on_tex, (z));
			float B_element = tex1Dfetch(B_on_tex, (z));
			local_sum += A_element * B_element;
		}
		thread_sums[threadIdx.x] = local_sum;
		__syncthreads();

	/* Reduction performed in each block */
	i = BLOCK_SIZE / 2;   	
	while ( i != 0 ) 
	{
		
		if ( threadIdx.x < i ) {
		
			thread_sums[threadIdx.x] += thread_sums[ threadIdx.x + i ];
		}
		__syncthreads();

		i = i / 2;
	}

	if(threadIdx.x == 0)
	{
		atomicAdd(P, thread_sums[0]);
	}		

}

#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
