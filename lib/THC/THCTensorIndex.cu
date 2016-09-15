#include "THC.h"
#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"
#include "THCDeviceUtils.cuh"
#include <algorithm> // for std::min
#include <cuda_fp16.h>

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexCopyLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexCopySmallIndex(TensorInfo<T, IndexType> dst,
                                    TensorInfo<T, IndexType> src,
                                    TensorInfo<long, IndexType> indices,
                                    int dstCopyDim,
                                    int srcCopyDim,
                                    IndexType innerSize,
                                    long dstCopyDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;

    if (dstIndex < dstCopyDimSize) {
      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
           linearIndex < innerSize;
           linearIndex += gridDim.x * blockDim.x) {
        IndexType dstOffset =
          IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);

        dstOffset += dstIndex * dst.strides[dstCopyDim];

        IndexType srcOffset =
          IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
        srcOffset += srcIndex * src.strides[srcCopyDim];

        dst.data[dstOffset] = src.data[srcOffset];
      }
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexCopySmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexCopyLargeIndex(TensorInfo<T, IndexType> dst,
                                    TensorInfo<T, IndexType> src,
                                    TensorInfo<long, IndexType> indices,
                                    int dstCopyDim,
                                    int srcCopyDim,
                                    IndexType innerSize,
                                    long dstCopyDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < innerSize * indices.sizes[0];
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;

    if (dstIndex < dstCopyDimSize) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex * dst.strides[dstCopyDim];

      IndexType srcOffset =
        IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
      srcOffset += srcIndex * src.strides[srcCopyDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

template <typename T>
__device__ T atomicAdd1(T *address, T val) {
  unsigned int * address_as_ui =
      (unsigned int *) (address - ((size_t)address & 3));
  unsigned int old = *address_as_ui, sum, newval, assumed;
  unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int sel = selectors[(size_t)address & 3];

  do {
    assumed = old;
    sum = val +  (T)__byte_perm(old, 0, (size_t)address & 3 | 0x4440);
    newval = __byte_perm(old, sum, sel);
    old = atomicCAS(address_as_ui, assumed, newval);
   } while (assumed != old);

   return (T)sum;
}

__device__ unsigned char atomicAdd(unsigned char *address, unsigned char val) {
  unsigned int * address_as_ui =
      (unsigned int *) (address - ((size_t)address & 3));
  unsigned int old = *address_as_ui, sum, newval, assumed;
  unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int sel = selectors[(size_t)address & 3];

  do {
    assumed = old;
    sum = val +  (unsigned char)__byte_perm(old, 0, (size_t)address & 3 | 0x4440);
    newval = __byte_perm(old, sum, sel);
    old = atomicCAS(address_as_ui, assumed, newval);
   } while (assumed != old);

   return (unsigned char)sum;
}

__device__ char atomicAdd(char *address, char val) {
  unsigned int * address_as_ui =
      (unsigned int *) (address - ((size_t)address & 3));
  unsigned int old = *address_as_ui, sum, newval, assumed;
  unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int sel = selectors[(size_t)address & 3];

  do {
    assumed = old;
    sum = val +  (char)__byte_perm(old, 0, (size_t)address & 3 | 0x4440);
    newval = __byte_perm(old, sum, sel);
    old = atomicCAS(address_as_ui, assumed, newval);
   } while (assumed != old);

   return (char)sum;
}

__device__ short atomicAdd(short *address, short val) {
  unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui, sum, newval, assumed;

  do {
    assumed = old;
    sum = val +  (short)__byte_perm(old, 0, ((size_t)address & 2) ? 0x4432 : 0x4410);
    newval = __byte_perm(old, sum, ((size_t)address & 2) ? 0x5410 : 0x3254);
    old = atomicCAS(address_as_ui, assumed, newval);
   } while (assumed != old);

   return (short)sum;
}

__device__ long atomicAdd(long *address, long val) {
  unsigned int * address_as_ui = (unsigned int *) (address);
  unsigned int old = *address_as_ui, newval, assumed;

  do {
    assumed = old;
    newval = val +  (long)old;
    old = atomicCAS(address_as_ui, assumed, newval);
   } while (assumed != old);

   return (long)newval;
}

#ifdef CUDA_HALF_INSTRUCTIONS
#define HALF_ADD(a, b) __hadd(a, b)
#else
#define HALF_ADD(a, b) __float2half(__half2float(a) + __half2float(b))
#endif

__device__ half atomicAdd(half *address, half val) {
  unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui, assumed;
  half hsum;

  do {
    assumed = old;
    half *hp = (half *)((char *)(&old) + ((size_t)address & 2));
    hsum = HALF_ADD(val, *hp);
    memcpy(hp, &hsum, sizeof(half));
    old = atomicCAS(address_as_ui, assumed, old);
   } while (assumed != old);

   return hsum;
}

// from CUDA C Programmic Guide
__device__  double atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                    __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexAddLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexAddSmallIndex(TensorInfo<T, IndexType> dst,
                                   TensorInfo<T, IndexType> src,
                                   TensorInfo<long, IndexType> indices,
                                   int dstAddDim,
                                   int srcAddDim,
                                   IndexType innerSize,
                                   long dstAddDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;

    if (dstIndex < dstAddDimSize) {
      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
           linearIndex < innerSize;
           linearIndex += gridDim.x * blockDim.x) {
        IndexType dstOffset =
          IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
        dstOffset += dstIndex * dst.strides[dstAddDim];

        IndexType srcOffset =
          IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
        srcOffset += srcIndex * src.strides[srcAddDim];

        atomicAdd(&dst.data[dstOffset], src.data[srcOffset]);
      }
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexAddSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexAddLargeIndex(TensorInfo<T, IndexType> dst,
                                   TensorInfo<T, IndexType> src,
                                   TensorInfo<long, IndexType> indices,
                                   int dstAddDim,
                                   int srcAddDim,
                                   IndexType innerSize,
                                   long dstAddDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < innerSize * indices.sizes[0];
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;

    if (dstIndex < dstAddDimSize) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
        IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      atomicAdd(&dst.data[dstOffset], src.data[srcOffset]);
    }
  }
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexFillLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int IdxDim>
__global__ void indexFillSmallIndex(TensorInfo<T, IndexType> dst,
                                    TensorInfo<long, IndexType> indices,
                                    int dstFillDim,
                                    IndexType innerSize,
                                    long dstFillDimSize,
                                    T val) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    // Lua indices begin at 1
    IndexType dstIndex_ =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;

    if (dstIndex < dstFillDimSize) {
      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
           linearIndex < innerSize;
           linearIndex += gridDim.x * blockDim.x) {
        IndexType dstOffset =
          IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
        dstOffset += dstIndex_ * dst.strides[dstFillDim];

        dst.data[dstOffset] = val;
      }
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexFillSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int IdxDim>
__global__ void indexFillLargeIndex(TensorInfo<T, IndexType> dst,
                                    TensorInfo<long, IndexType> indices,
                                    int dstFillDim,
                                    IndexType innerSize,
                                    long dstFillDimSize,
                                    T val) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < innerSize * indices.sizes[0];
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType dstIndex_ =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;

    if (dstIndex_ < dstFillDimSize) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex_ * dst.strides[dstFillDim];

      dst.data[dstOffset] = val;
    }
  }
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexSelectLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndex(TensorInfo<T, IndexType> dst,
                                      TensorInfo<T, IndexType> src,
                                      TensorInfo<long, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType innerSize,
                                      long srcSelectDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    // Lua indices begin at 1
    IndexType srcIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;

    if (srcIndex < srcSelectDimSize) {
      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
           linearIndex < innerSize;
           linearIndex += gridDim.x * blockDim.x) {
        IndexType dstOffset =
          IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
        dstOffset += dstIndex * dst.strides[dstSelectDim];

        IndexType srcOffset =
          IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
        srcOffset += srcIndex * src.strides[srcSelectDim];

        dst.data[dstOffset] = src.data[srcOffset];
      }
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexSelectSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectLargeIndex(TensorInfo<T, IndexType> dst,
                                      TensorInfo<T, IndexType> src,
                                      TensorInfo<long, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType totalSize,
                                      IndexType innerSize,
                                      long srcSelectDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType srcIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;

    if (srcIndex < srcSelectDimSize) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

#include "generic/THCTensorIndex.cu"
#include "THCGenerateAllTypes.h"
