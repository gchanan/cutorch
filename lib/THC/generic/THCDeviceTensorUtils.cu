#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCDeviceTensorUtils.cu"
#else

/// Constructs a THCDeviceTensor initialized from a THCudaTensor. Will
/// error if the dimensionality does not match exactly.
template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>
toDeviceTensor(THCState* state, THCTensor* t);

template <typename T, int Dim, typename IndexT>
THCDeviceTensor<T, Dim, IndexT, DefaultPtrTraits>
toDeviceTensor(THCState* state, THCTensor* t) {
  return toDeviceTensor<T, Dim, IndexT, DefaultPtrTraits>(state, t);
}

template <typename T, int Dim>
THCDeviceTensor<T, Dim, int, DefaultPtrTraits>
toDeviceTensor(THCState* state, THCTensor* t) {
  return toDeviceTensor<T, Dim, int, DefaultPtrTraits>(state, t);
}

/*template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
//THC_API
THCDeviceTensor<T, Dim, IndexT, PtrTraits>
toDeviceTensor(THCState* state, THCudaTensor* t);

template <typename T, int Dim, typename IndexT>
//THC_API
THCDeviceTensor<T, Dim, IndexT, DefaultPtrTraits>
toDeviceTensor(THCState* state, THCudaTensor* t) {
  return toDeviceTensor<T, Dim, IndexT, DefaultPtrTraits>(state, t);
}

template <typename T, int Dim>
THCDeviceTensor<T, Dim, int, DefaultPtrTraits>
toDeviceTensor(THCState* state, THCudaTensor* t) {
  return toDeviceTensor<T, Dim, int, DefaultPtrTraits>(state, t);
}*/

#include "THCDeviceTensorUtils-inl.cuh"

#endif
