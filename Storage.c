#include "torch/utils.h"
#include "THC.h"
#include "THFile.h"
#include "luaT.h"

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,CReal,Storage_,NAME)
#define torch_Storage TH_CONCAT_STRING_3(torch.,CReal,Storage)
#define cutorch_Storage_(NAME) TH_CONCAT_4(cutorch_,CReal,Storage_,NAME)
#define cutorch_StorageCopy_(NAME) TH_CONCAT_4(cutorch_,Real,StorageCopy_,NAME)

#include "generic/CStorage.c"
#include "THCGenerateAllTypes.h"
#ifndef CUDA_HALF_TENSOR
#include "generic/CStorageCopy.c"
#include "THGenerateHalfType.h"
#endif
