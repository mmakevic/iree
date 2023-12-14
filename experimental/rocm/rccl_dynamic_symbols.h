// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_RCCL_DYNAMIC_SYMBOLS_H_
#define IREE_HAL_RCCL_DYNAMIC_SYMBOLS_H_

#include "experimental/rocm/rocm_dynamic_symbols.h"
#include "experimental/rocm/rccl_headers.h"
#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// DynamicSymbols allow loading dynamically a subset of RCCL API. It
// loads all the function declared in `rccl_dynamic_symbol_tables.h` and fail if
// any of the symbol is not available. The functions signatures are matching
// the declarations in `rccl.h`.

// RCCL API dynamic symbols
typedef struct iree_hal_rocm_rccl_dynamic_symbols_t {
  iree_dynamic_library_t* loader_library;

#define RCCL_PFN_DECL(rcclSymbolName, ...) \
  ncclResult_t (*rcclSymbolName)(__VA_ARGS__);
#define RCCL_PFN_STR_DECL(rcclSymbolName, ...) \
  const char* (*rcclSymbolName)(__VA_ARGS__);
#include "experimental/rocm/rccl_dynamic_symbol_table.h"  // IWYU pragma: export
#undef RCCL_PFN_DECL
#undef RCCL_PFN_STR_DECL
} iree_hal_rocm_rccl_dynamic_symbols_t;

// Initializes |out_syms| in-place with dynamically loaded RCCL symbols.
// iree_hal_rocm_rccl_dynamic_symbols_deinitialize must be used to release the
// library resources.
iree_status_t iree_hal_rocm_rccl_dynamic_symbols_initialize(
    iree_allocator_t allocator, 
    const iree_hal_rocm_dynamic_symbols_t* rocm_syms,
    iree_hal_rocm_rccl_dynamic_symbols_t* out_syms);

// Deinitializes |syms| by unloading the backing library. All function pointers
// will be invalidated. They _may_ still work if there are other reasons the
// library remains loaded so be careful.
void iree_hal_rocm_rccl_dynamic_symbols_deinitialize(
    iree_hal_rocm_rccl_dynamic_symbols_t* syms);


#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_ROCM_DYNAMIC_SYMBOLS_H_