// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/rocm/rccl_dynamic_symbols.h"

#include <iree/base/alignment.h>
#include <iree/base/status.h>
#include <iree/base/string_view.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "third_party/rccl/rccl.h"

static const char* RCCLLoaderSearchNames[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "rccl.dll",
#else
    "librccl.so",
#endif
};

static iree_status_t iree_hal_rocm_rccl_check_version(
  iree_dynamic_library_t* rccl_library) {
  ncclResult_t (*ncclGetVersion)(int*) = NULL;

  iree_status_t status = iree_dynamic_library_lookup_symbol(
      rccl_library, "ncclGetVersion", (void**)&ncclGetVersion);
  if(!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "ncclGetVersion symbol not found in dynamic library");
  }

  int rccl_version;
  ncclResult_t result = ncclGetVersion(&rccl_version);
  if(result != ncclSuccess) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "ncclGetVersion() failed with error %d", result);
  }

  int major = 0;
  int minor = 0;
  int patch = 0;
  if (rccl_version < 20000) {
    major = rccl_version / 1000;
    minor = (rccl_version % 1000) / 100;
  } else {
    major = rccl_version / 10000;
    minor = (rccl_version % 10000) / 100;
  }
  patch = rccl_version % 100;
  int required_minimum_version = NCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, 0);
  if (major != NCCL_MAJOR || rccl_version < required_minimum_version) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "NCCL version is %d.%d.%d, but >=%d.%d and <%d is required", major,
        minor, patch, NCCL_MAJOR, NCCL_MINOR, NCCL_MAJOR + 1);
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_rccl_dynamic_symbols_resolve_all(
    iree_hal_rocm_rccl_dynamic_symbols_t* syms) {
#define RCCL_PFN_DECL(rcclSymbolName, ...)                              \
  {                                                                     \
    static const char* name = #rcclSymbolName;                          \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(            \
        syms->loader_library, name, (void**)&syms->rcclSymbolName));    \
  }
#define RCCL_PFN_STR_DECL(rcclSymbolName, ...)                          \
  {                                                                     \
    static const char* name = #rcclSymbolName;                          \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(            \
        syms->loader_library, name, (void**)&syms->rcclSymbolName));    \
  }
#include "experimental/rocm/rccl_dynamic_symbol_table.h"    // IWYU pragma: keep
#undef RCCL_PFN_DECL
#undef RCCL_PFN_STR_DECL
  return iree_ok_status();
}

iree_status_t iree_hal_rocm_rccl_dynamic_symbols_initialize(
    iree_allocator_t allocator, 
    const iree_hal_rocm_dynamic_symbols_t* rocm_syms,
    iree_hal_rocm_rccl_dynamic_symbols_t* out_syms) {
  if (!rocm_syms->loader_library) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "ROCm dynamic symbols must be resolved prior to loading RCCL symbols");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_syms, 0, sizeof(*out_syms));
  iree_status_t status = iree_dynamic_library_load_from_files(
    IREE_ARRAYSIZE(RCCLLoaderSearchNames), 
    RCCLLoaderSearchNames, IREE_DYNAMIC_LIBRARY_FLAG_NONE, 
    allocator, &out_syms->loader_library);
  if(iree_status_is_not_found(status)){
    iree_status_ignore(status);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "RCCL runtime library version %d.%d and greater not available; "
        "ensure installed and the shared library (rccl.dll/librccl.so) "
        "is on your PATH/LD_LIBRARY_PATH.",
        NCCL_MAJOR, NCCL_MINOR);
  }

  if(iree_status_is_ok(status)){
    status = iree_hal_rocm_rccl_check_version(out_syms->loader_library);
  }
  
  if(iree_status_is_ok(status)){
    status = iree_hal_rocm_rccl_dynamic_symbols_resolve_all(out_syms);
  }
  
  if(!iree_status_is_ok(status)){
    iree_dynamic_library_release(out_syms->loader_library);
    out_syms->loader_library = NULL;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_rocm_rccl_dynamic_symbols_deinitialize(
    iree_hal_rocm_rccl_dynamic_symbols_t* syms) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_dynamic_library_release(syms->loader_library);
  memset(syms, 0, sizeof(*syms));
  IREE_TRACE_ZONE_END(z0);
}