// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include "experimental/rocm/rocm_dynamic_symbols.h"
#include "experimental/rocm/rccl_dynamic_symbols.h"

#include <iostream>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace rocm {
namespace {

#define ROCM_CHECK_ERRORS(expr)    \
  {                                \
    hipError_t status = expr;      \
    ASSERT_EQ(hipSuccess, status); \
  }

TEST(DynamicSymbolsTest, CreateFromSystemLoader) {
  iree_hal_rocm_dynamic_symbols_t symbols;
  iree_status_t status = iree_hal_rocm_dynamic_symbols_initialize(
      iree_allocator_system(), &symbols);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Symbols cannot be loaded, skipping test.";
    GTEST_SKIP();
  }

  int device_count = 0;
  ROCM_CHECK_ERRORS(symbols.hipInit(0));
  ROCM_CHECK_ERRORS(symbols.hipGetDeviceCount(&device_count));
  if (device_count > 0) {
    hipDevice_t device;
    ROCM_CHECK_ERRORS(symbols.hipDeviceGet(&device, /*ordinal=*/0));
  }

  iree_hal_rocm_dynamic_symbols_deinitialize(&symbols);
}

#define RCCL_CHECK_ERRORS(expr)     \
  {                                 \
    ncclResult_t status = expr;     \
    ASSERT_EQ(ncclSuccess, status); \
  }         

TEST(RCCLDynamicSymbolsTest, CreateFromSystemLoader) {
  iree_hal_rocm_dynamic_symbols_t rocm_symbols;
  iree_status_t status = iree_hal_rocm_dynamic_symbols_initialize(
      iree_allocator_system(), &rocm_symbols);
  if(!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    std::cerr << "ROCm symbols cannot be loaded, skipping test.";
    GTEST_SKIP();
  }

  iree_hal_rocm_rccl_dynamic_symbols_t rccl_symbols;
  status = iree_hal_rocm_rccl_dynamic_symbols_initialize(
      iree_allocator_system(), &rocm_symbols, &rccl_symbols);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    std::cerr << "ROCm RCCL symbols cannot be loaded, skipping test.";
    GTEST_SKIP();
  }

  int rccl_version;
  RCCL_CHECK_ERRORS(rccl_symbols.ncclGetVersion(&rccl_version));
  ASSERT_EQ(NCCL_VERSION_CODE, rccl_version);
  iree_hal_rocm_rccl_dynamic_symbols_deinitialize(&rccl_symbols);
  iree_hal_rocm_dynamic_symbols_deinitialize(&rocm_symbols);
}

}  // namespace
}  // namespace rocm
}  // namespace hal
}  // namespace iree
