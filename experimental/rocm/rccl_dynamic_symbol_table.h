// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

RCCL_PFN_DECL(ncclGetVersion, int*)
RCCL_PFN_DECL(ncclGetUniqueId, ncclUniqueId*)
RCCL_PFN_DECL(ncclCommInitRankConfig, ncclComm_t*, int, ncclUniqueId, int,
                   ncclConfig_t*)
RCCL_PFN_DECL(ncclCommInitRank, ncclComm_t*, int, ncclUniqueId, int)
RCCL_PFN_DECL(ncclCommInitAll, ncclComm_t*, int, const int*)
RCCL_PFN_DECL(ncclCommSplit, ncclComm_t, int, int, ncclComm_t*,
                   ncclConfig_t*)
RCCL_PFN_DECL(ncclCommFinalize, ncclComm_t)
RCCL_PFN_DECL(ncclCommDestroy, ncclComm_t)
RCCL_PFN_DECL(ncclCommAbort, ncclComm_t)
RCCL_PFN_STR_DECL(ncclGetErrorString, ncclResult_t)
RCCL_PFN_STR_DECL(ncclGetLastError, ncclComm_t)
RCCL_PFN_DECL(ncclCommGetAsyncError, ncclComm_t, ncclResult_t*)
RCCL_PFN_DECL(ncclCommCount, const ncclComm_t, int*)
RCCL_PFN_DECL(ncclCommCuDevice, const ncclComm_t, int*)
RCCL_PFN_DECL(ncclCommUserRank, const ncclComm_t, int*)
RCCL_PFN_DECL(ncclRedOpCreatePreMulSum, ncclRedOp_t*, void*,
                   ncclDataType_t, ncclScalarResidence_t, ncclComm_t)
RCCL_PFN_DECL(ncclRedOpDestroy, ncclRedOp_t, ncclComm_t)
RCCL_PFN_DECL(ncclReduce, const void*, void*, size_t, ncclDataType_t,
                   ncclRedOp_t, int, ncclComm_t, hipStream_t)
RCCL_PFN_DECL(ncclBcast, void*, size_t, ncclDataType_t, int, ncclComm_t,
                   hipStream_t)
RCCL_PFN_DECL(ncclBroadcast, const void*, void*, size_t, ncclDataType_t,
                   int, ncclComm_t, hipStream_t)
RCCL_PFN_DECL(ncclAllReduce, const void*, void*, size_t, ncclDataType_t,
                   ncclRedOp_t, ncclComm_t, hipStream_t)
RCCL_PFN_DECL(ncclReduceScatter, const void*, void*, size_t,
                   ncclDataType_t, ncclRedOp_t, ncclComm_t, hipStream_t)
RCCL_PFN_DECL(ncclAllGather, const void*, void*, size_t, ncclDataType_t,
                   ncclComm_t, hipStream_t)
RCCL_PFN_DECL(ncclSend, const void*, size_t, ncclDataType_t, int,
                   ncclComm_t, hipStream_t)
RCCL_PFN_DECL(ncclRecv, void*, size_t, ncclDataType_t, int, ncclComm_t,
                   hipStream_t)
RCCL_PFN_DECL(ncclGroupStart)
RCCL_PFN_DECL(ncclGroupEnd)