/* Copyright 2025 The JAX Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/python/dlpack_types.h"

#include <Python.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "include/dlpack/dlpack.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "xla/python/dlpack_types_base.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {
// Helper to convert an absl::UnimplementedError to xla::Unimplemented
template <typename T>
absl::StatusOr<T> ConvertUnimplementedError(absl::StatusOr<T> result) {
  if (!result.ok() && absl::IsUnimplemented(result.status())) {
    return xla::Unimplemented("%s", result.status().message());
  }
  return result;
}
}  // namespace

absl::StatusOr<DLDataType> PrimitiveTypeToDLDataType(PrimitiveType type) {
  return ConvertUnimplementedError(
      dlpack_types_base::PrimitiveTypeToDLDataType(type));
}

absl::StatusOr<PrimitiveType> DLDataTypeToPrimitiveType(DLDataType type) {
  return ConvertUnimplementedError(
      dlpack_types_base::DLDataTypeToPrimitiveType(type));
}

absl::StatusOr<nanobind::dlpack::dtype> PrimitiveTypeToNbDLDataType(
    xla::PrimitiveType type) {
  TF_ASSIGN_OR_RETURN(DLDataType dl_type, PrimitiveTypeToDLDataType(type));

  nanobind::dlpack::dtype nb_type;
  nb_type.lanes = dl_type.lanes;
  nb_type.bits = dl_type.bits;
  nb_type.code = dl_type.code;

  return nb_type;
}

}  // namespace xla
