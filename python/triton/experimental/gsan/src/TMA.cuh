#pragma once

#include "Compat.h"

namespace gsan {

struct TensorDescInfo {
  static constexpr int kShapeWordBase = 8;
  static constexpr int kStrideWordBase = 3;
  static constexpr int kStrideUnitBytes = 16;

  uintptr_t base;
  const uint32_t *words;
  int rank;
  int bytesPerElem;

  GSAN_DEVICE int64_t shape(int dim) const {
    return static_cast<int64_t>(words[kShapeWordBase + rank - 1 - dim]) + 1;
  }

  GSAN_DEVICE uintptr_t byteStride(int dim) const {
    if (dim + 1 == rank)
      return bytesPerElem;
    return static_cast<uintptr_t>(words[kStrideWordBase + rank - 2 - dim]) *
           kStrideUnitBytes;
  }
};

// The native AMD descriptor ABI is flat {i32 x 12/20}, with tensor
// dimensions in elements and non-inner strides in 48-bit element units.
// Tile dimensions and LDS padding describe physical instruction distribution;
// the checker receives the logical tile shape separately.
struct TDMDescInfo {
  uintptr_t base;
  const uint32_t *words;
  int rank;
  int bytesPerElem;

  GSAN_DEVICE bool enabled() const { return (words[0] & 3) != 0; }

  GSAN_DEVICE int64_t shape(int dim) const {
    switch (rank - 1 - dim) {
    case 0:
      return static_cast<uint32_t>((words[5] >> 16) | (words[6] << 16));
    case 1:
      return static_cast<uint32_t>((words[6] >> 16) | (words[7] << 16));
    case 2:
      return words[12];
    case 3:
      return words[13];
    default:
      return static_cast<uint32_t>((words[17] >> 16) | (words[18] << 16));
    }
  }

  GSAN_DEVICE uintptr_t byteStride(int dim) const {
    uintptr_t stride;
    switch (rank - 1 - dim) {
    case 0:
      stride = 1;
      break;
    case 1:
      stride = words[9] | (static_cast<uintptr_t>(words[10] & 0xFFFF) << 32);
      break;
    case 2:
      stride = (words[10] >> 16) | (static_cast<uintptr_t>(words[11]) << 16);
      break;
    case 3:
      stride = words[14] | (static_cast<uintptr_t>(words[15] & 0xFFFF) << 32);
      break;
    default:
      stride = words[16] | (static_cast<uintptr_t>(words[17] & 0xFFFF) << 32);
      break;
    }
    return stride * bytesPerElem;
  }
};

inline GSAN_DEVICE TDMDescInfo decodeTDMDesc(const void *descriptor, int rank,
                                            int bytesPerElem) {
  auto *words = reinterpret_cast<const uint32_t *>(descriptor);
  uintptr_t base =
      words[2] | (static_cast<uintptr_t>(words[3] & 0x7FFFFFFF) << 32);
  return {base, words, rank, bytesPerElem};
}

#if defined(__AMDGCN__)
inline GSAN_DEVICE TDMDescInfo decodeTensorDesc(const void *descriptor,
                                                int rank, int bytesPerElem) {
  return decodeTDMDesc(descriptor, rank, bytesPerElem);
}
#else
inline GSAN_DEVICE TensorDescInfo decodeTensorDesc(const void *descriptor,
                                                   int rank, int bytesPerElem) {
  return {*reinterpret_cast<const uintptr_t *>(descriptor),
          reinterpret_cast<const uint32_t *>(descriptor), rank, bytesPerElem};
}

#endif

} // namespace gsan
