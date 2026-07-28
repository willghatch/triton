#ifndef PROTON_PROFILER_ROCPROFSDK_PC_SAMPLING_CORRECTION_H_
#define PROTON_PROFILER_ROCPROFSDK_PC_SAMPLING_CORRECTION_H_

#include "rocprofiler-sdk/pc_sampling.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace proton::pc_sampling_correction {

enum class InstructionKind { Ext, RegularInternal, SICacheInv };

enum class CorrectionAction { Keep, Drop };

struct DecodedInstruction {
  std::string text;
  uint64_t size{0};
};

struct InstructionStreamWindow {
  uint64_t ext1Offset{0};
  uint64_t ext2Offset{0};
  bool hasExt1{false};
  bool hasExt2{false};
  uint64_t regularInternalTotalBytes{0};
  uint16_t regularInternalCount{0};
  uint16_t icacheInvCount{0};
};

struct InternalEntry {
  uint64_t offset{0};
  std::shared_ptr<const InstructionStreamWindow> window;
};

struct CorrectionInput {
  rocprofiler_pc_t pc{0, 0};
  bool hasWaveIssued{false};
  bool waveIssued{false};
  bool hasReasonNotIssued{false};
  uint32_t reasonNotIssued{0};
};

struct CorrectionResult {
  CorrectionAction action{CorrectionAction::Keep};
  rocprofiler_pc_t pc{0, 0};
};

InstructionKind classify(std::string_view instruction);

constexpr bool isGfx1250(uint32_t gfxTargetVersion) {
  return gfxTargetVersion >= 125000u && gfxTargetVersion < 125100u;
}

bool shouldCorrect(bool enabled, std::string_view decodedInstruction,
                   const CorrectionInput &input);

class CodeObjectClassification {
public:
  using DecodeFn =
      std::function<std::optional<DecodedInstruction>(uint64_t offset)>;

  void addSymbol(uint64_t symbolVaddr, uint64_t symbolSize,
                 const DecodeFn &decode);
  void sort();
  std::optional<InternalEntry> find(uint64_t offset) const;

  const std::vector<InternalEntry> &entries() const { return entries_; }
  bool empty() const { return entries_.empty(); }

private:
  std::vector<InternalEntry> entries_;
};

class CorrectionManager {
public:
  void setEnabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_relaxed);
  }
  bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

  void publish(uint64_t codeObjectId,
               std::shared_ptr<const CodeObjectClassification> classification);
  void markUnavailable(uint64_t codeObjectId);
  void erase(uint64_t codeObjectId);
  bool needsCorrection(uint64_t codeObjectId) const;
  std::optional<InternalEntry> lookup(uint64_t codeObjectId,
                                      uint64_t offset) const;
  CorrectionResult correct(std::string_view decodedInstruction,
                           const CorrectionInput &input) const;

private:
  std::atomic<bool> enabled_{false};
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<const CodeObjectClassification>>
      classifications_;
  std::unordered_map<uint64_t, bool> unavailableCodeObjects_;
};

} // namespace proton::pc_sampling_correction

#endif // PROTON_PROFILER_ROCPROFSDK_PC_SAMPLING_CORRECTION_H_
