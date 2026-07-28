#include "Profiler/RocprofSDK/RocprofSDKPCSamplingCorrection.h"

#include <algorithm>
#include <array>
#include <utility>

namespace proton::pc_sampling_correction {

namespace {

bool startsWith(std::string_view str, std::string_view prefix) {
  return str.size() >= prefix.size() &&
         str.compare(0, prefix.size(), prefix) == 0;
}

} // namespace

InstructionKind classify(std::string_view instruction) {
  static constexpr std::array<std::string_view, 13> regularInternals = {
      "s_nop",
      "s_sleep",
      "s_monitor_sleep",
      "s_wait",
      "s_barrier_wait",
      // The upstream rocprofiler workaround currently treats s_setprio as
      // regular internal, while one upstream test asks for hardware
      // confirmation. Keep this searchable if the upstream answer changes.
      "s_setprio",
      "s_delay_alu",
      "s_sethalt",
      "s_setkill",
      "s_singleuse_vdst",
      "s_round_mode",
      "s_denorm_mode",
      "s_version",
  };

  if (startsWith(instruction, "s_icache_inv"))
    return InstructionKind::SICacheInv;

  for (auto prefix : regularInternals) {
    if (startsWith(instruction, prefix))
      return InstructionKind::RegularInternal;
  }

  return InstructionKind::Ext;
}

bool shouldCorrect(bool enabled, std::string_view decodedInstruction,
                   const CorrectionInput &input) {
  if (!enabled)
    return false;
  if (classify(decodedInstruction) == InstructionKind::Ext)
    return false;
  if (input.hasWaveIssued && input.waveIssued)
    return true;
  if (input.hasReasonNotIssued &&
      input.reasonNotIssued ==
          ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN)
    return true;
  return false;
}

void CodeObjectClassification::addSymbol(uint64_t symbolVaddr,
                                         uint64_t symbolSize,
                                         const DecodeFn &decode) {
  auto window = std::make_shared<InstructionStreamWindow>();
  const uint64_t end = symbolVaddr + symbolSize;

  for (uint64_t offset = symbolVaddr; offset < end;) {
    auto instruction = decode(offset);
    if (!instruction || instruction->size == 0 ||
        offset + instruction->size > end)
      break;

    switch (classify(instruction->text)) {
    case InstructionKind::Ext:
      if (window->hasExt1 &&
          (window->regularInternalCount > 0 || window->icacheInvCount > 0)) {
        window->ext2Offset = offset;
        window->hasExt2 = true;
      }
      window = std::make_shared<InstructionStreamWindow>();
      window->ext1Offset = offset;
      window->hasExt1 = true;
      break;
    case InstructionKind::RegularInternal:
      ++window->regularInternalCount;
      window->regularInternalTotalBytes += instruction->size;
      entries_.push_back({offset, window});
      break;
    case InstructionKind::SICacheInv:
      ++window->icacheInvCount;
      entries_.push_back({offset, window});
      break;
    }

    offset += instruction->size;
  }
}

void CodeObjectClassification::sort() {
  std::sort(entries_.begin(), entries_.end(),
            [](const InternalEntry &a, const InternalEntry &b) {
              return a.offset < b.offset;
            });
}

std::optional<InternalEntry>
CodeObjectClassification::find(uint64_t offset) const {
  auto it = std::lower_bound(entries_.begin(), entries_.end(), offset,
                             [](const InternalEntry &entry, uint64_t value) {
                               return entry.offset < value;
                             });
  if (it == entries_.end() || it->offset != offset)
    return std::nullopt;
  return *it;
}

void CorrectionManager::publish(
    uint64_t codeObjectId,
    std::shared_ptr<const CodeObjectClassification> classification) {
  std::lock_guard<std::mutex> lock(mutex_);
  unavailableCodeObjects_.erase(codeObjectId);
  classifications_.insert_or_assign(codeObjectId, std::move(classification));
}

void CorrectionManager::markUnavailable(uint64_t codeObjectId) {
  std::lock_guard<std::mutex> lock(mutex_);
  classifications_.erase(codeObjectId);
  unavailableCodeObjects_[codeObjectId] = true;
}

void CorrectionManager::erase(uint64_t codeObjectId) {
  std::lock_guard<std::mutex> lock(mutex_);
  classifications_.erase(codeObjectId);
  unavailableCodeObjects_.erase(codeObjectId);
}

bool CorrectionManager::needsCorrection(uint64_t codeObjectId) const {
  if (!enabled())
    return false;
  std::lock_guard<std::mutex> lock(mutex_);
  return classifications_.count(codeObjectId) > 0 ||
         unavailableCodeObjects_.count(codeObjectId) > 0;
}

std::optional<InternalEntry> CorrectionManager::lookup(uint64_t codeObjectId,
                                                       uint64_t offset) const {
  std::shared_ptr<const CodeObjectClassification> classification;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = classifications_.find(codeObjectId);
    if (it != classifications_.end())
      classification = it->second;
  }
  if (!classification)
    return std::nullopt;
  return classification->find(offset);
}

CorrectionResult
CorrectionManager::correct(std::string_view decodedInstruction,
                           const CorrectionInput &input) const {
  CorrectionResult result{CorrectionAction::Keep, input.pc};
  if (!shouldCorrect(enabled(), decodedInstruction, input))
    return result;

  auto entry = lookup(input.pc.code_object_id, input.pc.code_object_offset);
  if (!entry)
    return {CorrectionAction::Drop, input.pc};

  const auto &window = *entry->window;
  if (!window.hasExt1 || !window.hasExt2)
    return {CorrectionAction::Drop, input.pc};

  if (window.regularInternalCount == 0) {
    result.pc.code_object_offset = window.ext1Offset;
  } else if (window.icacheInvCount == 0) {
    result.pc.code_object_offset = window.ext2Offset;
  } else {
    const uint64_t trailingRegularBoundary =
        window.ext2Offset - window.regularInternalTotalBytes;
    if (input.pc.code_object_offset < trailingRegularBoundary)
      result.pc.code_object_offset = window.ext1Offset;
    else
      return {CorrectionAction::Drop, input.pc};
  }

  return result;
}

} // namespace proton::pc_sampling_correction
