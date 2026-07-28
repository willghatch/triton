#include "Profiler/RocprofSDK/RocprofSDKPCSamplingCorrection.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace proton::pc_sampling_correction;

namespace {

struct FakeStream {
  std::vector<DecodedInstruction> instructions;
  size_t next{0};

  std::optional<DecodedInstruction> operator()(uint64_t) {
    if (next >= instructions.size())
      return std::nullopt;
    return instructions[next++];
  }

  uint64_t totalSize() const {
    uint64_t total = 0;
    for (const auto &instruction : instructions)
      total += instruction.size;
    return total;
  }
};

CodeObjectClassification
buildSingleSymbol(std::vector<DecodedInstruction> instructions) {
  FakeStream stream{std::move(instructions)};
  CodeObjectClassification classification;
  classification.addSymbol(0, stream.totalSize(),
                           [&](uint64_t offset) { return stream(offset); });
  classification.sort();
  return classification;
}

CorrectionInput makeInput(uint64_t codeObjectId, uint64_t offset,
                          bool waveIssued = true,
                          uint32_t reasonNotIssued = 0) {
  CorrectionInput input;
  input.pc.code_object_id = codeObjectId;
  input.pc.code_object_offset = offset;
  input.hasWaveIssued = true;
  input.waveIssued = waveIssued;
  input.hasReasonNotIssued = true;
  input.reasonNotIssued = reasonNotIssued;
  return input;
}

constexpr uint32_t kArbiterNotWin =
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN;

} // namespace

TEST(RocprofSDKPCSamplingCorrectionClassify, RecognizesRegularInternals) {
  EXPECT_EQ(classify("s_nop 0"), InstructionKind::RegularInternal);
  EXPECT_EQ(classify("s_sleep 1"), InstructionKind::RegularInternal);
  EXPECT_EQ(classify("s_waitcnt 0"), InstructionKind::RegularInternal);
  EXPECT_EQ(classify("s_wait_idle"), InstructionKind::RegularInternal);
  EXPECT_EQ(classify("s_barrier_wait -1"), InstructionKind::RegularInternal);
}

TEST(RocprofSDKPCSamplingCorrectionClassify, RecognizesIcacheInv) {
  EXPECT_EQ(classify("s_icache_inv"), InstructionKind::SICacheInv);
}

TEST(RocprofSDKPCSamplingCorrectionClassify, TreatsExternalInstructionsAsExt) {
  EXPECT_EQ(classify("v_add_f32_e32 v0, v1, v2"), InstructionKind::Ext);
  EXPECT_EQ(classify("s_mov_b32 s0, 1"), InstructionKind::Ext);
  EXPECT_EQ(classify("s_branch 8"), InstructionKind::Ext);
}

TEST(RocprofSDKPCSamplingCorrectionClassify, SetPrioMatchesUpstreamBranch) {
  EXPECT_EQ(classify("s_setprio 1"), InstructionKind::RegularInternal);
}

TEST(RocprofSDKPCSamplingCorrectionBuild, TracksLeadingAndTrailingInternals) {
  auto leading = buildSingleSymbol({{"s_nop", 4}, {"s_nop", 4}, {"v_add", 4}});
  ASSERT_EQ(leading.entries().size(), 2u);
  EXPECT_FALSE(leading.entries()[0].window->hasExt1);

  auto trailing = buildSingleSymbol({{"v_add", 4}, {"s_nop", 4}, {"s_nop", 4}});
  ASSERT_EQ(trailing.entries().size(), 2u);
  EXPECT_TRUE(trailing.entries()[0].window->hasExt1);
  EXPECT_EQ(trailing.entries()[0].window->ext1Offset, 0u);
  EXPECT_FALSE(trailing.entries()[0].window->hasExt2);
}

TEST(RocprofSDKPCSamplingCorrectionBuild, BuildsForwardAndBackwardWindows) {
  auto regularOnly = buildSingleSymbol(
      {{"v_add", 4}, {"s_nop", 4}, {"s_nop", 4}, {"v_mov", 4}});
  ASSERT_EQ(regularOnly.entries().size(), 2u);
  EXPECT_EQ(regularOnly.entries()[0].window->regularInternalCount, 2u);
  EXPECT_EQ(regularOnly.entries()[0].window->icacheInvCount, 0u);
  EXPECT_EQ(regularOnly.entries()[0].window->regularInternalTotalBytes, 8u);
  EXPECT_EQ(regularOnly.entries()[0].window->ext2Offset, 12u);

  auto icacheOnly = buildSingleSymbol(
      {{"v_add", 4}, {"s_icache_inv", 4}, {"s_icache_inv", 4}, {"v_mov", 4}});
  ASSERT_EQ(icacheOnly.entries().size(), 2u);
  EXPECT_EQ(icacheOnly.entries()[0].window->regularInternalCount, 0u);
  EXPECT_EQ(icacheOnly.entries()[0].window->icacheInvCount, 2u);
  EXPECT_EQ(icacheOnly.entries()[0].window->ext1Offset, 0u);
}

TEST(RocprofSDKPCSamplingCorrectionBuild, SortsEntriesAcrossSymbols) {
  CodeObjectClassification classification;
  FakeStream later{{{"v_add", 4}, {"s_nop", 4}, {"v_mov", 4}}};
  classification.addSymbol(100, later.totalSize(),
                           [&](uint64_t offset) { return later(offset); });
  FakeStream earlier{{{"v_add", 4}, {"s_nop", 4}, {"v_mov", 4}}};
  classification.addSymbol(0, earlier.totalSize(),
                           [&](uint64_t offset) { return earlier(offset); });

  classification.sort();

  ASSERT_EQ(classification.entries().size(), 2u);
  EXPECT_EQ(classification.entries()[0].offset, 4u);
  EXPECT_EQ(classification.entries()[1].offset, 104u);
  EXPECT_TRUE(classification.find(4).has_value());
  EXPECT_FALSE(classification.find(0).has_value());
}

TEST(RocprofSDKPCSamplingCorrectionGate, RequiresInternalAndImpossibleState) {
  EXPECT_FALSE(shouldCorrect(false, "s_nop", makeInput(1, 4)));
  EXPECT_FALSE(shouldCorrect(true, "v_add", makeInput(1, 4)));
  EXPECT_FALSE(
      shouldCorrect(true, "s_nop", makeInput(1, 4, false, /*reason=*/0)));
  EXPECT_TRUE(shouldCorrect(true, "s_nop", makeInput(1, 4, true)));
  EXPECT_TRUE(shouldCorrect(true, "s_icache_inv",
                            makeInput(1, 4, false, kArbiterNotWin)));
}

TEST(RocprofSDKPCSamplingCorrectionManager, CorrectsRegularOnlyForwardToExt2) {
  CorrectionManager manager;
  manager.setEnabled(true);
  manager.publish(
      1, std::make_shared<CodeObjectClassification>(buildSingleSymbol(
             {{"v_add", 4}, {"s_nop", 4}, {"s_nop", 4}, {"v_mov", 4}})));

  auto result = manager.correct("s_nop", makeInput(1, 4));

  ASSERT_EQ(result.action, CorrectionAction::Keep);
  EXPECT_EQ(result.pc.code_object_offset, 12u);
}

TEST(RocprofSDKPCSamplingCorrectionManager, CorrectsIcacheOnlyBackwardToExt1) {
  CorrectionManager manager;
  manager.setEnabled(true);
  manager.publish(1, std::make_shared<CodeObjectClassification>(
                         buildSingleSymbol({{"v_add", 4},
                                            {"s_icache_inv", 4},
                                            {"s_icache_inv", 4},
                                            {"v_mov", 4}})));

  auto result = manager.correct("s_icache_inv", makeInput(1, 8));

  ASSERT_EQ(result.action, CorrectionAction::Keep);
  EXPECT_EQ(result.pc.code_object_offset, 0u);
}

TEST(RocprofSDKPCSamplingCorrectionManager, DropsAmbiguousMixedBoundary) {
  CorrectionManager manager;
  manager.setEnabled(true);
  manager.publish(
      1, std::make_shared<CodeObjectClassification>(buildSingleSymbol(
             {{"v_add", 4}, {"s_icache_inv", 4}, {"s_nop", 4}, {"v_mov", 4}})));

  auto result = manager.correct("s_nop", makeInput(1, 8));

  EXPECT_EQ(result.action, CorrectionAction::Drop);
}

TEST(RocprofSDKPCSamplingCorrectionManager,
     DropsKnownBadMissingClassification) {
  CorrectionManager manager;
  manager.setEnabled(true);

  auto result = manager.correct("s_nop", makeInput(99, 4));

  EXPECT_EQ(result.action, CorrectionAction::Drop);
}

TEST(RocprofSDKPCSamplingCorrectionManager, EraseKeepsInFlightEntriesAlive) {
  CorrectionManager manager;
  manager.publish(
      1, std::make_shared<CodeObjectClassification>(
             buildSingleSymbol({{"v_add", 4}, {"s_nop", 4}, {"v_mov", 4}})));

  auto entry = manager.lookup(1, 4);
  ASSERT_TRUE(entry.has_value());
  manager.erase(1);

  EXPECT_EQ(entry->offset, 4u);
  EXPECT_EQ(entry->window->ext2Offset, 8u);
  EXPECT_FALSE(manager.lookup(1, 4).has_value());
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
