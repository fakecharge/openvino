// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <vector>
#include "subgraph_tests/input_orientation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

    std::map<std::string, std::string> additional_config = {
//            {"GNA_COMPACT_MODE", "NO"},
            {"GNA_DEVICE_MODE", "GNA_SW_FP32"}
    };

    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16,
    };

    INSTANTIATE_TEST_CASE_P(multioutput_eltwise_identity, InputOrientationTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::Values(additional_config)),
                            InputOrientationTest::getTestCaseName);
}  // namespace
