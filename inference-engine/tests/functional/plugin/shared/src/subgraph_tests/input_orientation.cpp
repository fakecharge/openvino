// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <debug.h>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_tests/input_orientation.hpp"

namespace LayerTestsDefinitions {
    std::string InputOrientationTest::getTestCaseName(const testing::TestParamInfo<PermuteConvPermuteTuple> &obj) {
//        std::vector<std::vector<size_t>> input;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::map<std::string, std::string> additional_config;
        std::tie(netPrecision, targetName, additional_config) = obj.param;
        std::ostringstream results;

//        results << "IS=" << CommonTestUtils::vec2str(input[0]) << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void InputOrientationTest::SetUp() {
//        std::vector<std::vector<size_t>> inputs;
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additional_config;
        std::tie(netPrecision, targetDevice, additional_config) = this->GetParam();
        configuration.insert(additional_config.begin(), additional_config.end());
//        threshold = 6e-1;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, { {1, 336, 1, 1}});

//        auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

        std::vector<size_t> outFormShapes1 = { 1, 1, 168, 2 };
        std::vector<size_t> outFormShapes3 = { 1, 336, 1, 1 };
        auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, outFormShapes1);
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);


        auto permute1 = std::make_shared<ngraph::opset1::Transpose>(reshape1,
                                                                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));
        permute1->set_friendly_name("permute1");

        auto conv1 = ngraph::builder::makeConvolution(permute1, ngPrc, { 1, 1 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                                                      ngraph::op::PadType::VALID, 8);

        auto permute2 = std::make_shared<ngraph::opset1::Transpose>(conv1,
                                                                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 2, 3, 1 }));
        permute2->set_friendly_name("permute2");


        std::vector<size_t> outFormShapes2 = { 1, 1344 };
        std::vector<size_t> outFormShapes4 = { 1, 336 };

        auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes2);
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern2, false);

        auto pattern4 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes4);

        auto pattern3 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, outFormShapes3);
        auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern3, false);
        auto conv2 = ngraph::builder::makeConvolution(reshape3, ngPrc, { 1, 1 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                                                      ngraph::op::PadType::VALID, 336);
        auto reshape4 = std::make_shared<ngraph::opset1::Reshape>(conv2, pattern4, false);

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape2),
                std::make_shared<ngraph::opset1::Result>(reshape4)
                        };
        function = std::make_shared<ngraph::Function>(results, params, "SplitConvConcat");
    }

    TEST_P(InputOrientationTest, CompareWithRefs){
        Run();
    };
} // namespace LayerTestsDefinitions
