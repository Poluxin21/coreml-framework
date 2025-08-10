#ifndef ML_CORE_H
#define ML_CORE_H
#pragma once
#include <string>
#include <vector>

namespace mlcore {
    bool init(const std::string& model_path);
    bool run(const std::vector<float>& input, std::vector<float>& output);
}
#endif //ML_CORE_H
