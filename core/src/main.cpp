#include "../include/ml_core.h"
#include <iostream>

int main() {
    if (!mlcore::init("models/mlp.onnx")) {
        std::cerr << "Falha ao inicializar." << std::endl;
        return 1;
    }

    std::vector<float> input = {0.5f, -0.3f, 0.8f};
    std::vector<float> output;

    if (mlcore::run(input, output)) {
        std::cout << "Saída:";
        for (auto v : output) std::cout << " " << v;
        std::cout << std::endl;
    }
}