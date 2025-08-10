#include "../include/ml_core.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>

namespace {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mlcore");
    Ort::Session* session = nullptr;
    Ort::SessionOptions session_options;
}

namespace mlcore {
    bool init(const std::string& model_path) {
        try {
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session = new Ort::Session(env, reinterpret_cast<const wchar_t *>(model_path.c_str()), session_options);
            return true;
        } catch (const Ort::Exception& e) {
            std::cerr << "Erro ao carregar modelo: " << e.what() << std::endl;
            return false;
        }
    }

    bool run(const std::vector<float>& input, std::vector<float>& output) {
        if (!session) return false;

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session->GetInputNameAllocated(0, allocator);
        auto input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        size_t input_tensor_size = 1;
        for (auto dim : input_shape) if (dim > 0) input_tensor_size *= dim;
        if (input.size() != input_tensor_size) return false;

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(input.data()),
            input_tensor_size, input_shape.data(), input_shape.size());

        auto output_name = session->GetOutputNameAllocated(0, allocator);
        auto output_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        size_t output_tensor_size = 1;
        for (auto dim : output_shape) if (dim > 0) output_tensor_size *= dim;
        output.resize(output_tensor_size);

        auto output_tensor = Ort::Value::CreateTensor<float>(mem_info, output.data(),
            output_tensor_size, output_shape.data(), output_shape.size());

        session->Run(Ort::RunOptions{nullptr}, reinterpret_cast<const char * const *>(input_name.get()),
            &input_tensor, 1, reinterpret_cast<const char * const *>(output_name.get()),
            &output_tensor, 1);
        return true;
    }
}