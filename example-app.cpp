#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <exception>

#include <nlohmann/json.hpp>
#include <torch/script.h>

using namespace std;
using json = nlohmann::json;

const bool DEBUG_VERBOSE = false;

json get_img_convert_to_tensor(string);
string get_json_from_file(string);

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    // Deserialize the ScriptModule from a file using torch::jit::load().
    shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

    assert(module != nullptr);
    cout << "Model loaded.\n";

    vector<torch::jit::IValue> inputs;
    //inputs.push_back(torch::ones({1, 3, 28, 28}));

    json j = get_img_convert_to_tensor(get_json_from_file("../29300.png.json"));
    //cout << j << endl;

    /* References:
        - https://nlohmann.github.io/json/classnlohmann_1_1basic__json_a16f9445f7629f634221a42b967cdcd43.html#a16f9445f7629f634221a42b967cdcd43
        - https://github.com/pytorch/pytorch/issues/14000
    */
    auto blob = j.get<vector<vector<vector<vector<float>>>>>();
    auto tensor = torch::empty(1 * 3 * 28 * 28);
    float* data = tensor.data<float>();

    for (const auto& i : blob) {
        for (const auto& j : i) {
            for (const auto& k : j) {
                for (const auto& l : k) {
                    *data++ = l;
                }
            }
        }
    }

    //cout << data << endl;

    /*at::TensorOptions options(at::ScalarType::Byte);

    at::Tensor t = torch::from_blob(data, {1, 3, 28, 28});
    t = t.toType(at::kFloat);
    */

    inputs.clear();
    inputs.emplace_back(tensor.resize_({1, 3, 28, 28}));

    // Run inference
    auto output = module->forward(inputs).toTensor();
    cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

}

string get_json_from_file(string file) {
    ifstream ifs(file);
    string line;
    string data;

    while (getline(ifs, line)) {
        data.append(line);
    }

    if (DEBUG_VERBOSE) cout << "json: " << data << endl;

    return data;
}

json get_img_convert_to_tensor(string data) {
    if (data.length() > 0) {
        json j = json::parse(data);
        return j;
    } else {
        return json::parse("[]");
    }
}
