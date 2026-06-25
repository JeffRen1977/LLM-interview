/**
 * Lab C++ · QNN-style runtime stub (API shape practice)
 *
 * Build:
 *   cd qualcomm/cpp && cmake -B build && cmake --build build && ./build/qnn_stub
 *
 * Maps to prep §3.7, §10 — real QNN uses similar C API on device.
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct QnnTensor {
    const char* name;
    float* data;
    int size;
};

struct QnnContext {
    std::vector<float> weights;
    bool loaded;
};

static int qnn_context_create(QnnContext** ctx) {
    *ctx = new QnnContext{{}, false};
    return 0;
}

static int qnn_context_load_binary(QnnContext* ctx, const char* path) {
    (void)path;
    ctx->weights = {0.5f, -0.25f, 0.1f, 0.3f};
    ctx->loaded = true;
    printf("[QNN] loaded context binary (%zu weights)\n", ctx->weights.size());
    return 0;
}

static int qnn_graph_execute(QnnContext* ctx, QnnTensor* inputs, int n_in,
                             QnnTensor* outputs, int n_out) {
    if (!ctx->loaded) return -1;
    float sum = 0.f;
    for (int i = 0; i < n_in; ++i) {
        for (int j = 0; j < inputs[i].size; ++j) {
            sum += inputs[i].data[j] * ctx->weights[j % ctx->weights.size()];
        }
    }
    outputs[0].data[0] = sum > 0 ? sum : 0;  // relu-ish
    (void)n_out;
    return 0;
}

static void qnn_context_free(QnnContext* ctx) {
    delete ctx;
}

int main() {
    QnnContext* ctx = nullptr;
    qnn_context_create(&ctx);
    qnn_context_load_binary(ctx, "model_context.bin");

    float in_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out_data[] = {0.0f};
    QnnTensor in{"input", in_data, 4};
    QnnTensor out{"output", out_data, 1};

    int rc = qnn_graph_execute(ctx, &in, 1, &out, 1);
    printf("[QNN] execute rc=%d output=%.4f\n", rc, out_data[0]);

    qnn_context_free(ctx);
    printf("C++ stub done — mirror this flow in real QNN C API samples.\n");
    return 0;
}
