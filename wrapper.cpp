#include "wrapper.h"
#include "llama.cpp/include/llama.h"
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/ggml/include/ggml-alloc.h"
#include "llama.cpp/ggml/include/ggml-backend.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/sampling.h"
#include "llama.cpp/common/speculative.h"
#include "llama.cpp/common/chat.h"
#include "llama.cpp/vendor/nlohmann/json.hpp"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "llama.cpp/ggml/include/ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// CUDA backend header for GPU info
#ifdef GGML_USE_CUDA
#include "llama.cpp/ggml/include/ggml-cuda.h"
#endif

// NVML dynamic loading for GPU utilization
#ifdef GGML_USE_CUDA
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Minimal NVML type definitions (avoids requiring nvml.h)
typedef void* nvmlDevice_t;
typedef enum { NVML_SUCCESS = 0 } nvmlReturn_t;
typedef struct { unsigned int gpu; unsigned int memory; } nvmlUtilization_t;

typedef nvmlReturn_t (*nvmlInit_t)(void);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_t)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_t)(nvmlDevice_t, nvmlUtilization_t*);

static struct {
    bool attempted = false;
    bool available = false;
    nvmlInit_t init = nullptr;
    nvmlDeviceGetHandleByIndex_t getHandle = nullptr;
    nvmlDeviceGetUtilizationRates_t getUtil = nullptr;
} g_nvml;

static void nvml_load() {
    if (g_nvml.attempted) return;
    g_nvml.attempted = true;

    void* lib = nullptr;
#ifdef _WIN32
    // Try NVML from CUDA toolkit or driver path.
    lib = (void*)LoadLibraryA("nvml.dll");
    if (!lib) {
        // Try System32 path where nvidia drivers install it.
        char path[MAX_PATH];
        GetSystemDirectoryA(path, sizeof(path));
        std::string fullpath = std::string(path) + "\\nvml.dll";
        lib = (void*)LoadLibraryA(fullpath.c_str());
    }
#else
    lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
#endif
    if (!lib) return;

#ifdef _WIN32
    #define LOADSYM(sym) (void*)GetProcAddress((HMODULE)lib, sym)
#else
    #define LOADSYM(sym) dlsym(lib, sym)
#endif

    g_nvml.init = (nvmlInit_t)LOADSYM("nvmlInit_v2");
    if (!g_nvml.init) g_nvml.init = (nvmlInit_t)LOADSYM("nvmlInit");
    g_nvml.getHandle = (nvmlDeviceGetHandleByIndex_t)LOADSYM("nvmlDeviceGetHandleByIndex_v2");
    if (!g_nvml.getHandle) g_nvml.getHandle = (nvmlDeviceGetHandleByIndex_t)LOADSYM("nvmlDeviceGetHandleByIndex");
    g_nvml.getUtil = (nvmlDeviceGetUtilizationRates_t)LOADSYM("nvmlDeviceGetUtilizationRates");

    #undef LOADSYM

    if (!g_nvml.init || !g_nvml.getHandle || !g_nvml.getUtil) return;
    if (g_nvml.init() != NVML_SUCCESS) return;
    g_nvml.available = true;
}

static int nvml_get_utilization(int device_id) {
    nvml_load();
    if (!g_nvml.available) return -1;
    nvmlDevice_t dev;
    if (g_nvml.getHandle((unsigned int)device_id, &dev) != NVML_SUCCESS) return -1;
    nvmlUtilization_t util;
    if (g_nvml.getUtil(dev, &util) != NVML_SUCCESS) return -1;
    return (int)util.gpu;
}
#endif // GGML_USE_CUDA

// Global error handling
static std::string g_last_error;

// Global log level control
static ggml_log_level g_min_log_level = GGML_LOG_LEVEL_INFO;

// Custom Go log callback (NULL = use default stderr logger)
static llama_wrapper_log_callback g_custom_log_callback = nullptr;
static void* g_custom_log_user_data = nullptr;

// Log callback that respects LLAMA_LOG environment variable
static void llama_log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level < g_min_log_level) {
        return;
    }
    if (g_custom_log_callback != nullptr) {
        g_custom_log_callback(static_cast<int>(level), text, g_custom_log_user_data);
    } else {
        fprintf(stderr, "%s", text);
    }
}

extern "C" {

// Initialise logging based on LLAMA_LOG environment variable
// Supported values: none, debug, info (default), warn, error
void llama_wrapper_init_logging() {
    const char* log_level = std::getenv("LLAMA_LOG");
    if (log_level != nullptr) {
        std::string level_str(log_level);
        if (level_str == "none") {
            g_min_log_level = GGML_LOG_LEVEL_NONE;
        } else if (level_str == "debug") {
            g_min_log_level = GGML_LOG_LEVEL_DEBUG;
        } else if (level_str == "info") {
            g_min_log_level = GGML_LOG_LEVEL_INFO;
        } else if (level_str == "warn") {
            g_min_log_level = GGML_LOG_LEVEL_WARN;
        } else if (level_str == "error") {
            g_min_log_level = GGML_LOG_LEVEL_ERROR;
        }
    }
    llama_log_set(llama_log_callback, nullptr);
}

void llama_wrapper_set_log_callback(llama_wrapper_log_callback callback, void* user_data) {
    g_custom_log_callback = callback;
    g_custom_log_user_data = user_data;
}

// Forward declarations of Go callback functions
extern bool goTokenCallback(uintptr_t handle, const char* token);
extern bool goProgressCallback(float progress, void* user_data);

// Separate wrappers for model and context
struct llama_wrapper_model_t {
    llama_model* model;
    int n_gpu_layers;  // Number of GPU layers requested (for stats reporting)
};

struct llama_wrapper_context_t {
    llama_context* ctx;
    llama_model* model;  // Reference to parent model
    std::vector<int> cached_tokens;  // Cache for prefix matching optimisation
    int last_prompt_tokens;     // Token count from last generation
    int last_completion_tokens; // Token count from last generation
    double last_prompt_eval_ms; // Prompt eval time from last generation
    double last_eval_ms;        // Token generation time from last generation
};

const char* llama_wrapper_last_error() {
    return g_last_error.c_str();
}

void llama_wrapper_free_result(char* result) {
    if (result) {
        free(result);
    }
}

// Static no-op callback for silent loading
static bool silent_progress_callback(float progress, void* user_data) {
    (void)progress;
    (void)user_data;
    return true;  // Continue loading
}

// Empty device list: a null-terminated array with no devices.
// When passed as model_params.devices, llama.cpp skips all GPU device
// enumeration entirely — the model loads purely on CPU.
static ggml_backend_dev_t cpu_only_devices[] = { nullptr };

// Convert our params to llama.cpp model params
static struct llama_model_params convert_model_params(llama_wrapper_model_params params) {
    struct llama_model_params model_params = llama_model_default_params();

    // Only set n_gpu_layers if not -1 (which means "use default/all layers")
    // llama.cpp default is -1 which effectively means all layers
    if (params.n_gpu_layers != -1) {
        model_params.n_gpu_layers = params.n_gpu_layers;
    }

    // When n_gpu_layers == 0 (CPU only), pass an empty device list to bypass
    // GPU enumeration entirely. This prevents a vector::_M_range_check crash
    // in load_tensors that occurs when CUDA devices are enumerated but no
    // layers are assigned to them.
    if (params.n_gpu_layers == 0) {
        model_params.devices = cpu_only_devices;
    } else {
        model_params.main_gpu = params.main_gpu ? atoi(params.main_gpu) : 0;
    }
    model_params.use_mmap = params.mmap;
    model_params.use_mlock = params.mlock;
    model_params.no_host = false;  // Use host buffers (b6709 added field)

    // Configure progress callback
    if (params.disable_progress_callback) {
        model_params.progress_callback = silent_progress_callback;
        model_params.progress_callback_user_data = nullptr;
    } else if (params.progress_callback) {
        model_params.progress_callback = params.progress_callback;
        model_params.progress_callback_user_data = params.progress_callback_user_data;
    }
    // Otherwise NULL → llama.cpp installs default dot printer

    return model_params;
}

// Convert our params to llama.cpp context params
static struct llama_context_params convert_context_params(llama_wrapper_model_params params) {
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.n_ctx > 0 ? params.n_ctx : 2048;
    ctx_params.n_batch = params.n_batch > 0 ? params.n_batch : 512;
    ctx_params.n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx_params.n_threads_batch = params.n_threads_batch > 0 ? params.n_threads_batch : ctx_params.n_threads;
    ctx_params.n_seq_max = params.n_parallel > 0 ? params.n_parallel : 1;
    ctx_params.embeddings = params.embeddings;
    ctx_params.no_perf = false;  // Enable perf counters for token usage timing

    // Set KV cache quantization type
    if (params.kv_cache_type != nullptr) {
        std::string cache_type(params.kv_cache_type);
        if (cache_type == "f16") {
            ctx_params.type_k = GGML_TYPE_F16;
            ctx_params.type_v = GGML_TYPE_F16;
        } else if (cache_type == "q8_0") {
            ctx_params.type_k = GGML_TYPE_Q8_0;
            ctx_params.type_v = GGML_TYPE_Q8_0;
        } else if (cache_type == "q4_0") {
            ctx_params.type_k = GGML_TYPE_Q4_0;
            ctx_params.type_v = GGML_TYPE_Q4_0;
        }
        // If unrecognized, leave as default (f16)
    }

    // Set Flash Attention mode
    if (params.flash_attn != nullptr) {
        std::string fa_mode(params.flash_attn);
        if (fa_mode == "enabled") {
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        } else if (fa_mode == "disabled") {
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        } else if (fa_mode == "auto") {
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        }
        // If unrecognized, leave as default (auto)
    }

    return ctx_params;
}

void* llama_wrapper_model_load(const char* model_path, llama_wrapper_model_params params) {
    if (!model_path) {
        g_last_error = "Model path cannot be null";
        return nullptr;
    }

    try {
        // Initialize llama backend
        llama_backend_init();

        // Load model (weights only)
        auto model_params = convert_model_params(params);
        llama_model* model = llama_model_load_from_file(model_path, model_params);
        if (!model) {
            g_last_error = "Failed to load model from: " + std::string(model_path);
            return nullptr;
        }

        // Create wrapper (model only, no context)
        auto wrapper = new llama_wrapper_model_t();
        wrapper->model = model;
        // Store n_gpu_layers for stats reporting
        // If -1 was passed (meaning "use default"), llama.cpp uses 999 layers
        wrapper->n_gpu_layers = (params.n_gpu_layers == -1) ? 999 : params.n_gpu_layers;

        return wrapper;
    } catch (const std::exception& e) {
        g_last_error = "Exception loading model: " + std::string(e.what());
        return nullptr;
    }
}

void llama_wrapper_model_free(void* model) {
    if (!model) return;

    auto wrapper = static_cast<llama_wrapper_model_t*>(model);
    if (wrapper->model) {
        llama_model_free(wrapper->model);
        wrapper->model = nullptr;  // Prevent double-free
    }
    delete wrapper;
}

void* llama_wrapper_context_create(void* model, llama_wrapper_model_params params) {
    if (!model) {
        g_last_error = "Model cannot be null";
        return nullptr;
    }

    try {
        auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);

        // Create context from model
        auto ctx_params = convert_context_params(params);
        llama_context* ctx = llama_init_from_model(model_wrapper->model, ctx_params);
        if (!ctx) {
            g_last_error = "Failed to create context";
            return nullptr;
        }

        // Create context wrapper
        auto ctx_wrapper = new llama_wrapper_context_t();
        ctx_wrapper->ctx = ctx;
        ctx_wrapper->model = model_wrapper->model;  // Keep reference to parent model

        return ctx_wrapper;
    } catch (const std::exception& e) {
        g_last_error = "Exception creating context: " + std::string(e.what());
        return nullptr;
    }
}

void llama_wrapper_context_free(void* ctx) {
    if (!ctx) return;

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    if (wrapper->ctx) {
        llama_free(wrapper->ctx);
        wrapper->ctx = nullptr;  // Prevent double-free
    }
    delete wrapper;
}

// Get model's native maximum context length from GGUF metadata
int llama_wrapper_get_model_context_length(void* model) {
    if (!model) {
        return 32768;  // Fallback if model is null
    }

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);

    // Query model's native context length from GGUF metadata
    int n_ctx_train = llama_model_n_ctx_train(model_wrapper->model);

    // Return model's training context, or reasonable fallback
    return (n_ctx_train > 0) ? n_ctx_train : 32768;
}

// Get model's embedding dimension
int llama_wrapper_model_n_embd(void* model) {
    if (!model) {
        return -1;  // Error if model is null
    }

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);
    return llama_model_n_embd(model_wrapper->model);
}

// Helper function to find common prefix length between two token vectors
static int findCommonPrefix(const std::vector<int>& a, const std::vector<int>& b) {
    int commonLen = 0;
    size_t minLen = std::min(a.size(), b.size());
    for (size_t i = 0; i < minLen; i++) {
        if (a[i] != b[i]) {
            break;
        }
        commonLen++;
    }
    return commonLen;
}

char* llama_wrapper_generate_with_tokens(void* ctx, const int* tokens, int n_tokens, int prefix_len, llama_wrapper_generate_params params) {
    if (!ctx || !tokens) {
        g_last_error = "Context and tokens cannot be null";
        return nullptr;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    if (!wrapper->ctx) {
        g_last_error = "Context has been freed";
        return nullptr;
    }

    // Reset perf counters so timing is per-request, not cumulative.
    llama_perf_context_reset(wrapper->ctx);

    try {
        // Convert C tokens to vector
        std::vector<llama_token> prompt_tokens(tokens, tokens + n_tokens);

        if (prompt_tokens.empty()) {
            g_last_error = "Token array is empty";
            return nullptr;
        }

        // Check context size with safety margin BEFORE manipulating KV cache
        int available_ctx = llama_n_ctx(wrapper->ctx);
        if (available_ctx <= 0) {
            g_last_error = "Invalid context size";
            return nullptr;
        }
        // Check if prompt fits with room for at least a few generated tokens
        int tokens_needed = (int)prompt_tokens.size() + params.max_tokens;
        if (tokens_needed > available_ctx) {
            char err_msg[256];
            snprintf(err_msg, sizeof(err_msg),
                    "Prompt too long for context size: need %d tokens (%d prompt + %d generation) but context is only %d tokens",
                    tokens_needed, (int)prompt_tokens.size(), params.max_tokens > 0 ? params.max_tokens : 128, available_ctx);
            g_last_error = err_msg;
            return nullptr;
        }
        if ((int)prompt_tokens.size() >= available_ctx - 1) {
            g_last_error = "Prompt too long for context size (need at least 1 token for generation)";
            return nullptr;
        }

        // Clear KV cache from divergence point onwards
        // For full cache hits, we'll refresh the last prompt token, so clear from prefix_len - 1
        // For partial matches, clear from prefix_len as usual
        int clear_from = (prefix_len == n_tokens && n_tokens > 0) ? prefix_len - 1 : prefix_len;
        // Only clear if clear_from is valid and within context bounds
        if (clear_from >= 0 && clear_from < available_ctx) {
            llama_memory_seq_rm(llama_get_memory(wrapper->ctx), 0, clear_from, -1);
        }

        // Create sampling parameters - use the struct directly instead of calling a function
        common_params_sampling sampling_params;
        // Basic sampling
        sampling_params.seed = params.seed;
        sampling_params.temp = params.temperature;
        sampling_params.top_k = params.top_k;
        sampling_params.top_p = params.top_p;
        sampling_params.min_p = params.min_p;
        sampling_params.typ_p = params.typ_p;
        sampling_params.top_n_sigma = params.top_n_sigma;
        sampling_params.min_keep = params.min_keep;

        // Repetition penalties
        sampling_params.penalty_last_n = params.penalty_last_n;
        sampling_params.penalty_repeat = params.penalty_repeat;
        sampling_params.penalty_freq = params.penalty_freq;
        sampling_params.penalty_present = params.penalty_present;

        // DRY sampling
        sampling_params.dry_multiplier = params.dry_multiplier;
        sampling_params.dry_base = params.dry_base;
        sampling_params.dry_allowed_length = params.dry_allowed_length;
        sampling_params.dry_penalty_last_n = params.dry_penalty_last_n;
        // Convert dry_sequence_breakers from C array to std::vector
        sampling_params.dry_sequence_breakers.clear();
        for (int i = 0; i < params.dry_sequence_breakers_count; i++) {
            sampling_params.dry_sequence_breakers.push_back(std::string(params.dry_sequence_breakers[i]));
        }

        // Dynamic temperature
        sampling_params.dynatemp_range = params.dynatemp_range;
        sampling_params.dynatemp_exponent = params.dynatemp_exponent;

        // XTC sampling
        sampling_params.xtc_probability = params.xtc_probability;
        sampling_params.xtc_threshold = params.xtc_threshold;

        // Mirostat sampling
        sampling_params.mirostat = params.mirostat;
        sampling_params.mirostat_tau = params.mirostat_tau;
        sampling_params.mirostat_eta = params.mirostat_eta;

        // Other parameters
        sampling_params.n_prev = params.n_prev;
        sampling_params.n_probs = params.n_probs;
        sampling_params.ignore_eos = params.ignore_eos;

        // Initialise sampler
        common_sampler* sampler = common_sampler_init(wrapper->model, sampling_params);
        if (!sampler) {
            g_last_error = "Failed to initialise sampler";
            return nullptr;
        }

        // Validate generation parameters
        // Reject negative max_tokens (0 is allowed and means "use default")
        if (params.max_tokens < 0) {
            common_sampler_free(sampler);
            g_last_error = "Invalid max_tokens value (must be >= 0)";
            return nullptr;
        }
        int n_predict = params.max_tokens > 0 ? params.max_tokens : 128;

        // After clearing cache from prefix_len onwards, cache ends at prefix_len - 1
        // Next position to use is prefix_len
        int n_past = prefix_len;

        // Process prompt tokens from prefix_len onwards using explicit positions
        if (prefix_len < n_tokens) {
            int tokens_to_process = n_tokens - prefix_len;
            int n_batch = llama_n_batch(wrapper->ctx);

            // Process tokens in chunks that respect n_batch limit
            for (int chunk_start = 0; chunk_start < tokens_to_process; chunk_start += n_batch) {
                int chunk_size = std::min(n_batch, tokens_to_process - chunk_start);
                llama_batch batch = llama_batch_init(chunk_size, 0, 1);
                common_batch_clear(batch);

                // Add tokens for this chunk with explicit positions
                for (int i = 0; i < chunk_size; i++) {
                    int token_idx = prefix_len + chunk_start + i;
                    int position = prefix_len + chunk_start + i;
                    // Only the very last token of the entire prompt needs logits
                    bool needs_logits = (chunk_start + i == tokens_to_process - 1);
                    common_batch_add(batch, prompt_tokens[token_idx], position, { 0 }, needs_logits);
                }

                if (llama_decode(wrapper->ctx, batch) != 0) {
                    if (params.debug) {
                        fprintf(stderr, "WARNING: prompt decode failed for chunk starting at %d\n", chunk_start);
                    }
                    llama_batch_free(batch);
                    common_sampler_free(sampler);
                    g_last_error = "Failed to decode prompt";
                    return nullptr;
                }

                llama_batch_free(batch);
            }

            n_past = n_tokens;  // Position now at end of prompt
        } else if (prefix_len == n_tokens && n_tokens > 0) {
            // Full cache hit - refresh last token's logits to ensure determinism
            // This is critical: without this, we sample from stale logits from the previous generation
            // The last prompt token is at position n_tokens - 1 (0-indexed positions)
            llama_batch batch = llama_batch_init(512, 0, 1);
            common_batch_clear(batch);
            common_batch_add(batch, prompt_tokens[n_tokens - 1], n_tokens - 1, { 0 }, true);

            if (llama_decode(wrapper->ctx, batch) != 0) {
                if (params.debug) {
                    fprintf(stderr, "WARNING: logit refresh failed\n");
                }
                llama_batch_free(batch);
                common_sampler_free(sampler);
                g_last_error = "Failed to refresh logits for cached prompt";
                return nullptr;
            }

            llama_batch_free(batch);
            n_past = n_tokens;  // Set position to end of prompt for generation
        }
        // If n_tokens == 0, nothing to decode

        // Generation loop - follows simple.cpp pattern
        std::string result;
        int n_decode = 0;

        if (params.debug) {
            fprintf(stderr, "DEBUG: Starting generation loop, n_predict=%d, n_past=%d\n", n_predict, n_past);
        }

        // Main generation loop - decode first, then sample
        for (int n_gen = 0; n_gen < n_predict; n_gen++) {
            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: First iteration, about to sample\n");
            }

            // Sample the next token (using logits from previous decode or prompt)
            llama_token new_token_id = common_sampler_sample(sampler, wrapper->ctx, -1);

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: Sampled token: %d\n", new_token_id);
            }

            // Check for EOS
            if (llama_vocab_is_eog(llama_model_get_vocab(wrapper->model), new_token_id)) {
                if (params.debug) {
                    fprintf(stderr, "INFO: End of generation token encountered\n");
                }
                break;
            }

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: About to convert token to text\n");
            }

            // Convert token to text
            std::string token_str = common_token_to_piece(wrapper->ctx, new_token_id);

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: Token text: '%s'\n", token_str.c_str());
            }

            // Call callback if provided
            if (params.callback_handle != 0) {
                if (!goTokenCallback(params.callback_handle, token_str.c_str())) {
                    if (params.debug) {
                        fprintf(stderr, "INFO: Generation stopped by callback\n");
                    }
                    break;
                }
            }

            result += token_str;

            // Check stop words
            for (int j = 0; j < params.stop_words_count; j++) {
                if (result.find(params.stop_words[j]) != std::string::npos) {
                    if (params.debug) {
                        fprintf(stderr, "INFO: Stop word found, ending generation\n");
                    }
                    goto generation_done;
                }
            }

            if (params.debug && n_gen == 0) {
                // Query actual cache state before decode
                int cache_pos = llama_memory_seq_pos_max(llama_get_memory(wrapper->ctx), 0);
                fprintf(stderr, "DEBUG: About to decode token, n_past=%d, cache_pos_max=%d\n", n_past, cache_pos);
            }

            // Decode the sampled token to get logits for next iteration
            // Allocate enough space for the batch (minimum 512 tokens as per llama.cpp examples)
            llama_batch gen_batch = llama_batch_init(512, 0, 1);
            common_batch_clear(gen_batch);
            common_batch_add(gen_batch, new_token_id, n_past, { 0 }, true);

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: Batch token=%d, pos=%d, n_tokens=%d\n", new_token_id, n_past, gen_batch.n_tokens);
            }

            // Increment position for next iteration
            n_past++;

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: Batch prepared, calling llama_decode\n");
            }

            if (llama_decode(wrapper->ctx, gen_batch) != 0) {
                if (params.debug) {
                    fprintf(stderr, "WARNING: decode failed, stopping generation\n");
                }
                llama_batch_free(gen_batch);
                break;
            }

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: Decode succeeded, freeing batch\n");
            }

            llama_batch_free(gen_batch);
            n_decode += 1;

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: First iteration complete\n");
            }
        }

generation_done:
        common_sampler_free(sampler);

        // Return allocated string (caller must free)
        char* c_result = (char*)malloc(result.length() + 1);
        if (c_result) {
            memcpy(c_result, result.c_str(), result.length());
            c_result[result.length()] = '\0';
        } else {
            g_last_error = "Failed to allocate memory for result";
        }

        // Store token counts and timing for retrieval by _ex wrappers
        wrapper->last_prompt_tokens = (int)prompt_tokens.size();
        wrapper->last_completion_tokens = n_decode;

        // Capture timing from llama.cpp perf counters
        auto perf = llama_perf_context(wrapper->ctx);
        wrapper->last_prompt_eval_ms = perf.t_p_eval_ms;
        wrapper->last_eval_ms = perf.t_eval_ms;

        return c_result;

    } catch (const std::exception& e) {
        g_last_error = "Exception during generation: " + std::string(e.what());
        return nullptr;
    }
}

// Simple wrapper that tokenises the prompt and handles prefix caching automatically
char* llama_wrapper_generate(void* ctx, llama_wrapper_generate_params params) {
    if (!ctx) {
        g_last_error = "Context cannot be null";
        return nullptr;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    if (!wrapper->ctx) {
        g_last_error = "Context has been freed";
        return nullptr;
    }

    try {
        // Tokenise the prompt
        std::vector<llama_token> prompt_tokens = common_tokenize(wrapper->ctx, params.prompt, true, true);

        if (prompt_tokens.empty()) {
            g_last_error = "Failed to tokenize prompt";
            return nullptr;
        }

        // Convert to int vector for comparison
        std::vector<int> tokens_int(prompt_tokens.begin(), prompt_tokens.end());

        // Find common prefix with cached tokens (only if prefix caching enabled)
        int prefix_len = params.enable_prefix_caching
            ? findCommonPrefix(wrapper->cached_tokens, tokens_int)
            : 0;

        // Update cache to new token sequence (only if prefix caching enabled)
        if (params.enable_prefix_caching) {
            wrapper->cached_tokens = tokens_int;
        } else {
            wrapper->cached_tokens.clear();  // Ensure cache is empty when disabled
        }

        // Call token-based generation with prefix caching
        return llama_wrapper_generate_with_tokens(ctx, tokens_int.data(), tokens_int.size(), prefix_len, params);

    } catch (const std::exception& e) {
        g_last_error = "Exception during generation: " + std::string(e.what());
        return nullptr;
    }
}

// Extended generation functions that return token usage info

void llama_wrapper_free_generate_result(llama_wrapper_generate_result* result) {
    if (result && result->text) {
        free(result->text);
        result->text = nullptr;
    }
}

llama_wrapper_generate_result llama_wrapper_generate_with_tokens_ex(void* ctx, const int* tokens, int n_tokens, int prefix_len, llama_wrapper_generate_params params) {
    llama_wrapper_generate_result res = {nullptr, 0, 0, 0.0, 0.0};
    res.text = llama_wrapper_generate_with_tokens(ctx, tokens, n_tokens, prefix_len, params);
    if (res.text && ctx) {
        auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
        res.prompt_tokens = wrapper->last_prompt_tokens;
        res.completion_tokens = wrapper->last_completion_tokens;
        res.prompt_eval_ms = wrapper->last_prompt_eval_ms;
        res.eval_ms = wrapper->last_eval_ms;
    }
    return res;
}

llama_wrapper_generate_result llama_wrapper_generate_ex(void* ctx, llama_wrapper_generate_params params) {
    llama_wrapper_generate_result res = {nullptr, 0, 0, 0.0, 0.0};
    res.text = llama_wrapper_generate(ctx, params);
    if (res.text && ctx) {
        auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
        res.prompt_tokens = wrapper->last_prompt_tokens;
        res.completion_tokens = wrapper->last_completion_tokens;
        res.prompt_eval_ms = wrapper->last_prompt_eval_ms;
        res.eval_ms = wrapper->last_eval_ms;
    }
    return res;
}

// NOTE: speculative decoding API changed upstream (common_speculative_init now takes
// common_params_speculative& instead of two llama_context*). These functions are stubbed
// out until the wrapper is updated to match the new API.
char* llama_wrapper_generate_draft_with_tokens(void* ctx_target, void* ctx_draft, const int* tokens, int n_tokens, int target_prefix_len, int draft_prefix_len, llama_wrapper_generate_params params) {
    (void)ctx_target; (void)ctx_draft; (void)tokens; (void)n_tokens;
    (void)target_prefix_len; (void)draft_prefix_len; (void)params;
    g_last_error = "Speculative decoding not available (API update pending)";
    return nullptr;
}

char* llama_wrapper_generate_draft(void* ctx_target, void* ctx_draft, llama_wrapper_generate_params params) {
    (void)ctx_target; (void)ctx_draft; (void)params;
    g_last_error = "Speculative decoding not available (API update pending)";
    return nullptr;
}

int llama_wrapper_tokenize(void* ctx, const char* text, int* tokens, int max_tokens) {
    if (!ctx || !text || !tokens) {
        g_last_error = "Invalid parameters for tokenization";
        return -1;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);

    try {
        std::vector<llama_token> token_vec = common_tokenize(wrapper->ctx, text, true, true);

        int count = std::min((int)token_vec.size(), max_tokens);
        for (int i = 0; i < count; i++) {
            tokens[i] = token_vec[i];
        }

        return count;
    } catch (const std::exception& e) {
        g_last_error = "Exception during tokenization: " + std::string(e.what());
        return -1;
    }
}

// Tokenise with dynamic allocation (C manages memory)
// Caller must free the returned tokens array with llama_wrapper_free_tokens
void llama_wrapper_tokenize_alloc(void* ctx, const char* text, int** tokens, int* count) {
    // Initialise outputs to safe defaults
    if (tokens) *tokens = nullptr;
    if (count) *count = -1;

    if (!ctx || !text || !tokens || !count) {
        g_last_error = "Invalid parameters for tokenization";
        return;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);

    try {
        // Tokenise text (no truncation)
        std::vector<llama_token> token_vec = common_tokenize(wrapper->ctx, text, true, true);

        // Allocate exact size needed
        int n_tokens = token_vec.size();
        int* allocated_tokens = (int*)malloc(n_tokens * sizeof(int));
        if (!allocated_tokens) {
            g_last_error = "Failed to allocate memory for tokens";
            return;
        }

        // Copy tokens from vector to allocated array
        for (int i = 0; i < n_tokens; i++) {
            allocated_tokens[i] = token_vec[i];
        }

        // Return pointer and count
        *tokens = allocated_tokens;
        *count = n_tokens;

    } catch (const std::exception& e) {
        g_last_error = "Exception during tokenization: " + std::string(e.what());
        if (tokens && *tokens) {
            free(*tokens);
            *tokens = nullptr;
        }
        if (count) *count = -1;
    }
}

// Free tokens allocated by llama_wrapper_tokenize_alloc
void llama_wrapper_free_tokens(int* tokens) {
    if (tokens) {
        free(tokens);
    }
}

int llama_wrapper_embeddings(void* ctx, const char* text, float* embeddings, int max_embeddings) {
    if (!ctx || !text || !embeddings) {
        g_last_error = "Invalid parameters for embeddings";
        return -1;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);

    try {
        // Clear KV cache to ensure clean state
        llama_memory_seq_rm(llama_get_memory(wrapper->ctx), 0, -1, -1);

        // Tokenize text
        std::vector<llama_token> tokens = common_tokenize(wrapper->ctx, text, true, true);

        if (tokens.empty()) {
            g_last_error = "Failed to tokenize text for embeddings";
            return -1;
        }

        // Evaluate tokens in chunks that respect n_batch limit
        int n_batch = llama_n_batch(wrapper->ctx);
        int n_tokens = tokens.size();

        for (int i = 0; i < n_tokens; i += n_batch) {
            int chunk_size = std::min(n_batch, n_tokens - i);
            llama_batch batch = llama_batch_init(chunk_size, 0, 1);
            common_batch_clear(batch);

            // Add tokens for this chunk
            for (int j = 0; j < chunk_size; j++) {
                // All tokens need logits for embeddings
                common_batch_add(batch, tokens[i + j], i + j, { 0 }, true);
            }

            if (llama_decode(wrapper->ctx, batch) != 0) {
                llama_batch_free(batch);
                g_last_error = "Failed to decode tokens for embeddings";
                return -1;
            }

            llama_batch_free(batch);
        }

        // Get embeddings from sequence 0 (works for both single and multi-sequence contexts)
        const float* embd = llama_get_embeddings_seq(wrapper->ctx, 0);
        if (!embd) {
            g_last_error = "Failed to get embeddings from context";
            return -1;
        }

        // Copy embeddings
        int n_embd = llama_model_n_embd(wrapper->model);
        int count = std::min(n_embd, max_embeddings);

        memcpy(embeddings, embd, count * sizeof(float));

        return count;
    } catch (const std::exception& e) {
        g_last_error = "Exception during embedding generation: " + std::string(e.what());
        return -1;
    }
}

int llama_wrapper_embeddings_batch(void* ctx, const char** texts, int n_texts, float* embeddings, int n_embd) {
    if (!ctx || !texts || !embeddings || n_texts <= 0 || n_embd <= 0) {
        g_last_error = "Invalid parameters for batch embeddings";
        return -1;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);

    try {
        // Clear KV cache to ensure clean state
        llama_memory_clear(llama_get_memory(wrapper->ctx), true);

        // Tokenize all texts
        std::vector<std::vector<llama_token>> all_tokens;
        all_tokens.reserve(n_texts);

        for (int i = 0; i < n_texts; i++) {
            if (!texts[i]) {
                g_last_error = "Null text in batch at index " + std::to_string(i);
                return -1;
            }
            std::vector<llama_token> tokens = common_tokenize(wrapper->ctx, texts[i], true, true);
            if (tokens.empty()) {
                g_last_error = "Failed to tokenize text at index " + std::to_string(i);
                return -1;
            }
            all_tokens.push_back(std::move(tokens));
        }

        // Get batch size and max sequences
        int n_batch = llama_n_batch(wrapper->ctx);
        int n_seq_max = llama_n_seq_max(wrapper->ctx);

        // Initialize batch
        llama_batch batch = llama_batch_init(n_batch, 0, n_seq_max);

        int embeddings_stored = 0;  // Track how many embeddings we've extracted

        // Process texts in batches
        int s = 0;  // Current sequence ID in batch
        for (int k = 0; k < n_texts; k++) {
            const auto& tokens = all_tokens[k];
            int n_tokens = tokens.size();

            // Check if adding this text would exceed batch size or sequence limit
            if (batch.n_tokens + n_tokens > n_batch || s >= n_seq_max) {
                // Decode current batch
                if (llama_decode(wrapper->ctx, batch) != 0) {
                    llama_batch_free(batch);
                    g_last_error = "Failed to decode batch";
                    return -1;
                }

                // Extract embeddings for all sequences in this batch
                for (int seq = 0; seq < s; seq++) {
                    const float* embd = llama_get_embeddings_seq(wrapper->ctx, seq);
                    if (!embd) {
                        llama_batch_free(batch);
                        g_last_error = "Failed to get embeddings for sequence " + std::to_string(seq);
                        return -1;
                    }
                    // Copy embedding to output buffer
                    memcpy(embeddings + embeddings_stored * n_embd, embd, n_embd * sizeof(float));
                    embeddings_stored++;
                }

                // Clear KV cache for processed sequences before resetting
                for (int seq = 0; seq < s; seq++) {
                    llama_memory_seq_rm(llama_get_memory(wrapper->ctx), seq, -1, -1);
                }

                // Reset for next batch
                s = 0;
                common_batch_clear(batch);
            }

            // Add tokens for this text with unique seq_id
            for (int j = 0; j < n_tokens; j++) {
                // Position is relative to this sequence (starts at 0)
                // All tokens need logits for embeddings
                common_batch_add(batch, tokens[j], j, { s }, true);
            }

            s++;  // Move to next sequence ID
        }

        // Process final batch if there are remaining sequences
        if (s > 0) {
            if (llama_decode(wrapper->ctx, batch) != 0) {
                llama_batch_free(batch);
                g_last_error = "Failed to decode final batch";
                return -1;
            }

            // Extract embeddings for remaining sequences
            for (int seq = 0; seq < s; seq++) {
                const float* embd = llama_get_embeddings_seq(wrapper->ctx, seq);
                if (!embd) {
                    llama_batch_free(batch);
                    g_last_error = "Failed to get embeddings for final sequence " + std::to_string(seq);
                    return -1;
                }
                memcpy(embeddings + embeddings_stored * n_embd, embd, n_embd * sizeof(float));
                embeddings_stored++;
            }
        }

        llama_batch_free(batch);

        // Verify we got all embeddings
        if (embeddings_stored != n_texts) {
            g_last_error = "Embedding count mismatch: expected " + std::to_string(n_texts) +
                          ", got " + std::to_string(embeddings_stored);
            return -1;
        }

        return embeddings_stored;

    } catch (const std::exception& e) {
        g_last_error = "Exception during batch embedding generation: " + std::string(e.what());
        return -1;
    }
}

int llama_wrapper_get_cached_token_count(void* ctx) {
    if (!ctx) {
        g_last_error = "Context cannot be null";
        return -1;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    return static_cast<int>(wrapper->cached_tokens.size());
}

// Get the chat template from model metadata
// Returns nullptr if no template is available
const char* llama_wrapper_get_chat_template(void* model) {
    if (!model) {
        return nullptr;
    }

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);

    // Get default chat template (name = nullptr)
    const char* tmpl = llama_model_chat_template(model_wrapper->model, nullptr);

    return tmpl;  // May be nullptr if model has no template
}

// Apply chat template to messages
// Returns allocated string with formatted prompt (caller must free with llama_wrapper_free_result)
// Returns nullptr on error
char* llama_wrapper_apply_chat_template(const char* tmpl, const char** roles, const char** contents, int n_messages, bool add_assistant) {
    if (!tmpl || !roles || !contents || n_messages < 0) {
        g_last_error = "Invalid parameters for chat template application";
        return nullptr;
    }

    try {
        // Build array of llama_chat_message structs
        std::vector<llama_chat_message> messages;
        messages.reserve(n_messages);

        for (int i = 0; i < n_messages; i++) {
            if (!roles[i] || !contents[i]) {
                g_last_error = "Role or content cannot be null";
                return nullptr;
            }
            messages.push_back({roles[i], contents[i]});
        }

        // Start with a reasonable buffer size (8KB)
        std::vector<char> buffer(8192);

        // Try to apply template
        int32_t result_len = llama_chat_apply_template(
            tmpl,
            messages.data(),
            n_messages,
            add_assistant,
            buffer.data(),
            buffer.size()
        );

        // If buffer was too small, resize and retry
        if (result_len > (int32_t)buffer.size()) {
            buffer.resize(result_len);
            result_len = llama_chat_apply_template(
                tmpl,
                messages.data(),
                n_messages,
                add_assistant,
                buffer.data(),
                buffer.size()
            );
        }

        // Check for errors
        if (result_len < 0) {
            g_last_error = "Failed to apply chat template (template detection or application error)";
            return nullptr;
        }

        // Allocate result and copy
        char* c_result = (char*)malloc(result_len + 1);
        if (c_result) {
            memcpy(c_result, buffer.data(), result_len);
            c_result[result_len] = '\0';
        } else {
            g_last_error = "Failed to allocate memory for chat template result";
            return nullptr;
        }

        return c_result;
    } catch (const std::exception& e) {
        g_last_error = "Exception during chat template application: " + std::string(e.what());
        return nullptr;
    }
}

// Parse model output to extract reasoning/thinking content
// Returns NULL on error. Free result with llama_wrapper_free_parsed_message()
llama_wrapper_parsed_message* llama_wrapper_parse_reasoning(
    const char* text,
    bool is_partial,
    llama_wrapper_reasoning_format format,
    int chat_format
) {
    if (!text) {
        g_last_error = "Text cannot be null for reasoning parsing";
        return nullptr;
    }

    try {
        // Configure syntax for parsing
        common_chat_parser_params syntax;
        syntax.format = static_cast<common_chat_format>(chat_format);
        syntax.reasoning_format = static_cast<common_reasoning_format>(format);
        syntax.reasoning_in_content = false;  // Extract to separate field for streaming
        syntax.thinking_forced_open = false;
        syntax.parse_tool_calls = false;  // Don't need tool parsing for this use case

        // Parse the text
        common_chat_msg msg = common_chat_parse(std::string(text), is_partial, syntax);

        // Allocate result struct
        auto* result = new llama_wrapper_parsed_message;
        result->content = strdup(msg.content.c_str());
        result->reasoning_content = msg.reasoning_content.empty()
            ? nullptr
            : strdup(msg.reasoning_content.c_str());

        return result;
    } catch (const std::exception& e) {
        g_last_error = "Exception during reasoning parsing: " + std::string(e.what());
        return nullptr;
    }
}

void llama_wrapper_free_parsed_message(llama_wrapper_parsed_message* msg) {
    if (!msg) return;

    if (msg->content) {
        free(const_cast<char*>(msg->content));
    }
    if (msg->reasoning_content) {
        free(const_cast<char*>(msg->reasoning_content));
    }
    delete msg;
}

void* llama_wrapper_chat_templates_init(void* model, const char* template_override) {
    if (!model) return nullptr;

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);
    std::string tmpl_override = template_override ? template_override : "";

    auto templates = common_chat_templates_init(model_wrapper->model, tmpl_override);
    return templates.release();  // Transfer ownership
}

void llama_wrapper_chat_templates_free(void* templates) {
    if (!templates) return;
    common_chat_templates_free(static_cast<common_chat_templates*>(templates));
}

int llama_wrapper_chat_templates_get_format(void* templates) {
    if (!templates) return 0;  // COMMON_CHAT_FORMAT_CONTENT_ONLY = 0

    auto tmpl = static_cast<common_chat_templates*>(templates);

    try {
        // Apply with minimal dummy messages just to trigger format detection
        common_chat_templates_inputs inputs;
        inputs.use_jinja = true;
        inputs.add_generation_prompt = true;

        // Create a minimal dummy message to satisfy template application
        common_chat_msg dummy_msg;
        dummy_msg.role = "user";
        dummy_msg.content = "test";  // Non-empty to avoid potential issues
        inputs.messages.push_back(dummy_msg);

        auto params = common_chat_templates_apply(tmpl, inputs);
        return static_cast<int>(params.format);
    } catch (const std::exception& e) {
        // If template application fails, return CONTENT_ONLY as fallback
        g_last_error = "Format detection failed: " + std::string(e.what());
        return 0;  // COMMON_CHAT_FORMAT_CONTENT_ONLY
    }
}

// Get model metadata string value by key
const char* llama_wrapper_model_meta_string(void* model, const char* key) {
    if (!model || !key) return nullptr;

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);

    // Use llama.cpp's metadata API with buffer
    static char buffer[2048];  // Static buffer for metadata strings
    int32_t result = llama_model_meta_val_str(model_wrapper->model, key, buffer, sizeof(buffer));

    if (result < 0) {
        return nullptr;  // Key doesn't exist
    }

    return buffer;
}

// Get count of metadata key-value pairs
int llama_wrapper_model_meta_count(void* model) {
    if (!model) return 0;

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);
    return llama_model_meta_count(model_wrapper->model);
}

// Get number of CUDA devices
int llama_wrapper_get_gpu_count() {
#ifdef GGML_USE_CUDA
    return ggml_backend_cuda_get_device_count();
#else
    return 0;
#endif
}

// Get GPU device information
bool llama_wrapper_get_gpu_info(int device_id, llama_wrapper_gpu_info* info) {
    if (!info) return false;

#ifdef GGML_USE_CUDA
    int count = ggml_backend_cuda_get_device_count();
    if (device_id < 0 || device_id >= count) return false;

    // Get device description
    ggml_backend_cuda_get_device_description(device_id, info->device_name, sizeof(info->device_name));
    info->device_id = device_id;

    // Get memory info
    size_t free_mem, total_mem;
    ggml_backend_cuda_get_device_memory(device_id, &free_mem, &total_mem);
    info->free_memory_mb = free_mem / (1024 * 1024);
    info->total_memory_mb = total_mem / (1024 * 1024);

    // Get GPU utilization via NVML (dynamic load)
    info->utilization_pct = nvml_get_utilization(device_id);

    return true;
#else
    return false;
#endif
}

// Get runtime information about model and context
void llama_wrapper_get_runtime_info(void* model, void* ctx, const char* kv_cache_type, llama_wrapper_runtime_info* info) {
    if (!model || !info) return;

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);

    // Get layer counts (llama.cpp uses singular "layer" not "layers")
    info->total_layers = llama_model_n_layer(model_wrapper->model);
    // GPU layers loaded is minimum of requested and total layers
    // (can't load more layers than the model has)
    info->gpu_layers = std::min(model_wrapper->n_gpu_layers, info->total_layers);

    if (ctx) {
        auto ctx_wrapper = static_cast<llama_wrapper_context_t*>(ctx);
        info->n_ctx = llama_n_ctx(ctx_wrapper->ctx);
        info->n_batch = llama_n_batch(ctx_wrapper->ctx);

        // Calculate KV cache size properly accounting for GQA/MQA
        // Formula: 2 * n_ctx * (head_dim * n_head_kv) * n_layers * bytes_per_element
        int n_embd = llama_model_n_embd(model_wrapper->model);
        int n_head = llama_model_n_head(model_wrapper->model);
        int n_head_kv = llama_model_n_head_kv(model_wrapper->model);
        int head_dim = n_embd / n_head;

        // Determine element size based on quantization type
        float bytes_per_element = 2.0f;  // Default f16

        if (kv_cache_type) {
            std::string cache_type(kv_cache_type);
            if (cache_type == "f16") {
                bytes_per_element = 2.0f;
            } else if (cache_type == "q8_0") {
                bytes_per_element = 1.125f;  // ~1 byte + overhead
            } else if (cache_type == "q4_0") {
                bytes_per_element = 0.625f;  // ~0.5 bytes + overhead
            }
        }

        // K and V cache: n_ctx * head_dim * n_head_kv * 2 (K+V) * n_layers * element_size
        long long cache_bytes = (long long)info->n_ctx * head_dim * n_head_kv * 2LL * info->total_layers * bytes_per_element;
        info->kv_cache_size_mb = cache_bytes / (1024 * 1024);
    } else {
        // No context - use defaults or zeros
        info->n_ctx = 0;
        info->n_batch = 0;
        info->kv_cache_size_mb = 0;
    }
}

// --- Vision/multimodal support ---

void* llama_wrapper_mtmd_init(void* model, const char* mmproj_path,
    bool use_gpu, int n_threads, const char* flash_attn) {
    if (!model || !mmproj_path) {
        g_last_error = "Model and mmproj_path cannot be null";
        return nullptr;
    }

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu     = use_gpu;
    mparams.n_threads   = n_threads;
    mparams.warmup      = true;

    if (flash_attn) {
        std::string fa(flash_attn);
        if (fa == "enabled") {
            mparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        } else if (fa == "disabled") {
            mparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
    }

    // Force print_timings so we get encoding time in logs
    mparams.print_timings = true;

    // Route mtmd/clip logs through our log callback so they appear in the
    // application log instead of only going to stderr.
    mtmd_helper_log_set(llama_log_callback, nullptr);

    mtmd_context* ctx = mtmd_init_from_file(mmproj_path, model_wrapper->model, mparams);
    if (!ctx) {
        g_last_error = std::string("Failed to load mmproj from ") + mmproj_path;
        return nullptr;
    }

    return ctx;
}

void llama_wrapper_mtmd_free(void* mtmd_ctx) {
    if (mtmd_ctx) {
        mtmd_free(static_cast<mtmd_context*>(mtmd_ctx));
    }
}

void* llama_wrapper_mtmd_bitmap_from_buf(void* mtmd_ctx,
    const unsigned char* buf, int len) {
    if (!mtmd_ctx || !buf || len <= 0) {
        g_last_error = "Invalid arguments for bitmap creation";
        return nullptr;
    }

    auto ctx = static_cast<mtmd_context*>(mtmd_ctx);
    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_buf(ctx, buf, (size_t)len);
    if (!bmp) {
        g_last_error = "Failed to create bitmap from buffer";
        return nullptr;
    }

    return bmp;
}

void llama_wrapper_mtmd_bitmap_free(void* bitmap) {
    if (bitmap) {
        mtmd_bitmap_free(static_cast<mtmd_bitmap*>(bitmap));
    }
}

char* llama_wrapper_vision_generate(void* ctx, void* mtmd_ctx,
    const char* text, void** bitmaps, int n_bitmaps,
    llama_wrapper_generate_params params) {
    if (!ctx || !mtmd_ctx || !text) {
        g_last_error = "Context, mtmd_context, and text cannot be null";
        return nullptr;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    if (!wrapper->ctx) {
        g_last_error = "Context has been freed";
        return nullptr;
    }

    auto vision_ctx = static_cast<mtmd_context*>(mtmd_ctx);

    try {
        // Clear entire KV cache (vision requests don't use prefix caching)
        llama_memory_clear(llama_get_memory(wrapper->ctx), true);
        wrapper->cached_tokens.clear();

        // Prepare text input
        mtmd_input_text input_text;
        input_text.text          = text;
        input_text.add_special   = true;
        input_text.parse_special = true;

        // Prepare bitmap pointer array
        std::vector<const mtmd_bitmap*> bitmap_ptrs((size_t)n_bitmaps);
        for (int i = 0; i < n_bitmaps; i++) {
            bitmap_ptrs[i] = static_cast<const mtmd_bitmap*>(bitmaps[i]);
        }

        // Tokenize text + images into chunks
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        int32_t tok_res = mtmd_tokenize(vision_ctx, chunks, &input_text,
            bitmap_ptrs.data(), (size_t)n_bitmaps);
        if (tok_res != 0) {
            mtmd_input_chunks_free(chunks);
            char err_msg[128];
            snprintf(err_msg, sizeof(err_msg),
                "Failed to tokenize vision input (error %d)", tok_res);
            g_last_error = err_msg;
            return nullptr;
        }

        {
            char buf[128];
            snprintf(buf, sizeof(buf), "vision: tokenized %zu chunks, n_images=%d",
                mtmd_input_chunks_size(chunks), n_bitmaps);
            llama_log_callback(GGML_LOG_LEVEL_INFO, buf, nullptr);
        }

        // Eval all chunks into KV cache
        llama_pos n_past = 0;
        llama_pos new_n_past = 0;
        int32_t eval_res = mtmd_helper_eval_chunks(vision_ctx, wrapper->ctx,
            chunks, n_past, 0, llama_n_batch(wrapper->ctx), true, &new_n_past);

        mtmd_input_chunks_free(chunks);

        if (eval_res != 0) {
            g_last_error = "Failed to evaluate vision chunks";
            return nullptr;
        }

        // Check context space for generation
        int available_ctx = llama_n_ctx(wrapper->ctx);
        int n_predict = params.max_tokens > 0 ? params.max_tokens : 128;
        if ((int)new_n_past + n_predict > available_ctx) {
            n_predict = available_ctx - (int)new_n_past;
            if (n_predict <= 0) {
                g_last_error = "No room for generation after vision input";
                return nullptr;
            }
        }

        // Set up sampler (same as text-only path)
        common_params_sampling sampling_params;
        sampling_params.seed               = params.seed;
        sampling_params.temp               = params.temperature;
        sampling_params.top_k              = params.top_k;
        sampling_params.top_p              = params.top_p;
        sampling_params.min_p              = params.min_p;
        sampling_params.typ_p              = params.typ_p;
        sampling_params.top_n_sigma        = params.top_n_sigma;
        sampling_params.min_keep           = params.min_keep;
        sampling_params.penalty_last_n     = params.penalty_last_n;
        sampling_params.penalty_repeat     = params.penalty_repeat;
        sampling_params.penalty_freq       = params.penalty_freq;
        sampling_params.penalty_present    = params.penalty_present;
        sampling_params.dry_multiplier     = params.dry_multiplier;
        sampling_params.dry_base           = params.dry_base;
        sampling_params.dry_allowed_length = params.dry_allowed_length;
        sampling_params.dry_penalty_last_n = params.dry_penalty_last_n;
        sampling_params.dry_sequence_breakers.clear();
        for (int i = 0; i < params.dry_sequence_breakers_count; i++) {
            sampling_params.dry_sequence_breakers.push_back(std::string(params.dry_sequence_breakers[i]));
        }
        sampling_params.dynatemp_range    = params.dynatemp_range;
        sampling_params.dynatemp_exponent = params.dynatemp_exponent;
        sampling_params.xtc_probability   = params.xtc_probability;
        sampling_params.xtc_threshold     = params.xtc_threshold;
        sampling_params.mirostat          = params.mirostat;
        sampling_params.mirostat_tau      = params.mirostat_tau;
        sampling_params.mirostat_eta      = params.mirostat_eta;
        sampling_params.n_prev            = params.n_prev;
        sampling_params.n_probs           = params.n_probs;
        sampling_params.ignore_eos        = params.ignore_eos;

        common_sampler* sampler = common_sampler_init(wrapper->model, sampling_params);
        if (!sampler) {
            g_last_error = "Failed to initialise sampler for vision generation";
            return nullptr;
        }

        {
            char buf[256];
            snprintf(buf, sizeof(buf),
                "vision: sampling temp=%.3f top_k=%d top_p=%.3f min_p=%.3f max_tokens=%d n_past=%d",
                sampling_params.temp, sampling_params.top_k, sampling_params.top_p,
                sampling_params.min_p, n_predict, (int)new_n_past);
            llama_log_callback(GGML_LOG_LEVEL_INFO, buf, nullptr);
        }

        // Generation loop
        std::string result;
        llama_pos gen_past = new_n_past;

        // Convert stop words for checking
        std::vector<std::string> stop_words;
        for (int i = 0; i < params.stop_words_count; i++) {
            stop_words.push_back(std::string(params.stop_words[i]));
        }

        for (int n_gen = 0; n_gen < n_predict; n_gen++) {
            llama_token new_token_id = common_sampler_sample(sampler, wrapper->ctx, -1);

            if (llama_vocab_is_eog(llama_model_get_vocab(wrapper->model), new_token_id)) {
                break;
            }

            std::string token_text = common_token_to_piece(wrapper->ctx, new_token_id);
            result += token_text;

            // Streaming callback
            if (params.callback_handle != 0) {
                if (!goTokenCallback(params.callback_handle, token_text.c_str())) {
                    break; // Caller requested stop
                }
            }

            // Check stop words
            bool should_stop = false;
            for (const auto& sw : stop_words) {
                if (result.length() >= sw.length() &&
                    result.compare(result.length() - sw.length(), sw.length(), sw) == 0) {
                    result = result.substr(0, result.length() - sw.length());
                    should_stop = true;
                    break;
                }
            }
            if (should_stop) break;

            // Accept token into sampler (matches reference implementation)
            common_sampler_accept(sampler, new_token_id, true);

            // Decode the new token for next iteration
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_clear(batch);
            common_batch_add(batch, new_token_id, gen_past, {0}, true);
            gen_past++;

            if (llama_decode(wrapper->ctx, batch) != 0) {
                llama_batch_free(batch);
                common_sampler_free(sampler);
                g_last_error = "Failed to decode token during vision generation";
                return nullptr;
            }
            llama_batch_free(batch);
        }

        common_sampler_free(sampler);

        // Return result
        char* c_result = (char*)malloc(result.length() + 1);
        if (c_result) {
            memcpy(c_result, result.c_str(), result.length() + 1);
        }
        return c_result;

    } catch (const std::exception& e) {
        g_last_error = std::string("Vision generation error: ") + e.what();
        return nullptr;
    }
}

// --- GPU benchmark ---

bool llama_wrapper_bench_gpu(int device_id, llama_wrapper_bench_result* result) {
    if (!result) return false;

#ifdef GGML_USE_CUDA
    int count = ggml_backend_cuda_get_device_count();
    if (device_id < 0 || device_id >= count) return false;

    ggml_backend_t backend = ggml_backend_cuda_init(device_id);
    if (!backend) return false;

    // --- Bandwidth benchmark: delegate to shared ggml_add benchmark ---
    {
        double bw = 0;
        // Note: we can't call llama_wrapper_bench_bandwidth here because the
        // backend is already initialized above. Instead, we temporarily free it,
        // call the shared function, then re-init for the compute benchmark.
        ggml_backend_free(backend);
        if (llama_wrapper_bench_bandwidth(1, device_id, 0, &bw)) {
            result->bandwidth_gbs = bw;
        } else {
            result->bandwidth_gbs = 0;
        }
        backend = ggml_backend_cuda_init(device_id);
        if (!backend) return false;
    }

    // --- Compute benchmark: FP32 matrix multiply ---
    {
        const int M = 512;
        const int N = 512;
        const int K = 512;

        // ggml_mul_mat: result[M,N] = A[K,M]^T * B[K,N]
        // Memory needed: 3 tensors + graph overhead.
        size_t mem_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
        struct ggml_init_params ctx_params = {
            /* .mem_size   = */ mem_size,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        struct ggml_context* ctx = ggml_init(ctx_params);

        struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);

        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, c);

        // Allocate tensors on GPU.
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buf) {
            // Fill A, B with data on host then upload.
            size_t a_size = ggml_nbytes(a);
            size_t b_size = ggml_nbytes(b);
            std::vector<float> host_a(a_size / sizeof(float), 1.0f);
            std::vector<float> host_b(b_size / sizeof(float), 1.0f);
            ggml_backend_tensor_set(a, host_a.data(), 0, a_size);
            ggml_backend_tensor_set(b, host_b.data(), 0, b_size);
            ggml_backend_synchronize(backend);

            // Warmup.
            for (int i = 0; i < 5; i++) {
                ggml_backend_graph_compute(backend, graph);
                ggml_backend_synchronize(backend);
            }

            // Timed runs, 21 iterations with median for stability.
            const int iters = 21;
            double samples[21];
            double flops_per_iter = 2.0 * M * N * K;
            for (int i = 0; i < iters; i++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                ggml_backend_graph_compute(backend, graph);
                ggml_backend_synchronize(backend);
                auto t1 = std::chrono::high_resolution_clock::now();
                double secs = std::chrono::duration<double>(t1 - t0).count();
                samples[i] = (secs > 0) ? flops_per_iter / secs / 1e9 : 0;
            }
            std::sort(samples, samples + iters);
            result->gflops = samples[iters / 2]; // median

            ggml_backend_buffer_free(buf);
        } else {
            result->gflops = 0;
        }

        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    return true;
#else
    (void)device_id;
    return false;
#endif
}

// --- Memory bandwidth benchmark (device-local ggml_add, works on CPU and GPU) ---

bool llama_wrapper_bench_bandwidth(int device_type, int device_id, int n_threads, double* bandwidth_gbs) {
    if (!bandwidth_gbs) return false;
    *bandwidth_gbs = 0;

    // ggml_add reads 2 tensors and writes 1 → 3× tensor_size bytes through memory.
    const int64_t n_elements = 64 * 1024 * 1024; // 64M floats = 256 MB per tensor

    // Initialize backend.
    ggml_backend_t backend = nullptr;
    if (device_type == 0) {
        backend = ggml_backend_cpu_init();
        if (backend && n_threads > 0) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }
    } else {
#ifdef GGML_USE_CUDA
        int count = ggml_backend_cuda_get_device_count();
        if (device_id < 0 || device_id >= count) return false;
        backend = ggml_backend_cuda_init(device_id);
#else
        (void)device_id;
        return false;
#endif
    }
    if (!backend) return false;

    size_t mem_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
    struct ggml_init_params ctx_params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context* ctx = ggml_init(ctx_params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    struct ggml_tensor* c = ggml_add(ctx, a, b);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Fill tensors.
    size_t tensor_bytes = ggml_nbytes(a);
    std::vector<float> host_data(n_elements, 1.0f);
    ggml_backend_tensor_set(a, host_data.data(), 0, tensor_bytes);
    ggml_backend_tensor_set(b, host_data.data(), 0, tensor_bytes);
    ggml_backend_synchronize(backend);

    // Warmup.
    for (int i = 0; i < 5; i++) {
        ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
    }

    // Timed runs: 2 reads + 1 write = 3× tensor_bytes per iteration.
    // 21 iterations with median for stable results.
    const int iters = 21;
    double samples[21];
    double bytes_per_iter = 3.0 * (double)tensor_bytes;
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        samples[i] = (secs > 0) ? bytes_per_iter / secs / 1e9 : 0;
    }
    std::sort(samples, samples + iters);
    *bandwidth_gbs = samples[iters / 2]; // median

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

// --- Q4_K matrix-vector benchmark (comparable across CPU/GPU) ---

bool llama_wrapper_bench_q4k_matvec(int device_type, int device_id, int n_threads, double* score_gbs) {
    if (!score_gbs) return false;
    *score_gbs = 0;

    // Matrix dimensions: simulates realistic LLM decode (single token, large layer).
    const int64_t M = 8192; // rows of weight matrix
    const int64_t K = 8192; // columns of weight matrix
    const int64_t N = 1;    // single token — matches autoregressive decode

    // Initialize backend.
    ggml_backend_t backend = nullptr;
    if (device_type == 0) {
        // CPU backend.
        backend = ggml_backend_cpu_init();
        if (backend && n_threads > 0) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }
    } else {
#ifdef GGML_USE_CUDA
        int count = ggml_backend_cuda_get_device_count();
        if (device_id < 0 || device_id >= count) return false;
        backend = ggml_backend_cuda_init(device_id);
#else
        (void)device_id;
        return false;
#endif
    }
    if (!backend) return false;

    // Build a graph with multiple chained matmuls to simulate an LLM forward
    // pass (multiple layers). This keeps the GPU busy within a single
    // graph_compute call, avoiding dispatch overhead between layers.
    const int n_layers = 32; // simulate 32 transformer layers

    // Context needs room for: n_layers weight tensors + n_layers+1 activation
    // tensors + graph overhead.
    size_t mem_size = ggml_tensor_overhead() * (2 * n_layers + 2) + ggml_graph_overhead();
    struct ggml_init_params ctx_params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context* ctx = ggml_init(ctx_params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    // Build chain: x0 → W0*x0 → W1*x1 → ... → W31*x31
    // All weight matrices share the same dimensions (M=K so output feeds next input).
    struct ggml_tensor* weights[32];
    struct ggml_tensor* x0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor* x = x0;
    for (int i = 0; i < n_layers; i++) {
        weights[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, M);
        x = ggml_mul_mat(ctx, weights[i], x);
    }

    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, 2 * n_layers + 1, false);
    ggml_build_forward_expand(graph, x);

    // Allocate tensors on the backend.
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Fill weight matrices with quantized data.
    {
        size_t n_elements = M * K;
        std::vector<float> src(n_elements);
        for (size_t i = 0; i < n_elements; i++) {
            src[i] = 0.5f - (float)(i % 1000) / 1000.0f;
        }
        size_t w_size = ggml_nbytes(weights[0]);
        std::vector<uint8_t> q4_data(w_size);
        ggml_quantize_chunk(GGML_TYPE_Q4_K, src.data(), q4_data.data(), 0, M, K, nullptr);
        for (int i = 0; i < n_layers; i++) {
            ggml_backend_tensor_set(weights[i], q4_data.data(), 0, w_size);
        }
    }

    // Fill input activation with ones.
    {
        size_t b_elements = K * N;
        std::vector<float> host_x(b_elements, 1.0f);
        ggml_backend_tensor_set(x0, host_x.data(), 0, ggml_nbytes(x0));
    }

    ggml_backend_synchronize(backend);

    // Warmup — run enough to reach thermal steady state.
    for (int i = 0; i < 20; i++) {
        ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
    }

    // Timed runs — batch multiple graph_computes per sample for stable timing.
    // Each graph_compute = n_layers matmuls.
    const int reps_per_sample = 10;
    const int n_samples = 21;
    double flops_per_graph = (double)n_layers * 2.0 * M * K * N;
    double total_flops = flops_per_graph * reps_per_sample;
    double samples[21];
    for (int i = 0; i < n_samples; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps_per_sample; r++) {
            ggml_backend_graph_compute(backend, graph);
        }
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        samples[i] = (secs > 0) ? total_flops / secs / 1e9 : 0; // GFLOPS
    }
    std::sort(samples, samples + n_samples);
    *score_gbs = samples[n_samples / 2]; // GFLOPS

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

} // extern "C"
