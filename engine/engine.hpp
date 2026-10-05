#pragma once

#include <llama.h>

#include <atomic>
#include <functional>
#include <string>
#include <vector>

struct GenParams {
  int max_tokens = 24;
  std::vector<std::string> stop;
};

struct IngestStats {
  size_t lcp = 0;
  size_t delta = 0;
  size_t cached = 0;
  double prefill_ms = 0.0;
  bool ok = true;
};

struct GenResult {
  int n = 0; // tokens generated
  double ttft_ms = 0.0;
  bool ok = true;
};

class LlamaEngine {
public:
  ~LlamaEngine();

  bool load(const std::string &model_path, int n_ctx, int n_gpu_layers);

  IngestStats ingest(const std::string &text);

  GenResult generate(const GenParams &p,
                     const std::function<void(const std::string &)> &emit,
                     std::atomic<bool> &cancel);

  void reset();
  uint32_t n_ctx() const { return n_ctx_; }

private:
  std::vector<llama_token> tokenize(const std::string &text) const;
  bool truncate_from(llama_pos p0);

  llama_model *model_ = nullptr;
  llama_context *ctx_ = nullptr;
  llama_sampler *smpl_ = nullptr;

  std::vector<llama_token> cached_;
  size_t watermark_ = 0;
  uint32_t n_ctx_ = 0;
  size_t gen_reserve_ = 512;
  bool adds_bos_ = false;
};
