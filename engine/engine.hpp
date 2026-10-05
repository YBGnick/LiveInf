#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "llama.h"

struct IngestStats {
  bool ok = true;
  size_t lcp = 0;
  size_t delta = 0;
  size_t cached = 0;
  double prefill_ms = 0.0;
};

struct GenParams {
  int max_tokens = 24;
  std::vector<std::string> stop;
};

struct GenResult {
  bool ok = true;
  int n = 0;
  double ttft_ms = 0.0;
};

class LlamaEngine {
public:
  LlamaEngine() = default;
  ~LlamaEngine();
  LlamaEngine(const LlamaEngine &) = delete;
  LlamaEngine &operator=(const LlamaEngine &) = delete;

  bool load(const std::string &path, int n_ctx, int n_gpu_layers);
  IngestStats ingest(const std::string &text);
  GenResult generate(const GenParams &p,
                     const std::function<void(const std::string &)> &emit,
                     std::atomic<bool> &cancel);
  void reset();

  size_t n_ctx() const { return n_ctx_; }

private:
  bool truncate_from(llama_pos p0);
  std::vector<llama_token> tokenize(const std::string &text) const;

  llama_model *model_ = nullptr;
  llama_context *ctx_ = nullptr;
  llama_sampler *smpl_ = nullptr;
  const llama_vocab *vocab_ = nullptr;
  bool backend_up_ = false;

  size_t n_ctx_ = 0;
  size_t gen_reserve_ = 256;
  bool adds_bos_ = false;

  std::vector<llama_token> cached_;
  size_t watermark_ = 0;
};
