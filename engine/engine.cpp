#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <llama-cpp.h>
namespace {

using clock_ = std::chrono::steady_clock;

double ms_since(clock_::time_point t0) {
  return std::chrono::duration<double, std::milli>(clock_::now() - t0).count();
}

struct Batch {
  std::vector<llama_token> token;
  std::vector<llama_pos> pos;
  std::vector<int32_t> n_seq_id;
  std::vector<llama_seq_id> seq0{0};
  std::vector<llama_seq_id *> seq_ptr;
  std::vector<int8_t> logits;
  llama_batch batch{};

  Batch(const std::vector<llama_token> &toks, size_t pos0, bool logits_last) {
    const size_t n = toks.size();
    token.resize(n);
    pos.resize(n);
    n_seq_id.assign(n, 1);
    seq_ptr.resize(n);
    logits.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
      token[i] = toks[i];
      pos[i] = (llama_pos)(pos0 + i);
      seq_ptr[i] = seq0.data();
    }
    if (logits_last && n)
      logits[n - 1] = 1;
    batch.n_tokens = (int32_t)n;
    batch.token = token.data();
    batch.embd = nullptr;
    batch.pos = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id = seq_ptr.data();
    batch.logits = logits.data();
  }
  llama_batch get() const { return batch; }
};

size_t utf8_safe_len(const std::string &s) {
  size_t n = s.size();
  if (n == 0)
    return 0;
  auto cont = [&](size_t i) { return (((unsigned char)s[i]) & 0xC0) == 0x80; };
  if (!cont(n - 1))
    return n - 1;
  size_t k = 1;
  while (k < 3 && k < n && cont(n - 1 - k))
    ++k;
  if (k < n) {
    unsigned char lead = (unsigned char)s[n - 1 - k];
    size_t need = (lead & 0xE0) == 0xC0   ? 1
                  : (lead & 0xF0) == 0xE0 ? 2
                  : (lead & 0xF8) == 0xF0 ? 3
                                          : 0;
    if (need && k < need)
      return n - k - 1;
  }
  return n;
}

} // namespace

LlamaEngine::~LlamaEngine() {
  if (smpl_)
    llama_sampler_free(smpl_);
  if (ctx_)
    llama_free(ctx_);
  if (model_)
    llama_model_free(model_);
}

bool LlamaEngine::load(const std::string &path, int n_ctx, int n_gpu_layers) {
  llama_backend_init();

  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = n_gpu_layers;
  model_ = llama_model_load_from_file(path.c_str(), mp);
  if (!model_) {
    fprintf(stderr, "engine: failed to load '%s'\n", path.c_str());
    return false;
  }

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = n_ctx;
  cp.n_batch = 512;
  ctx_ = llama_init_from_model(model_, cp);
  if (!ctx_) {
    fprintf(stderr, "engine: context init failed\n");
    llama_model_free(model_);
    model_ = nullptr;
    return false;
  }
  n_ctx_ = llama_n_ctx(ctx_);

  llama_sampler_chain_params sp = {};
  smpl_ = llama_sampler_chain_init(sp);
  llama_sampler_push(smpl_, llama_sampler_init_greedy());
  // llama_sampler_push(smpl_, llama_sampler_init_temp(0.2f));
  // llama_sampler_push(smpl_, llama_sampler_init_dist(1.0f));

  auto probe = tokenize("hello");
  adds_bos_ = !probe.empty() && probe[0] == llama_token_bos(model_);
  return true;
}

bool LlamaEngine::truncate_from(llama_pos p0) {

  //   return llama_kv_cache_seq_rm(ctx_, 0, p0, -1);
  return llama_memory_seq_rm(llama_get_memory(ctx_), 0, p0, -1);
}

std::vector<llama_token> LlamaEngine::tokenize(const std::string &text) const {
  const int n = llama_tokenize(model_, text.data(), (int32_t)text.size(),
                               nullptr, 0, true, false);
  if (n <= 0)
    return {};
  std::vector<llama_token> toks(n);
  llama_tokenize(model_, text.data(), (int32_t)text.size(), toks.data(), n,
                 true, false);
  return toks;
}

void LlamaEngine::reset() {
  if (ctx_)
    truncate_from(0);
  cached_.clear();
  watermark_ = 0;
}

IngestStats LlamaEngine::ingest(const std::string &text) {
  auto t0 = clock_::now();
  IngestStats st;
  auto ids = tokenize(text);

  // overflow policy: wipe cache, keep only a token tail that fits
  if ((size_t)n_ctx_ <= ids.size() + gen_reserve_) {
    if (!truncate_from(0)) {
      st.ok = false;
      return st;
    }
    cached_.clear();
    size_t keep = (size_t)n_ctx_ - gen_reserve_ - 8;
    if (keep < ids.size())
      ids.erase(ids.begin(), ids.end() - keep);
    if (adds_bos_)
      ids.insert(ids.begin(), llama_token_bos(model_));
  }

  size_t lcp = 0;
  while (lcp < cached_.size() && lcp < ids.size() && cached_[lcp] == ids[lcp])
    ++lcp;

  if (lcp == ids.size() && lcp > 0)
    --lcp;

  if (lcp < cached_.size()) {
    if (!truncate_from((llama_pos)lcp)) {
      truncate_from(0);
      cached_.clear();
      watermark_ = 0;
      st.ok = false;
      st.prefill_ms = ms_since(t0);
      return st;
    }
    cached_.resize(lcp);
  }

  st.lcp = lcp;
  st.delta = ids.size() - lcp;

  const size_t NB = 512;
  for (size_t i = lcp; i < ids.size() && st.ok; i += NB) {
    size_t n = std::min(NB, ids.size() - i);
    std::vector<llama_token> chunk(ids.begin() + i, ids.begin() + i + n);
    Batch b(chunk, i, (i + n == ids.size()));
    if (llama_decode(ctx_, b.get()) != 0)
      st.ok = false;
  }
  if (!st.ok) {
    truncate_from(0);
    cached_.clear();
    watermark_ = 0;
    st.prefill_ms = ms_since(t0);
    return st;
  }

  cached_ = ids;
  watermark_ = cached_.size();
  st.cached = cached_.size();
  st.prefill_ms = ms_since(t0);
  return st;
}

GenResult
LlamaEngine::generate(const GenParams &p,
                      const std::function<void(const std::string &)> &emit,
                      std::atomic<bool> &cancel) {
  GenResult res;
  if (cached_.empty())
    return res;

  int budget = p.max_tokens;
  if (watermark_ + (size_t)budget + 1 > n_ctx_)
    budget = (int)(n_ctx_ - watermark_ - 1);
  if (budget <= 0) {
    res.ok = false;
    return res;
  }

  llama_sampler_reset(smpl_);
  auto t0 = clock_::now();
  bool first = true;

  std::string pending;
  llama_token tok = llama_sampler_sample(smpl_, ctx_, -1);

  int n = 0;
  while (n < budget && !cancel.load(std::memory_order_relaxed)) {
    if (llama_token_is_eog(model_, tok))
      break;

    char piece[256];
    int len = llama_token_to_piece(ctx_, tok, piece, sizeof(piece), 0, false);
    if (len > 0)
      pending.append(piece, (size_t)len);

    bool stopped = false;
    size_t upto = pending.size();
    for (const auto &s : p.stop) {
      if (s.empty())
        continue;
      auto at = pending.find(s);
      if (at != std::string::npos) {
        stopped = true;
        upto = std::min(upto, at);
        break;
      }
    }
    if (!stopped) {
      size_t maxstop = 0;
      for (const auto &s : p.stop)
        maxstop = std::max(maxstop, s.size());
      size_t hold = std::min(pending.size(), maxstop ? maxstop - 1 : 0);
      for (; hold > 0; --hold) {
        bool prefix = false;
        for (const auto &s : p.stop)
          if (s.compare(0, hold, pending, pending.size() - hold, hold) == 0) {
            prefix = true;
            break;
          }
        if (prefix)
          break;
      }
      upto = std::min(upto, pending.size() - hold);
      upto = std::min(upto, utf8_safe_len(pending.substr(0, upto)));
    }

    if (upto > 0) {
      emit(pending.substr(0, upto));
      if (first) {
        res.ttft_ms = ms_since(t0);
        first = false;
      }
      pending.erase(0, upto);
    }
    if (stopped)
      break;

    std::vector<llama_token> one{tok};
    Batch b(one, watermark_ + (size_t)n, true);
    if (llama_decode(ctx_, b.get()) != 0) {
      res.ok = false;
      break;
    }
    ++n;
    tok = llama_sampler_sample(smpl_, ctx_, -1);
  }

  truncate_from((llama_pos)watermark_);
  res.n = n;
  return res;
}
