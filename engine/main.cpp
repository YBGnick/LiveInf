#include "engine.hpp"
#include "protocol.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <thread>

using json = nlohmann::json;

int main(int argc, char **argv) {
  std::string model_path;
  int n_ctx = 8192;
  int n_gpu = 99;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ctx" && i + 1 < argc)
      n_ctx = std::atoi(argv[++i]);
    else if (a == "--gpu" && i + 1 < argc)
      n_gpu = std::atoi(argv[++i]);
    else if (!a.empty() && a[0] != '-' && model_path.empty())
      model_path = a;
  }
  if (model_path.empty()) {
    std::cerr << "usage: liveengine <model.gguf> [--ctx N] [--gpu N]\n";
    return 1;
  }

  LlamaEngine engine;
  if (!engine.load(model_path, n_ctx, n_gpu))
    return 1;
  proto::send(json{{"ev", "ready"}, {"n_ctx", engine.n_ctx()}});

  std::atomic<bool> cancel{false};
  std::thread worker;

  auto stop_worker = [&]() {
    cancel.store(true);
    if (worker.joinable())
      worker.join();
    cancel.store(false);
  };

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty())
      continue;
    json msg;
    try {
      msg = json::parse(line);
    } catch (...) {
      proto::send(json{{"ev", "error"}, {"msg", "bad json"}});
      continue;
    }

    const std::string op = msg.value("op", "");

    if (op == "edit") {
      stop_worker();
      auto st = engine.ingest(msg.value("text", std::string()));
      json ev{{"ev", "ingest"},
              {"lcp", st.lcp},
              {"delta", st.delta},
              {"cached", st.cached},
              {"ms", st.prefill_ms}};
      if (!st.ok)
        ev["err"] = "prefill failed (context full?)";
      proto::send(ev);

    } else if (op == "gen") {
      stop_worker();
      GenParams p;
      p.max_tokens = msg.value("max_tokens", 24);
      if (msg.contains("stop"))
        p.stop = msg["stop"].get<std::vector<std::string>>();
      worker = std::thread([&, p]() {
        auto res = engine.generate(
            p,
            [&](const std::string &txt) {
              proto::send(json{{"ev", "token"}, {"text", txt}});
            },
            cancel);
        if (cancel.load())
          proto::send(json{{"ev", "cancelled"}, {"n", res.n}});
        else
          proto::send(
              json{{"ev", "done"}, {"n", res.n}, {"ttft_ms", res.ttft_ms}});
        if (!res.ok)
          proto::send(json{{"ev", "error"}, {"msg", "generation error"}});
      });

    } else if (op == "cancel") {
      stop_worker();

    } else if (op == "reset") {
      stop_worker();
      engine.reset();
      proto::send(json{{"ev", "reset"}});

    } else {
      proto::send(json{{"ev", "error"}, {"msg", "unknown op"}});
    }
  }
  stop_worker();
  return 0;
}
