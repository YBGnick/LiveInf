#include "protocol.h"
#include <cstdio>
#include <mutex>

namespace proto {
namespace {
std::mutex g_out;
}

void send(const nlohmann::json &j) {
  std::lock_guard<std::mutex> lock(g_out);
  const std::string s = j.dump();
  fwrite(s.data(), 1, s.size(), stdout);
  fputc('\n', stdout);
  fflush(stdout);
}
} // namespace proto
