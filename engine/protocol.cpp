#include "protocol.hpp"
#include <cstdio>
#include <mutex>
#include <string>
namespace proto {
namespace {}

void send(const nlohmann::json &j) {
  static std::mutex g_out;
  std::lock_guard<std::mutex> lock(g_out);
  const std::string s = j.dump();
  fwrite(s.data(), 1, s.size(), stdout);
  fputc('\n', stdout);
  fflush(stdout);
}
} // namespace proto
