#pragma once
#include <nlohmann/json.hpp>

namespace proto {
void send(const nlohmann::json &j);
}
