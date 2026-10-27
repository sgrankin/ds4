#ifndef DS4_GRAMMAR_XML_H
#define DS4_GRAMMAR_XML_H

#include <cstdint>
#include <map>
#include <string>

namespace ds4 {
using GrammarControls = std::map<std::string, int32_t>;
std::string tokenAwareStructuralTag(const std::string &source, const GrammarControls &controls);
}
#endif
