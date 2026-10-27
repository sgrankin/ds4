// Token identity for the two XML parameter formats. Schema parsing/conversion
// remains XGrammar's; this narrow pass changes only its generated wrappers and
// raw-string stopping rule before normal grammar normalization.
#include "grammar_xml.h"
#include "vendor/xgrammar/cpp/grammar_functor.h"
#include "vendor/xgrammar/cpp/json_schema_converter.h"

#include <picojson.h>

#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace ds4 {
namespace {

using Impl = xgrammar::Grammar::Impl;
using Expr = Impl::GrammarExpr;
using Type = Impl::GrammarExprType;

class TokenXML : public xgrammar::GrammarMutator {
public:
    TokenXML(bool deepseek, const GrammarControls &controls)
        : deepseek_(deepseek), controls_(controls) {}

    xgrammar::Grammar Apply(const xgrammar::Grammar &grammar) override {
        source_ = grammar;
        // Every parameter value is referenced from its generated wrapper. Keep
        // that complete value subgraph unchanged: enum/const strings may be
        // identical to a closing delimiter, and nested objects are real JSON.
        for (int32_t i = 0; i < grammar->NumRules(); ++i)
            findValues(grammar->GetRule(i).body_expr_id);
        for (size_t i = 0; i < values_.size(); ++i)
            markRefs(grammar->GetRule(values_[i]).body_expr_id);
        for (int32_t id : values_) data_rules_.insert(grammar->GetRule(id).name);
        return xgrammar::GrammarMutator::Apply(grammar);
    }

protected:
    int32_t VisitByteString(const Expr &expression) override {
        if (data_rules_.count(cur_rule_name_)) return VisitElement(expression);
        const std::string text = base_grammar_->GetByteString(expression);
        if (deepseek_) {
            const std::string open = "<｜DSML｜ parameter name=\"";
            const std::string close = "</｜DSML｜ parameter>";
            if (text.compare(0, open.size(), open) == 0)
                return wrapper(text, 1, "｜DSML｜");
            if (text == close) return wrapper(text, 2, "｜DSML｜");
        } else {
            const std::string open = "<arg_key>";
            if (text.compare(0, open.size(), open) == 0) return wrapper(text, 0, open);
            for (const char *marker : {"</arg_key>", "<arg_value>", "</arg_value>"})
                if (text == marker) return wrapper(text, 0, marker);
        }
        return VisitElement(expression);
    }

    int32_t VisitTagDispatch(const Expr &expression) override {
        if (cur_rule_name_ != "xml_string") return xgrammar::GrammarMutator::VisitTagDispatch(expression);
        Impl::TokenTagDispatch dispatch{{}, false, {}};
        for (const auto &entry : controls_) dispatch.excludes.push_back(entry.second);
        return builder_->AddTokenTagDispatch(dispatch);
    }

private:
    int32_t wrapper(const std::string &text, size_t offset, const std::string &marker) {
        const auto found = controls_.find(marker);
        if (found == controls_.end())
            throw std::invalid_argument("token-aware XML requires dedicated token " + marker);
        std::vector<int32_t> parts;
        if (offset) parts.push_back(builder_->AddByteString(text.substr(0, offset)));
        parts.push_back(builder_->AddTokenSet({found->second}));
        const size_t end = offset + marker.size();
        if (end < text.size()) parts.push_back(builder_->AddByteString(text.substr(end)));
        return builder_->AddSequence(parts);
    }

    bool valueStart(const std::string &text) const {
        return deepseek_ ? text == "\" string=\"true\">" || text == "\" string=\"false\">"
                         : text == "<arg_value>";
    }

    bool valueEnd(const std::string &text) const {
        return text == (deepseek_ ? "</｜DSML｜ parameter>" : "</arg_value>");
    }

    void markRule(int32_t id) {
        if (value_ids_.insert(id).second) values_.push_back(id);
    }

    void markRefs(int32_t id) {
        const auto expression = source_->GetGrammarExpr(id);
        if (expression.type == Type::kRuleRef || expression.type == Type::kRepeat) {
            markRule(expression[0]);
        } else if (expression.type == Type::kSequence || expression.type == Type::kChoices) {
            for (int32_t child : expression) markRefs(child);
        }
    }

    void findValues(int32_t id) {
        const auto expression = source_->GetGrammarExpr(id);
        if (expression.type != Type::kSequence && expression.type != Type::kChoices) return;
        bool value = false;
        for (int32_t child : expression) {
            const auto next = source_->GetGrammarExpr(child);
            if (expression.type == Type::kSequence && next.type == Type::kByteString) {
                const auto text = source_->GetByteString(next);
                if (valueStart(text)) { value = true; continue; }
                if (valueEnd(text)) { value = false; continue; }
            }
            if (value) markRefs(child);
            findValues(child);
        }
    }

    bool deepseek_;
    const GrammarControls &controls_;
    xgrammar::Grammar source_{xgrammar::NullObj{}};
    std::vector<int32_t> values_;
    std::unordered_set<int32_t> value_ids_;
    std::unordered_set<std::string> data_rules_;
};

void adapt(picojson::value &value, const GrammarControls &controls) {
    if (value.is<picojson::array>()) {
        for (auto &item : value.get<picojson::array>()) adapt(item, controls);
        return;
    }
    if (!value.is<picojson::object>()) return;
    auto &node = value.get<picojson::object>();
    const auto type = node.find("type");
    if (type != node.end() && type->second.is<std::string>()) {
        const auto kind = type->second.get<std::string>();
        if (kind == "const_string" || kind == "grammar" || kind == "regex") return;
        if (kind == "json_schema") {
            const auto style = node.find("style");
            if (style == node.end() || !style->second.is<std::string>()) return;
            const auto name = style->second.get<std::string>();
            const bool deepseek = name == "ds4_deepseek41_xml";
            if (!deepseek && name != "ds4_glm_xml") return;
            if (controls.empty()) throw std::invalid_argument("token-aware XML needs control token IDs");
            const auto schema = node.find("json_schema");
            if (schema == node.end()) throw std::invalid_argument("XML format has no JSON schema");
            const auto whitespace = node.find("max_whitespace_cnt");
            std::optional<int> maximum;
            if (whitespace != node.end() && whitespace->second.is<int64_t>())
                maximum = int(whitespace->second.get<int64_t>());
            const auto order = node.find("any_order");
            const bool any_order = order != node.end() && order->second.is<bool>() && order->second.get<bool>();
            const auto excludes = node.find("excludes");
            if (excludes != node.end() && (!excludes->second.is<picojson::array>() || !excludes->second.get<picojson::array>().empty()))
                throw std::invalid_argument("token-aware XML does not support string exclusions");
            auto raw = xgrammar::JSONSchemaToGrammar(schema->second.serialize(), true,
                std::nullopt, std::nullopt, true, maximum, any_order,
                deepseek ? xgrammar::JSONFormat::kDeepSeekV41XML : xgrammar::JSONFormat::kGlmXML);
            auto converted = TokenXML(deepseek, controls).Apply(raw);
            picojson::object replacement;
            replacement["type"] = picojson::value("grammar");
            replacement["grammar"] = picojson::value(converted.ToString());
            value = picojson::value(std::move(replacement));
            return;
        }
    }
    for (auto &entry : node) adapt(entry.second, controls);
}

} // namespace

std::string tokenAwareStructuralTag(const std::string &source, const GrammarControls &controls) {
    // Ordinary structural tags use the upstream parser/cache without another
    // JSON pass. The private styles identify the two model adapter subtrees.
    if (source.find("ds4_deepseek41_xml") == std::string::npos &&
        source.find("ds4_glm_xml") == std::string::npos) return source;
    picojson::value value;
    const auto error = picojson::parse(value, source);
    if (!error.empty()) throw std::invalid_argument(error);
    adapt(value, controls);
    return value.serialize();
}

} // namespace ds4
