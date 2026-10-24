# Coding Agent Clients

[README](../README.md) | [Server](SERVER.md)

Start the server first:

```sh
./ds4-server --ctx 100000 --kv-disk-dir .workspaces/client-kv --kv-disk-space-mb 8192
```

Set the client's context limit no higher than the server's. Output tokens also
consume that context; a client limit does not enlarge the server allocation.
The examples use V4.1 Flash and localhost. Change the address for a trusted remote
server. The `dsv4-local` strings are placeholders, not server authentication.

## Pi

The installed Pi 0.83.0 CLI was exercised with the actual `persistent-fast`
wrapper. Start that server instead of the generic command above:

```sh
./ds4-local --server persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --port 8080 --tokens 2048
```

Create an isolated client directory, `mkdir -p .workspaces/pi-local`, and save
this as `.workspaces/pi-local/models.json`:

```json
{
  "providers": {
    "local-ds4": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-responses",
      "apiKey": "local",
      "models": [{
        "id": "ds4",
        "name": "DS4 local",
        "reasoning": true,
        "thinkingLevelMap": {"off": "none"},
        "input": ["text"],
        "contextWindow": 8192,
        "maxTokens": 2048,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
      }]
    }
  }
}
```

The measured isolated setup also used `.workspaces/pi-local/settings.json`:

```json
{
  "compaction": {"enabled": false},
  "retry": {"enabled": false, "provider": {"maxRetries": 0}}
}
```

From the ds4 directory, select the isolated config and start Pi. Keep
`pi_config` when changing to another project directory:

```sh
pi_config="$PWD/.workspaces/pi-local"
PI_CODING_AGENT_DIR="$pi_config" \
  pi --provider local-ds4 --model ds4 --api-key local --thinking off --tools read,write --no-session
```

The placeholder key `local` is not server authentication. This Responses model
configuration matches the wrapper's 8192-token context and the measured 2048
output cap per response. `--thinking off` is a client choice; the preset does
not force reasoning mode globally. The [actual coding qualification](../speed-bench/experiments/pi-agent-preset-hour34-20261004/README.md)
retains the isolated model/settings files, tool events and the first
implementation's 161-check result. It did not perform a context edit or
establish a suffix speedup.

### Legacy Chat Completions configuration

For the generic 100k server command above, the existing Chat Completions setup
remains an alternative:

Add this provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ds4": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dsv4-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "deepseek",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [{
        "id": "deepseek-v4.1-flash",
        "name": "DwarfStar V4.1 Flash",
        "reasoning": true,
        "thinkingLevelMap": {
          "off": null, "minimal": "low", "low": "low",
          "medium": "medium", "high": "high", "xhigh": "xhigh"
        },
        "input": ["text"],
        "contextWindow": 100000,
        "maxTokens": 16384,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
      }]
    }
  }
}
```

Select `ds4/deepseek-v4.1-flash` in Pi. This is a text-only client configuration;
image use also requires the appropriate client input declaration and server
vision encoder.

## OpenCode

Merge the provider into `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ds4": {
      "name": "DwarfStar",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4.1-flash": {
          "name": "DwarfStar V4.1 Flash",
          "limit": {"context": 100000, "output": 16384}
        }
      }
    }
  }
}
```

Select `ds4/deepseek-v4.1-flash` as the model.

## Codex CLI

Use the Responses API. Add a provider to the Codex configuration:

```toml
[model_providers.ds4]
name = "DwarfStar"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

```sh
codex --model deepseek-v4.1-flash -c model_provider=ds4
```

## Claude Code

Use the Anthropic-compatible endpoint. A shell wrapper can select the local
model for the main agent and its secondary model roles:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_AUTH_TOKEN="dsv4-local"
export ANTHROPIC_MODEL="deepseek-v4.1-flash"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4.1-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4.1-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4.1-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4.1-flash"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000
exec claude "$@"
```

Give the wrapper a different name from `claude`. Agent clients can send large
initial prompts, so the first prefill may take time. Disk caching helps reuse
compatible prefixes on later sessions.
