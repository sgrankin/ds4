// Installed official OpenAI SDK + installed Pi provider, local server only.
import OpenAI from '/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent/node_modules/openai/index.mjs';
import { stream as piStream } from '/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai/dist/api/openai-responses.js';
import { mkdir, writeFile, readFile, appendFile } from 'node:fs/promises';
import path from 'node:path';

const argv = process.argv.slice(2);
const flag = name => argv[argv.indexOf(name) + 1];
const baseURL = flag('--base-url');
const out = path.resolve(flag('--output'));
if (!baseURL || !['127.0.0.1', 'localhost'].includes(new URL(baseURL).hostname))
  throw new Error('a localhost --base-url is required');
await mkdir(out, { recursive: false });
const save = (name, value) => writeFile(path.join(out, name), JSON.stringify(value, null, 2) + '\n');
const pendingWire = [];
let wireIndex = 0;
const localFetch = async (url, options = {}) => {
  const target = new URL(typeof url === 'string' || url instanceof URL ? url : url.url);
  if (target.origin !== new URL(baseURL).origin) throw new Error('non-local request refused');
  const index = String(wireIndex++).padStart(2, '0');
  await save(`wire-${index}-request.json`, {
    url: target.href, method: options.method || 'GET',
    body: typeof options.body === 'string' ? JSON.parse(options.body) : null,
  });
  const response = await fetch(url, options);
  pendingWire.push(response.clone().text().then(async text => {
    await writeFile(path.join(out, `wire-${index}-response.txt`), text);
    await save(`wire-${index}-http.json`, {
      status: response.status, headers: Object.fromEntries(response.headers),
    });
  }));
  return response;
};
const client = new OpenAI({ apiKey: 'local', baseURL, maxRetries: 0, timeout: 120000, fetch: localFetch });
const models = await client.models.list();
const modelID = models.data[0]?.id;
if (!modelID) throw new Error('local server returned no model');
const parameters = {
  type: 'object', properties: {
    path: { type: 'string' }, whole: { type: 'boolean' }, raw: { type: 'boolean' },
  }, required: ['path', 'whole', 'raw'], additionalProperties: false,
};
const tool = { type: 'function', name: 'read', description: 'Read the complete owned configuration file.', parameters, strict: false };
const results = { sdk_version: '6.26.0', pi_version: '0.83.0', model: modelID, checks: {}, errors: [],
  scope: 'Actual client/parser and tiny retrieval smoke; no broad task-quality or latency claim.' };
const check = (condition, message) => { if (!condition) throw new Error(message); };
// The stream helper adds parsed=null annotations which GET does not add.
const wireOutput = output => JSON.stringify(output, (key, value) =>
  key === 'parsed' || key === 'parsed_arguments' ? undefined : value);
async function ownedConfig(name) {
  const file = path.join(out, `${name}-config.json`);
  const value = name === 'sdk' ? 'LIME-27' : 'COVE-42';
  await writeFile(file, JSON.stringify({ answer: value }) + '\n');
  return { file, value, prompt: `Read ${file} with read(path, whole=true, raw=true). After the read result, reply with only the answer field, without quotes or commentary.` };
}
async function fulfill(config, args) {
  check(path.resolve(args.path) === config.file && args.whole === true && args.raw === true,
    'tool must read exactly the owned file with whole=true/raw=true');
  const text = await readFile(config.file, 'utf8');
  return text;
}
async function sdkRound(name, input) {
  await save(`${name}-input.json`, input);
  const started = performance.now();
  const stream = client.responses.stream(input);
  for await (const event of stream)
    await appendFile(path.join(out, `${name}-events.jsonl`), JSON.stringify(event) + '\n');
  const response = await stream.finalResponse();
  await save(`${name}-response.json`, response);
  return { response, seconds: (performance.now() - started) / 1000 };
}
try {
  const config = await ownedConfig('sdk');
  const common = { model: modelID, temperature: 0, reasoning: { effort: 'none' }, max_output_tokens: 256, store: true };
  const first = await sdkRound('sdk-read', { ...common, input: config.prompt, tools: [tool] });
  const calls = first.response.output.filter(item => item.type === 'function_call');
  check(first.response.status === 'completed' && calls.length === 1 && calls[0].name === 'read', 'SDK did not complete exactly one read call');
  const call = calls[0];
  check(call.id && call.call_id, 'SDK function call IDs missing');
  const contents = await fulfill(config, JSON.parse(call.arguments));
  await save('sdk-tool-result.json', { call_id: call.call_id, file: config.file, contents });
  const final = await sdkRound('sdk-final', { ...common, previous_response_id: first.response.id,
    input: [{ type: 'function_call_output', call_id: call.call_id, output: contents }] });
  const fetchedFirst = await client.responses.retrieve(first.response.id);
  const fetchedFinal = await client.responses.retrieve(final.response.id);
  await save('sdk-retrieved-read.json', fetchedFirst);
  await save('sdk-retrieved-final.json', fetchedFinal);
  check(fetchedFirst.id === first.response.id && fetchedFinal.id === final.response.id &&
    wireOutput(fetchedFirst.output) === wireOutput(first.response.output) &&
    wireOutput(fetchedFinal.output) === wireOutput(final.response.output), 'stored response IDs/output differ from streamed finalResponse');
  const text = final.response.output.filter(item => item.type === 'message')
    .flatMap(item => item.content).filter(item => item.type === 'output_text').map(item => item.text).join('').trim();
  results.checks.sdk = { pass: final.response.status === 'completed' && text === config.value,
    read_id: first.response.id, final_id: final.response.id, call_id: call.call_id,
    final_text: text, expected: config.value, status: final.response.status,
    incomplete_details: final.response.incomplete_details || null, usage: final.response.usage,
    seconds: first.seconds + final.seconds, stored_output_exact: true };
} catch (error) { results.errors.push({ check: 'sdk', error: String(error.stack || error) }); }

try {
  const conversation = await client.conversations.create({ metadata: { smoke: 'installed-sdk' } });
  const appended = await client.conversations.items.create(conversation.id,
    { items: [{ type: 'message', role: 'user', content: 'A local SDK item.' }] });
  const list = await client.conversations.items.list(conversation.id, { order: 'asc', limit: 20 });
  check(appended.data?.length === 1 && list.data?.length === 1, 'conversation append/list shape differs');
  const item = await client.conversations.items.retrieve(list.data[0].id, { conversation_id: conversation.id });
  const fetched = await client.conversations.retrieve(conversation.id);
  const deletedItem = await client.conversations.items.delete(item.id, { conversation_id: conversation.id });
  const empty = await client.conversations.items.list(conversation.id);
  const deleted = await client.conversations.delete(conversation.id);
  check(item.id === list.data[0].id && fetched.id === conversation.id && empty.data.length === 0 && deleted.deleted === true,
    'conversation CRUD check failed');
  results.checks.conversations = { pass: true, conversation, appended, list: list.data, item, fetched, deletedItem, deleted };
} catch (error) { results.errors.push({ check: 'conversations', error: String(error.stack || error) }); }

try {
  const config = await ownedConfig('pi');
  const model = { id: modelID, name: modelID, api: 'openai-responses', provider: 'local-ds4', baseUrl: baseURL,
    reasoning: true, input: ['text'], contextWindow: 8192, maxTokens: 256,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, thinkingLevelMap: { off: 'none' } };
  const context = { systemPrompt: 'Use the read tool for the requested file. After receiving the result, output only the answer field.',
    messages: [{ role: 'user', content: config.prompt, timestamp: Date.now() }],
    tools: [{ name: 'read', description: tool.description, parameters }] };
  const options = { apiKey: 'local', sessionId: 'hour17-pi-read', cacheRetention: 'short', temperature: 0,
    maxTokens: 256, maxRetries: 0, timeoutMs: 120000, fetch: localFetch };
  async function round(name) {
    await save(`${name}-context.json`, context);
    const started = performance.now();
    const stream = piStream(model, context, { ...options, onPayload: payload => save(`${name}-payload.json`, payload).then(() => undefined) });
    for await (const event of stream)
      await appendFile(path.join(out, `${name}-events.jsonl`), JSON.stringify(event) + '\n');
    const message = await stream.result();
    await save(`${name}-response.json`, message);
    return { message, seconds: (performance.now() - started) / 1000 };
  }
  const first = await round('pi-read');
  const calls = first.message.content.filter(item => item.type === 'toolCall');
  check(first.message.stopReason === 'toolUse' && calls.length === 1 && calls[0].name === 'read', 'Pi did not parse exactly one read tool call');
  const call = calls[0];
  const contents = await fulfill(config, call.arguments);
  await save('pi-tool-result.json', { id: call.id, file: config.file, contents });
  context.messages.push(first.message, { role: 'toolResult', toolCallId: call.id, toolName: call.name,
    content: [{ type: 'text', text: contents }], isError: false, timestamp: Date.now() });
  const final = await round('pi-final');
  const text = final.message.content.filter(item => item.type === 'text').map(item => item.text).join('').trim();
  results.checks.pi = { pass: final.message.stopReason === 'stop' && text === config.value,
    read_id: first.message.responseId, final_id: final.message.responseId, tool_id: call.id,
    final_text: text, expected: config.value, stopReason: final.message.stopReason,
    rawStopReason: final.message.rawStopReason, usage: final.message.usage,
    seconds: first.seconds + final.seconds, full_input: true, store: false, sessionId: options.sessionId };
} catch (error) { results.errors.push({ check: 'pi', error: String(error.stack || error) }); }
await Promise.all(pendingWire);
results.pass = results.errors.length === 0 && ['sdk', 'conversations', 'pi'].every(name => results.checks[name]?.pass === true);
await save('result.json', results);
console.log(JSON.stringify(results));
process.exitCode = results.pass ? 0 : 3;
