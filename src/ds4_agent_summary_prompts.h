#ifndef DS4_AGENT_SUMMARY_PROMPTS_H
#define DS4_AGENT_SUMMARY_PROMPTS_H
/* Original live-context instructions, defined once by the Agent C TU. Native
 * and fallback system builders borrow exactly the same ordinary-content bytes. */
#ifdef DS4_AGENT_DEFINE_PROMPTS
const char agent_live_context_instructions[] =
            "\nThis file mirrors the conversation you ALREADY SEE, including reasoning, and is "
            "refreshed before tools run. Do not read/cat the complete mirror or paste old text into "
            "edit arguments: that copies it back into your context and can cause a cleanup loop. "
            "Use bash/Python to locate and edit text inside the file without printing it. "
            "If you need boundaries, print only turn headers or a small search/range result.\n"
            "Turns have [[DS4_TURN a3c91e704b82d6f0 role=assistant]] headers with random, stable IDs. "
            "Keep each ID with its turn when editing or moving it; file order determines "
            "conversation order. New turns get fresh IDs. These labels are file "
            "metadata, not conversation text. The raw chat delimiters inside each block must be "
            "preserved for any turn you keep. Leave the initial protected system/tool/prefix section "
            "unchanged; retain task requirements, useful evidence, and the latest user request.\n"
            "Target specific older turn IDs once. Do not repeatedly search for every mention of a "
            "topic: your cleanup request, commands, and search results can mention it too. Batch "
            "related edits; after the edit receipt reports success, continue the task.\n"
            "Recipe: delete complete obsolete turns (replace PATH_ABOVE and choose older IDs):\n"
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            "import re\n"
            "p = Path(\"PATH_ABOVE\"); s = p.read_text()\n"
            "parts = re.split(r\"(\\n\\[\\[DS4_TURN ([0-9a-f]{16}) role=\\w+\\]\\]\\n)\", s)\n"
            "drop = {\"a3c91e704b82d6f0\"}; out = [parts[0]]\n"
            "for i in range(1, len(parts), 3):\n"
            "    if parts[i+1] not in drop: out += [parts[i], parts[i+2]]\n"
            "p.write_text(''.join(out))\n"
            "PY\n"
            "Recipe: archive before deleting needed evidence. In the loop, for each dropped ID, "
            "write parts[i+2] to a separate file under p.parent / 'context-archive' before omitting "
            "the block. Later retrieve only the needed lines from that archive.\n"
            "Recipe: shorten an old turn. Locate its block by ID; replace its content with a "
            "specific summary while preserving its header and native chat delimiters. Do not "
            "retype the removed content. Print nothing or only a small count, never the old/new "
            "transcript. Accepted edits replace your context before tool results are appended.";
#else
extern const char agent_live_context_instructions[];
#endif
#endif
