# Claude Code Hooks: Complete Guide (2026 Reference)

*Source: aiorg.dev — Claude Code Hooks: Complete Guide with 20+ Ready-to-Use Examples*

## What Are Claude Code Hooks?

Hooks are shell commands, LLM prompts, or subagents that run automatically at specific points in Claude Code's lifecycle. They give **deterministic control** over an probabilistic system.

- **Command** — shell script (formatting, linting, security)
- **Prompt** — LLM yes/no decision (complex validation)
- **Agent** — multi-turn subagent (verification with file reads + commands)

**Where to configure:** `.claude/settings.json` (project) or `~/.claude/settings.json` (user).

---

## Hook Events (12 total)

| Category | Event | Can block? | Matcher examples |
|----------|--------|------------|------------------|
| **Session** | SessionStart | No | startup, resume, compact, clear |
| | PreCompact | No | manual, auto |
| | SessionEnd | No | clear, logout, other |
| **Tool** | PreToolUse | Yes | Bash, Edit, Write, Read, Glob, Grep, WebFetch, mcp__* |
| | PostToolUse | No | Same |
| | PostToolUseFailure | No | Same |
| | PermissionRequest | Yes | Same |
| **Agent** | SubagentStart | No | Bash, Explore, Plan |
| | SubagentStop | Yes | Same |
| | Stop | Yes | (always fires) |
| **User** | UserPromptSubmit | Yes | (always) |
| | Notification | No | permission_prompt, idle_prompt |

**Exit codes:** `0` = proceed, `2` = block, other = non-blocking error.

---

## Quick Start: Auto-Format on Write/Edit

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write 2>/dev/null; exit 0"
          }
        ]
      }
    ]
  }
}
```

---

## Ready-to-Use Snippets

### Security
- **Block destructive commands** — PreToolUse + Bash matcher, exit 2 on `rm -rf /`, `DROP TABLE`, `--force` push, etc.
- **Protect files** — PreToolUse Write|Edit, script that blocks `.env`, `.env.local`, `secrets/`, `.git/`, lockfiles.
- **Audit Bash** — PostToolUse Bash, append command + timestamp to `.claude/command-audit.log`.

### Code quality
- **Prettier** — PostToolUse Write|Edit (see above).
- **ESLint --fix** — PostToolUse Write|Edit, run on `.ts`/`.tsx`/`.js`/`.jsx`.
- **TypeScript check** — PostToolUse Write|Edit, `npx tsc --noEmit` on TS files.
- **Stop only if tests pass** — Stop hook running `verify-tests.sh`; script must check `stop_hook_active` to avoid infinite loop.

### Workflow
- **Post-compaction context** — SessionStart matcher `compact`, echo reminders (Bun, branch, last commit).
- **Env on session start** — SessionStart matcher `startup`, append to `$CLAUDE_ENV_FILE`.
- **Run tests when test files change** — PostToolUse Write|Edit, if path matches `*.test.*`/`*.spec.*` run vitest.

### Notifications (macOS)
- **Permission prompt** — Notification matcher `permission_prompt`, `osascript` display notification.
- **Task complete** — Notification matcher `idle_prompt`, notify when ready for next instruction.

### MCP
- **Log MCP ops** — PostToolUse matcher `mcp__.*`, append tool name + time to `.claude/mcp-audit.log`.
- **Rate-limit MCP** — PreToolUse, script that counts recent calls and exit 2 if over threshold.

### Permissions
- **Auto-allow WebFetch/WebSearch** — PreToolUse, output `{"decision":"allow"}` or hookSpecificOutput with `permissionDecision: "allow"`.
- **Auto-allow Read/Glob/Grep** — Same, output allow JSON.
- **Deny web in offline mode** — PreToolUse WebFetch|WebSearch, exit 2.

---

## Advanced: Prompt and Agent Hooks

**Prompt hook (Stop):** Ask LLM "Did the request get fully completed?" → respond `{"ok": true}` or `{"ok": false, "reason": "..."}`. If false, Claude continues.

**Agent hook (Stop):** Subagent runs tests, checks TypeScript, looks for `console.log` in prod code; report findings. Can use many tool turns.

---

## When to Use What

| Need | Use | Why |
|------|-----|-----|
| Always format on save | Hook (PostToolUse) | Must happen every time |
| Prefer Bun over npm | CLAUDE.md | Preference |
| Never modify .env | Hook (PreToolUse) | Hard block |
| API route patterns | .claude/rules/ | Context |
| Run /deploy to ship | Custom command | Workflow |
| Access Jira | MCP | External service |
| Verify tests before stop | Hook (Stop) | Enforcement |

---

## Troubleshooting

- **Hook not firing** — `/hooks` to see active; matcher is case-sensitive (e.g. `Bash` not `bash`); script must be executable.
- **JSON validation failed** — Shell profile printing to stdout; wrap interactive-only output in `[[ $- == *i* ]]`.
- **Infinite Stop loop** — In Stop hook, always check `stop_hook_active` and exit 0 if true.
- **jq not found** — `brew install jq` (macOS) or use Python for JSON.

**Test manually:**  
`echo '{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}' | ./.claude/hooks/block-dangerous.sh`  
Expect exit code 2.

---

*Full article and 20+ copy-paste configs: aiorg.dev*
