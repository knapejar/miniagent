@echo off
rem Claude Code driving Qwen3.8-27B on the Kaggle TPU through `python launch.py proxy`.
rem Start the proxy first (it listens on 127.0.0.1:8080 and adds the real API key).
rem Usage: tools\claude-qwen.cmd [claude args]      working dir: %USERPROFILE%\claude-qwen
title Claude Code - Qwen3.8 Kaggle
chcp 65001 >nul
rem drop markers inherited when this is started from another Claude Code session
for %%V in (CLAUDECODE CLAUDE_CODE_CHILD_SESSION CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_SESSION_ID CLAUDE_PID CLAUDE_EFFORT CLAUDE_CODE_EXECPATH CLAUDE_CODE_MESSAGING_TOKEN CLAUDE_CODE_MESSAGING_SOCKET CLAUDE_CODE_SESSION_ATTENDED) do set %%V=
set ANTHROPIC_BASE_URL=http://127.0.0.1:8080
set ANTHROPIC_AUTH_TOKEN=sk-proxy-adds-the-real-key
set ANTHROPIC_API_KEY=
set ANTHROPIC_MODEL=qwen3.8-27b
set ANTHROPIC_SMALL_FAST_MODEL=qwen3.8-27b
set ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8-27b
set ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8-27b
set ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-27b
set CLAUDE_CODE_SUBAGENT_MODEL=qwen3.8-27b
set CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144
set CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
set API_TIMEOUT_MS=1800000
if not exist "%USERPROFILE%\claude-qwen" mkdir "%USERPROFILE%\claude-qwen"
cd /d "%USERPROFILE%\claude-qwen"
claude %*
