"""Dangerous command patterns database."""

from __future__ import annotations

# Interpreters that enable arbitrary code execution
DANGEROUS_INTERPRETERS: set[str] = {
    "eval", "exec", "python", "python3", "node", "deno", "tsx",
    "ruby", "perl", "php", "lua", "bash", "sh", "zsh", "fish",
    "npx", "bunx", "bun run", "npm run", "yarn run", "pnpm run",
}

# Commands that are always blocked
DANGEROUS_COMMANDS: list[str] = [
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf $HOME",
    "mkfs.",
    ":(){:|:&};:",         # fork bomb
    "> /dev/sda",
    "dd if=/dev/zero",
    "chmod -R 777 /",
    "wget -O- | sh",
    "curl | sh",
    "curl | bash",
]

# Environment variables that should not be manipulated
DANGEROUS_ENV_VARS: set[str] = {
    "$IFS", "$LD_PRELOAD", "$LD_LIBRARY_PATH",
    "$HISTFILE", "$PATH", "$HOME",
    "$DYLD_INSERT_LIBRARIES",
}
