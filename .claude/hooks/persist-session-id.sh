#!/bin/bash
# Cozempic SessionStart hook: persist session_id as CLAUDE_SESSION_ID
# so it's available in Bash commands throughout the session.
#
# Install by adding to your .claude/settings.json:
#   "hooks": {
#     "SessionStart": [{
#       "hooks": [{
#         "type": "command",
#         "command": "/path/to/persist-session-id.sh"
#       }]
#     }]
#   }

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null)

if [ -n "$CLAUDE_ENV_FILE" ] && [ -n "$SESSION_ID" ]; then
    echo "export CLAUDE_SESSION_ID=\"$SESSION_ID\"" >> "$CLAUDE_ENV_FILE"
fi

exit 0
