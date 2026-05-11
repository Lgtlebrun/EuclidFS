#!/usr/bin/bash
# scripts/follow_logs.sh
# Usage: bash scripts/follow_logs.sh [filter]
# e.g.:  bash scripts/follow_logs.sh lrg_colours

LOG_DIR="/home/llebru/DS_selection_effects/logs"
FILTER="${1:-}"

# find the most recent log matching the filter
LATEST=$(ls -t ${LOG_DIR}/*${FILTER}*.log 2>/dev/null | head -1)

if [ -z "$LATEST" ]; then
    echo "No logs found matching '${FILTER}' in ${LOG_DIR}"
    exit 1
fi

echo "Following: $LATEST"
echo "----------------------------------------"
tail -n 1000 -f "$LATEST"