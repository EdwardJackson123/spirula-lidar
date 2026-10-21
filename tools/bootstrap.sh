#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python3 -m venv "$project_dir/.venv"
"$project_dir/.venv/bin/python" -m pip install -e "$project_dir[worker,test]"
printf '%s\n' 'Activate .venv/bin/activate, install Spirula separately, then run spirula-lidar doctor --spirula /path/to/spirula.'
