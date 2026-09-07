#!/usr/bin/env bash

set -euo pipefail

PYTHON="$(command -v python3 || command -v python)"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-/app/AKIDA/AKIDA_requirements.txt}"

echo "Python:"
"${PYTHON}" --version

echo "Akida:"
"${PYTHON}" -c 'import akida; print(akida.__version__)'

if [[ -f "${REQUIREMENTS_FILE}" ]]; then
    echo "Installing requirements from ${REQUIREMENTS_FILE}..."

    "${PYTHON}" -m pip install \
        --no-cache-dir \
        --no-warn-script-location \
        --user \
        -r "${REQUIREMENTS_FILE}"
fi

exec "${PYTHON}" /app/AKIDA/testset_inference.py