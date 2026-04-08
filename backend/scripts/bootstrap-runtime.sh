#!/usr/bin/env bash
set -euo pipefail

have() {
  command -v "$1" >/dev/null 2>&1
}

install_with_brew() {
  local pkg="$1"
  if have brew; then
    echo "[install] $pkg via brew"
    brew install "$pkg"
    return 0
  fi
  return 1
}

install_with_apt() {
  local pkg="$1"
  if have apt-get; then
    echo "[install] $pkg via apt-get"
    sudo apt-get update
    sudo apt-get install -y "$pkg"
    return 0
  fi
  return 1
}

ensure_cmd() {
  local cmd="$1"
  local brew_pkg="$2"
  local apt_pkg="$3"
  if have "$cmd"; then
    echo "[ok] $cmd"
    return
  fi
  install_with_brew "$brew_pkg" || install_with_apt "$apt_pkg" || {
    echo "[missing] $cmd (no supported package manager found)"
  }
}

echo "== Skill Runtime Bootstrap (Unix) =="
ensure_cmd git git git
ensure_cmd bash bash bash
ensure_cmd node node nodejs
ensure_cmd npm node npm
ensure_cmd python3 python python3
ensure_cmd jq jq jq
ensure_cmd curl curl curl
ensure_cmd sed gnu-sed sed
ensure_cmd awk gawk gawk
ensure_cmd base64 coreutils coreutils

echo
echo "Done. Re-run /api/skills preflight to confirm runtime readiness."
