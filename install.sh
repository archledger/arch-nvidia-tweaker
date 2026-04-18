#!/usr/bin/env bash
# ArchGPU — one-shot installer.
#
# Installs the makepkg prerequisites (`base-devel`, `rust`) if they're missing,
# then builds and installs the ArchGPU pacman package. Safe to re-run: each
# step checks before acting.
#
# Usage:
#   ./install.sh
#
# Requires: Arch Linux (or an Arch derivative with pacman + makepkg). Prompts
# for your sudo password when installing packages.

set -euo pipefail

# ── Colors for readable progress ─────────────────────────────────────────────
if [[ -t 1 ]]; then
    GREEN="\033[1;32m"; BLUE="\033[1;34m"; YELLOW="\033[1;33m"
    RED="\033[1;31m";   BOLD="\033[1m";    RESET="\033[0m"
else
    GREEN=""; BLUE=""; YELLOW=""; RED=""; BOLD=""; RESET=""
fi

say()  { printf "${BLUE}➜${RESET} ${BOLD}%s${RESET}\n" "$*"; }
ok()   { printf "${GREEN}✓${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}⚠${RESET} %s\n" "$*"; }
die()  { printf "${RED}✗ %s${RESET}\n" "$*" >&2; exit 1; }

# ── Sanity: Arch Linux? ──────────────────────────────────────────────────────
command -v pacman  >/dev/null 2>&1 || die "pacman not found. ArchGPU targets Arch Linux."
command -v makepkg >/dev/null 2>&1 || die "makepkg not found (should ship with pacman)."

# ── Must NOT run as root — makepkg refuses, and we want sudo to prompt once ──
if [[ $EUID -eq 0 ]]; then
    die "Do not run this script as root. Run as your normal user; sudo will prompt when needed."
fi

# ── 1. base-devel (fakeroot, make, gcc, binutils, …) ────────────────────────
if pacman -Qi fakeroot >/dev/null 2>&1; then
    ok "base-devel already installed (fakeroot present)."
else
    say "Installing base-devel (required by makepkg for fakeroot + toolchain)..."
    sudo pacman -S --needed base-devel
fi

# ── 2. rust (cargo + rustc) ─────────────────────────────────────────────────
if command -v cargo >/dev/null 2>&1; then
    ok "rust already installed ($(cargo --version))."
else
    say "Installing rust (required to compile archgpu)..."
    sudo pacman -S --needed rust
fi

# ── 3. Build + install the package ──────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/packaging"

say "Building and installing archgpu via makepkg -si ..."
echo
makepkg -si

echo
ok "ArchGPU installed. Try:"
echo
echo "    archgpu --detect         # hardware + bootloader + recommendation"
echo "    archgpu --diagnose       # read-only 14-point issue scan"
echo
echo "    # or launch the GUI from your app menu (\"ArchGPU\")"
echo
