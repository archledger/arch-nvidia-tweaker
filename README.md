# ArchGPU

**Multi-vendor GPU, bootloader, Wayland, and gaming-stack configurator for Arch Linux.** One-click hardware scan; auto-applies only the tweaks your host actually needs; verifies every change is live in the running kernel before marking it done.

Rust + [Slint](https://slint.dev). GUI + CLI. pacman-packaged. Idempotent. State-aware.

> Formerly `arch-nvidia-tweaker`. Renamed to reflect reality: the tool now drives NVIDIA, AMD (amdgpu), and Intel (i915/xe) equally — plus bootloader cmdline injection across GRUB, systemd-boot, Limine, and UKI, plus AUR helper bootstrap, plus legacy-config sanitation.

---

## Install

### One command (recommended)

```bash
git clone https://github.com/archledger/archgpu.git
cd archgpu
./install.sh
```

`install.sh` does everything:

1. Installs `base-devel` if missing (fakeroot + make + gcc — `makepkg` prerequisites)
2. Installs `rust` if missing (compiler toolchain)
3. Builds and installs the pacman package via `makepkg -si`

Safe to re-run: each step checks before acting.

### If you'd rather do it manually

```bash
sudo pacman -S --needed base-devel rust
git clone https://github.com/archledger/archgpu.git
cd archgpu/packaging
makepkg -si
```

### If you've already built and just want to install the `.pkg.tar.zst`

```bash
cd packaging
sudo pacman -U archgpu-*-x86_64.pkg.tar.zst
```

---

## What it does

| Action | Scope |
|---|---|
| **GPU detection** | Enumerates every display controller via `lspci`, classifies vendor (NVIDIA / AMD / Intel) and kernel driver (`xe` vs `i915`, `amdgpu` vs `radeon`). Recommends the correct driver package per GPU architecture (NVIDIA: `nvidia-open-dkms` for Turing+, `nvidia-dkms` for Pascal/Volta, AUR `nvidia-470xx-dkms` / `nvidia-390xx-dkms` for Maxwell/Kepler/Fermi). |
| **Kernel cmdline** | GPU-aware, multi-bootloader. Writes `nvidia-drm.modeset=1` + `nvidia-drm.fbdev=1` (NVIDIA), `amdgpu.ppfeaturemask=0xffffffff` (amdgpu), `i915.enable_guc=3` (i915). `xe` gets nothing — it handles GuC/HuC natively. Supports GRUB (`grub-mkconfig`), systemd-boot (`bootctl update`), Limine (no regen needed), UKI (`mkinitcpio -P`). |
| **Live verification** | After writing, every cmdline param is verified against `/sys/module/<m>/parameters/<p>`. UI shows ✓ Active (green) only when the running kernel confirms; ⟳ Reboot pending (yellow) when config is written but kernel hasn't adopted it yet. |
| **Wayland** | Minimal `/etc/profile.d/99-nvidia-wayland.sh` (comment-only — legacy `GBM_BACKEND=nvidia-drm` global export is deliberately NOT set; it breaks hybrid PRIME); initramfs modules via `MODULES+=()` drop-in (never touches `HOOKS` — removing the `kms` hook breaks simpledrm console handoff); PRIME `OutputClass` Xorg drop-in for hybrid hosts. |
| **Power** | nvidia-suspend / nvidia-hibernate / nvidia-resume systemd units, `NVreg_UseKernelSuspendNotifiers` (all NVIDIA hosts), `NVreg_DynamicPowerManagement=0x02` (laptops only), nouveau blacklist drop-in. |
| **Gaming** | `[multilib]` uncomment; per-vendor Vulkan + VA-API (RADV + `libva-mesa-driver` for AMD, `vulkan-intel` + `intel-media-driver` for Intel Gen8+, `nvidia-utils` + `libva-nvidia-driver` for NVIDIA); gamemode + mangohud + lib32 variants; `vm.max_map_count=1048576` sysctl; legacy NVIDIA drivers bootstrapped from AUR via `yay` (which is itself auto-installed if missing). |
| **Legacy sanitation** | Warns the user — in the GUI and in `--diagnose` — about configs that follow outdated tutorials: AMDVLK alongside RADV, deprecated `xf86-video-intel` DDX, `mesa-vdpau` (removed in Mesa 25), manually-authored `/etc/X11/xorg.conf`, `WLR_NO_HARDWARE_CURSORS=1` or `GBM_BACKEND=nvidia-drm` in other `/etc/profile.d/` scripts. |
| **SUDO_ASKPASS** | yay's internal `sudo` is routed through a DE-appropriate graphical askpass (`ksshaskpass`, `seahorse-ssh-askpass`, `lxqt-openssh-askpass`) so AUR installs don't hang the GUI when no TTY is available. |

---

## Usage

### GUI

Launch **ArchGPU** from your application menu (`pkexec` prompts for auth; polkit preserves your Wayland/X11 env so the window renders). Workflow:

1. Review the **Detected hardware** card — chassis, GPU inventory, bootloader type.
2. Click **✨ Auto-Optimize** — the hero button flips on exactly the switches that would improve this host AND aren't already active/pending.
3. Leave **Dry run** on for Preview; turn it off to actually Apply.
4. Watch the live `pacman` / `mkinitcpio` / `yay` / `grub-mkconfig` / `bootctl` stream in the dark terminal-style Console card.

### CLI

```bash
# Read-only
archgpu --detect                  # hardware + bootloader + per-tweak recommendation
archgpu --diagnose                # 14-point issue scan with remediation hints

# Preview (no writes, no subprocess side-effects)
archgpu --dry-run --apply-all

# Apply individual areas
sudo archgpu --apply-wayland      # env drop-in + initramfs modules + PRIME (hybrid)
sudo archgpu --apply-bootloader   # GPU-aware cmdline + per-bootloader regeneration
sudo archgpu --apply-power        # suspend services + modprobe + nouveau blacklist
sudo archgpu --apply-gaming --yes # multilib + Vulkan + gamemode + mangohud (+ AUR if needed)

# All at once
sudo archgpu --apply-all --yes
```

`--yes` passes `--noconfirm` to pacman (required for GUI installs with no TTY).

---

## UI state badges

Every per-tweak row shows exactly one of:

| Badge | Meaning |
|---|---|
| (Switch) | Unapplied — click Apply to make the change |
| **✓ Active** (green) | Config written AND the running kernel confirms it |
| **⟳ Reboot** (yellow) | Config written; kernel hasn't adopted it yet |
| **Unsupported** (orange) | Doesn't apply to this host (e.g. NVIDIA tweak on Intel-only box) |

The live-kernel probe reads `/sys/module/nvidia_drm/parameters/{modeset,fbdev}`, `/sys/module/amdgpu/parameters/ppfeaturemask`, and `/sys/module/i915/parameters/enable_guc` for the bootloader tweak — so green means "running right now", not "we edited a text file and hoped for the best".

---

## Architecture

| Module | Responsibility |
|---|---|
| `src/core/bootloader.rs` | `BootloaderType` + `detect_active_bootloader` + per-type `apply_{uki,grub,sdb,limine}`, pure param-parsers (`grub_add_param` / `sdb_add_param` / `limine_add_param`), **live-kernel probes** (`/sys/module/<m>/parameters/<p>`), 2-stage `check_state` → `Active` / `PendingReboot` / `Unapplied` / `Incompatible` |
| `src/core/gpu.rs` | `lspci -mm -nn -D` parser, `GpuVendor` / `NvidiaGeneration` classification, driver-family helpers (`uses_xe_driver`, `uses_i915_driver`, `uses_amdgpu_driver`, `uses_radeon_legacy_driver`), NVIDIA driver package recommendation per arch |
| `src/core/hardware.rs` | SMBIOS `chassis_type` → `FormFactor::{Laptop, Desktop, Unknown}` |
| `src/core/wayland.rs` | Modern comment-only profile.d drop-in, `MODULES+=()` mkinitcpio drop-in (HOOKS untouched), hybrid PRIME Xorg config, sanitation scanner for `/etc/X11/xorg.conf` + `WLR_NO_HARDWARE_CURSORS` + global `GBM_BACKEND=nvidia-drm` |
| `src/core/power.rs` | modprobe drop-in (universal + laptop-specific options), nouveau blacklist, `systemctl enable` of nvidia-suspend/hibernate/resume |
| `src/core/gaming.rs` | `[multilib]` state-machine uncommenter, GPU-aware package resolver (`resolve_gaming_packages` + `resolve_aur_packages`), sanitation scanner for AMDVLK / `xf86-video-intel` / `mesa-vdpau`, `vm.max_map_count` sysctl |
| `src/core/aur.rs` | Helper detection (yay / paru), `invoking_user` via `SUDO_USER` / `PKEXEC_UID` + allowlist, manual `yay-bin` bootstrap (git clone + `makepkg` as user → `pacman -U` as root), `SUDO_ASKPASS` routing per DE |
| `src/core/prime.rs` | Xorg `OutputClass` drop-in for hybrid GPUs (skipped when nvidia-utils ships its own) |
| `src/core/diagnostics.rs` | 14-point read-only scanner, `Finding{severity,title,detail,fix_hint}`, surfaces gaming + wayland sanitation warnings |
| `src/core/auto.rs` | `recommend(ctx, form, gpus) -> Actions` — cross-references hardware applicability AND `check_state().is_unapplied()`; used both at GUI startup and by the Auto-Optimize hero |
| `src/core/state.rs` | `TweakState::{Active, PendingReboot, Unapplied, Incompatible}` + helpers |
| `src/utils/process.rs` | `run_streaming(cmd, on_line)` — pipes stdout+stderr through mpsc into an `FnMut(&str)` |
| `src/utils/fs_helper.rs` | `atomic_write` (temp + fsync + rename), `backup_to_dir` (timestamped `/var/backups/archgpu/`), `write_dropin`, `ChangeReport::{AlreadyApplied, Applied, Planned}` |
| `src/cli.rs` | clap CLI; dispatches into `core::run_actions` with a `println!` progress sink |
| `src/gui.rs` | Slint wiring; `populate_detection` polls every `check_state`; `append_log` updates `log-text` + triggers reactive auto-scroll in the Console card |
| `ui/main_window.slint` | Dark Libadwaita-style UI: `Card` / `HeroButton` / `SecondaryButton` / `StateBadge` / `TweakRow` components |
| `packaging/PKGBUILD` | Builds directly from project root (`$startdir/..`) — never touches makepkg's `$srcdir`. `options=('!debug')` so `debugedit` isn't required. |

---

## Tests

```bash
cargo test --release --locked
```

**110+ passing tests** covering GPU parsing across real device IDs, NVIDIA architecture classification, per-vendor package resolution, multilib state-machine editing, bootloader detection for all four types with UKI+SDB coexistence, per-bootloader param-add helpers (quoted / unquoted / empty / idempotent / commented-line edge cases), `check_state` transitions (Active / PendingReboot / Unapplied / Incompatible) including live-sysfs seeding, Auto-Optimize recommendation rules across five hardware scenarios, `atomic_write` / `backup_to_dir` / `write_dropin` behavior, subprocess streaming (interleaved stdout/stderr, non-zero exit propagation), sanitation matrix (AMDVLK / xf86-video-intel / mesa-vdpau / legacy X11 configs / deprecated env vars).

```bash
cargo clippy --all-targets --all-features -- -D warnings
# 0 warnings, fails on any
```

---

## Runtime dependencies

- `pciutils` (lspci)
- `polkit` (pkexec launcher)
- Slint Linux graphics stack: `fontconfig`, `freetype2`, `libxkbcommon`, `wayland`, `libx11`, `libxcb`, `libxcursor`, `libxi`, `libxrandr`, `mesa`

### optdepends

- `nvidia-utils`, `nvidia-prime` — runtime userspace + `prime-run`
- `gamemode`, `mangohud` — auto-installed by `--apply-gaming`
- `yay` — auto-bootstrapped if missing and a legacy AUR driver is needed
- `sbctl` — Secure Boot signing (via its own mkinitcpio post-hook; this tool doesn't sign)
- `ksshaskpass` (KDE/Wayland) or `seahorse` (GNOME) — graphical askpass for yay's inner sudo

---

## Safety

- Every mutating write is preceded by a timestamped backup in `/var/backups/archgpu/`.
- Every write is atomic (temp file + fsync + rename). Crash-safe.
- `--dry-run` runs the full detection + planning pipeline with zero side effects.
- NVIDIA-specific actions are hard-gated on NVIDIA GPU presence. On AMD/Intel hosts they cleanly report "skipped" instead of doing damage.
- yay is **never** run as root. The bootstrap `install(1)`s a user-owned `/tmp` dir, then `sudo -u <user>` for git clone + makepkg, finishing with `pacman -U` as root — no user sudo prompt is required for the bootstrap itself.
- `$SUDO_USER` / `$PKEXEC_UID→getent` are validated against a POSIX allowlist (`[A-Za-z0-9_-]{1,32}`) before interpolation into any path or subprocess.
- Master configs (`/etc/default/grub`, `/etc/pacman.conf`, `/etc/kernel/cmdline`, `/boot/loader/entries/*.conf`, `/boot/limine.conf`) are only touched when the config system provides no drop-in alternative — and those edits are surgical (preserve surrounding content) and backed up first. Everything else (`profile.d`, `modprobe.d`, `mkinitcpio.conf.d`, `sysctl.d`, `X11/xorg.conf.d`) uses drop-ins.
- `/etc/mkinitcpio.conf` is **never** touched directly. We write only to `/etc/mkinitcpio.conf.d/nvidia-modules.conf` with `MODULES+=(…)`. The `HOOKS=()` array is never modified — removing `kms` would break simpledrm console handoff.

---

## License

Dual-licensed: **MIT OR Apache-2.0** (same as most Rust crates).
