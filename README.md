# Arch NVIDIA Tweaker

An automated configurator for NVIDIA, Wayland, power-management, and gaming setup on Arch Linux. Detects your bootloader (GRUB / systemd-boot / Limine / UKI) and GPU, then applies only the tweaks that are actually missing.

Built with Rust + [Slint](https://slint.dev). CLI + GUI. Idempotent. State-aware. pacman-packaged.

---

## Features

- **Auto-Optimize** — one-click hardware + state scan; the hero button flips on *only* toggles whose tweak is applicable to your hardware AND currently Unapplied.
- **Multi-bootloader cmdline injection** — GRUB (`/etc/default/grub` + `grub-mkconfig`), systemd-boot (`/boot/loader/entries/*.conf` + `bootctl update`), Limine (`cmdline:` / `kernel_cmdline:` / `KERNEL_CMDLINE=` across v9+ and legacy syntax), UKI (`/etc/kernel/cmdline` + `mkinitcpio -P`). Detection picks the right source automatically.
- **State-aware (`TweakState::{Applied, Unapplied, Incompatible}`)** — every module exposes a read-only `check_state()` probe; the GUI disables Switches for tweaks that are already done (green *✓ Applied* badge) or don't apply to this hardware (orange *Unsupported* badge).
- **GPU-aware packages** — lspci-driven NVIDIA architecture classification (Blackwell / Ada / Hopper / Ampere / Turing / Volta / Pascal / Maxwell / Kepler / Fermi), recommends `nvidia-open-dkms` (Turing+ post-NVIDIA-590), `nvidia-dkms` (Pascal/Volta), or AUR legacy drivers (`nvidia-470xx-dkms` / `nvidia-390xx-dkms`). Non-NVIDIA hosts get `vulkan-intel` or `vulkan-radeon` with NVIDIA actions cleanly skipped.
- **AUR auto-bootstrap** — if a legacy NVIDIA driver is needed but no `yay` / `paru` is installed, the tool clones `yay-bin`, `makepkg`s it as the invoking user, and `pacman -U`'s the result as root — *without* requiring the user to have cached sudo credentials for the bootstrap itself.
- **SUDO_ASKPASS routing** — when `yay` is later used to install AUR packages from the GUI, its internal `sudo pacman` calls are routed through a DE-appropriate graphical askpass (`ksshaskpass` on KDE/Plasma/Wayland, `seahorse-ssh-askpass` on GNOME, `lxqt-openssh-askpass` on XFCE/LXQt, …), with `DISPLAY` / `WAYLAND_DISPLAY` / `XAUTHORITY` / `XDG_RUNTIME_DIR` / `XDG_CURRENT_DESKTOP` forwarded so the prompt actually renders.
- **Diagnostics scanner** — 14 read-only checks: `nvidia-drm.modeset=1` in cmdline, installed NVIDIA package, kernel module loaded, nouveau conflict, suspend services, mkinitcpio modules, `[multilib]`, Vulkan ICD per vendor, gamemode/mangohud, `video` group membership, `XDG_SESSION_TYPE`, `vm.max_map_count`, Bumblebee residue, AUR helper presence.
- **Live subprocess streaming** — pacman / mkinitcpio / grub-mkconfig / bootctl / makepkg / yay stdout+stderr piped line-by-line into the GUI's console card. Reactive auto-scroll: the ScrollView's `changed viewport-height` handler snaps to the bottom on every layout, like `tail -f`.
- **Idempotent, atomic, backed up** — every mutating write goes through `atomic_write` (temp file + fsync + rename) and is preceded by `backup_to_dir()` (timestamped copy in `/var/backups/arch-nvidia-tweaker/`).
- **pkexec-launchable GUI** — polkit policy with `auth_admin_keep` + `exec.allow_gui=true` preserves the user's Wayland/X11 session env so the Slint GUI renders when launched from the desktop menu.

---

## Installation

### Prerequisites

`makepkg` (the PKGBUILD build driver) requires the standard Arch **`base-devel`** group — it contains `fakeroot`, `make`, `patch`, and friends. On minimal installs / containers / fresh chroots these aren't present by default. If you hit:

```
ERROR: Cannot find the fakeroot binary
```

install the group and retry:

```bash
sudo pacman -S --needed base-devel
```

(The PKGBUILD sets `options=('!debug')` so `debugedit` is NOT required.)

### via `makepkg` (recommended)

```bash
git clone https://github.com/archledger/arch-nvidia-tweaker.git
cd arch-nvidia-tweaker/packaging
makepkg -si
```

That will:

1. Build with `cargo build --release --locked`
2. Run the unit tests (`cargo test --release --locked`)
3. Install binary → `/usr/bin/arch-nvidia-tweaker`
4. Install desktop entry → `/usr/share/applications/`
5. Install polkit policy → `/usr/share/polkit-1/actions/`

> ℹ️ The PKGBUILD lives in `packaging/` (not the project root) deliberately. `makepkg`'s `$srcdir` is always `$startdir/src`, which would otherwise collide with this project's own `src/` directory. With PKGBUILD one level down, `$srcdir` becomes `packaging/src` and the Rust source is never at risk.

### Install a prebuilt package

If you've already run `makepkg` once, the `.pkg.tar.zst` sits in `packaging/` and can be installed directly (no rebuild):

```bash
cd packaging
sudo pacman -U arch-nvidia-tweaker-*-x86_64.pkg.tar.zst
```

### build-only (no install)

```bash
cargo build --release --locked
./target/release/arch-nvidia-tweaker --detect
```

---

## Usage

### GUI

After `makepkg -si`, launch **Arch NVIDIA Tweaker** from your desktop menu. `pkexec` prompts for auth; polkit preserves the Wayland display socket so the window renders.

Workflow:

1. Review the **Detected hardware** card (chassis, GPU inventory, bootloader).
2. Click **✨ Auto-Optimize (Recommended)** — it flips on exactly the toggles that would improve this host.
3. Keep **Dry run** on for a Preview first; turn it off to actually Apply.
4. Watch the live pacman / mkinitcpio / yay stream in the Console card.

### CLI

```bash
# Read-only
arch-nvidia-tweaker --detect                  # hardware + bootloader + recommendation
arch-nvidia-tweaker --diagnose                # 14-point issue scan with suggested fixes

# Preview (no writes)
arch-nvidia-tweaker --dry-run --apply-all

# Apply individual areas
sudo arch-nvidia-tweaker --apply-wayland      # env vars + mkinitcpio modules + PRIME for hybrid
sudo arch-nvidia-tweaker --apply-bootloader   # nvidia-drm.modeset=1 on detected bootloader
sudo arch-nvidia-tweaker --apply-power        # suspend/resume services + modprobe + nouveau blacklist
sudo arch-nvidia-tweaker --apply-gaming --yes # multilib + Vulkan + gamemode + mangohud (+ AUR if needed)

# Apply everything
sudo arch-nvidia-tweaker --apply-all --yes
```

`--yes` passes `--noconfirm` to pacman (required for GUI installs with no TTY).

---

## Architecture

| Module | Responsibility |
|---|---|
| `src/core/bootloader.rs` | `BootloaderType` enum + per-type `detect_active_bootloader` / `check_state` / `apply_{uki,grub,sdb,limine}`, plus pure-string parsers `grub_add_param` / `sdb_add_param` / `limine_add_param` for v9+ and legacy Limine syntax |
| `src/core/wayland.rs` | `/etc/profile.d/99-nvidia-wayland.sh` + `/etc/mkinitcpio.conf.d/nvidia-modules.conf`; calls `prime::apply` for hybrid hosts |
| `src/core/power.rs` | `options nvidia NVreg_*` modprobe drop-in (laptop-tuned), nouveau blacklist drop-in, `systemctl enable` of nvidia-suspend/hibernate/resume |
| `src/core/gaming.rs` | `[multilib]` uncommenter, `/etc/sysctl.d/99-gaming.conf` (`vm.max_map_count=1048576`), GPU-aware package resolver (`resolve_gaming_packages` + `resolve_aur_packages`) |
| `src/core/aur.rs` | `detect_helper`, `invoking_user` (SUDO_USER + PKEXEC_UID + getent, with username allowlist), `ensure_yay` (manual bootstrap), `install_aur_packages` (askpass routing, env preservation), `detect_askpass` (per-DE binary map) |
| `src/core/gpu.rs` | `lspci -mm -nn -D` parser, `GpuVendor` / `NvidiaGeneration` classification, `GpuInventory::{has_nvidia, is_hybrid, …}`, NVIDIA driver package recommendation per arch |
| `src/core/hardware.rs` | SMBIOS `chassis_type` → `FormFactor::{Laptop, Desktop, Unknown}` |
| `src/core/diagnostics.rs` | 14-point read-only scanner, `Finding{severity,title,detail,fix_hint}` |
| `src/core/auto.rs` | `recommend(ctx, form, gpus) -> Actions` — cross-references hardware applicability AND `check_state().is_unapplied()`; used both at GUI startup and by the Auto-Optimize hero button |
| `src/core/state.rs` | `TweakState::{Applied, Unapplied, Incompatible}` + `is_*` helpers |
| `src/core/prime.rs` | Xorg `OutputClass` drop-in for hybrid GPUs (skipped when nvidia-utils' shipped version is present) |
| `src/utils/process.rs` | `run_streaming(cmd, on_line)` — pipes stdout+stderr through mpsc into a `FnMut(&str)` |
| `src/utils/fs_helper.rs` | `atomic_write`, `backup_to_dir`, `write_dropin`, `ChangeReport::{AlreadyApplied, Applied, Planned}` |
| `src/cli.rs` | clap-derived CLI; dispatches into `core::run_actions` with a `println!` progress sink |
| `src/gui.rs` | Slint window wiring; `populate_detection` polls every `check_state`; `append_log` updates `log-text` + invokes `scroll_console_to_end` via `invoke_from_event_loop` for deferred snap |
| `ui/main_window.slint` | Dark Libadwaita-style UI: `Card` / `HeroButton` / `SecondaryButton` / `StateBadge` / `TweakRow` components, reactive console auto-scroll |
| `packaging/PKGBUILD` | Builds directly from `$startdir/..` (project root); never writes into `$srcdir` |

---

## Tests

```
cargo test --release --locked
```

**81 passing tests** covering:

- GPU parsing (Intel Arc, NVIDIA discrete, 3D-controller class, non-display skipping)
- NVIDIA generation classification across 14 real device IDs
- Package resolution (NVIDIA-only, Intel-only, AMD, hybrid, Pascal/Maxwell/Fermi AUR routing)
- Multilib detection + uncommenting with state-machine edge cases
- Bootloader detection for all four types with UKI+SDB coexistence
- `grub_add_param` / `sdb_add_param` / `limine_add_param` quoted/unquoted/empty/idempotent/commented-line cases
- `check_state` transitions (Applied / Unapplied / Incompatible) for each module
- Auto-Optimize recommendation rules under 5 hardware scenarios
- `atomic_write`, `backup_to_dir`, `write_dropin` (dry-run, idempotence, backup-on-replace)
- Subprocess streaming captures interleaved stdout/stderr and propagates non-zero exit

---

## Dependencies (runtime)

- `pciutils` — `lspci`
- `polkit` — `pkexec` launcher
- Slint's Linux graphics stack: `fontconfig`, `freetype2`, `libxkbcommon`, `wayland`, `libx11`, `libxcb`, `libxcursor`, `libxi`, `libxrandr`, `mesa`

### optdepends

- `nvidia-utils`, `nvidia-prime` — runtime userspace + `prime-run`
- `gamemode`, `mangohud` — auto-installed by `--apply-gaming`
- `yay` — auto-bootstrapped by `--apply-gaming` when an AUR driver is required
- `sbctl` — Secure Boot signing (via its own mkinitcpio post-hook; this tool doesn't sign on its own)
- `ksshaskpass` (KDE/Wayland) *or* `seahorse` (GNOME) — graphical askpass for yay's inner sudo when running from GUI

---

## Safety notes

- Every mutating write is preceded by a timestamped backup in `/var/backups/arch-nvidia-tweaker/`
- Every write is atomic (temp file + fsync + rename)
- `--dry-run` executes the full detection + planning pipeline without touching disk or running privileged commands
- NVIDIA-specific actions (Wayland / Bootloader / Power) are hard-gated on actual NVIDIA GPU presence — on non-NVIDIA hosts they cleanly report "skipped" instead of doing damage
- The tool never runs `yay` as root (makepkg refuses anyway). The bootstrap uses `install(1)` to chown a `/tmp` directory to the invoking user, then `sudo -u <user>` for `git clone` + `makepkg`, finishing with `pacman -U` as root. No user sudo prompt is required for the yay bootstrap itself.
- Usernames read from `$SUDO_USER` / `PKEXEC_UID` are validated against a conservative POSIX allowlist (`[A-Za-z0-9_-]{1,32}`) before being interpolated into any path or subprocess

---

## License

Dual-licensed under **MIT OR Apache-2.0** (same as most Rust crates).
