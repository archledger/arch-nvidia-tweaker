//! Phase 28 — reverse cleanup. Removes packages the host doesn't need:
//!   1. **Hardware-absent** vendor packages (NVIDIA stack on Intel-only host, etc.)
//!   2. **Legacy/deprecated** Xorg DDX drivers (`xf86-video-{intel,amdgpu,ati}`),
//!      pre-NEO Intel OpenCL (`beignet`), `bumblebee`, ancient AMD Catalyst
//!      (`catalyst`/`fglrx`), `mesa-amber` on modern hosts.
//!   3. **Defunct** packages that no longer exist on Arch but linger from old
//!      installs (`mesa-vdpau`, `libva-mesa-driver` and their lib32 variants —
//!      Mesa 26 bundles VA-API/VDPAU in-tree).
//!   4. **Conflicting choices**: `amdvlk`/`lib32-amdvlk` removed when
//!      `vulkan-radeon` is also installed (RADV is preferred per Arch Wiki and
//!      having both ICDs lets the wrong one win unpredictably).
//!
//! ## Safety design
//!
//! - **Opt-in only.** `Actions::all()` does NOT include cleanup; `auto::recommend`
//!   does NOT recommend it. Users invoke explicitly via `--apply-cleanup` or
//!   the GUI's two-click Preview→Confirm flow (Phase 30).
//! - **Pre-removal snapshot.** Before any `pacman -Rns` call, the full
//!   `pacman -Qq` output is written to `<backup_dir>/pre-cleanup-<unix-ts>.txt`.
//!   If the user disagrees with anything we removed, they can restore the exact
//!   pre-cleanup package set with:
//!   `sudo pacman -S --needed - < /var/backups/archgpu/pre-cleanup-<ts>.txt`
//! - **NEVER_REMOVE allowlist.** Hard-coded list of packages we refuse to touch
//!   even if some pathological future scenario somehow added them to a candidate
//!   list (kernel, base, bootloader, systemd, etc.). Defense-in-depth.
//! - **Dry-run respected.** With `ctx.mode == DryRun`, the plan is computed and
//!   reported but no `pacman` call is spawned.
//!
//! ## Why "hardware-absent vendor packages"
//!
//! A user who buys a new Intel laptop after years of NVIDIA workstations may end
//! up with `nvidia-utils`, `lib32-nvidia-utils`, `nvidia-settings`, etc. still
//! installed — they take up disk space, keep DKMS busy on every kernel update,
//! and (in NVIDIA's case) install `/etc/X11/xorg.conf.d/10-nvidia-drm-outputclass.conf`
//! which does the wrong thing on a non-NVIDIA host. Removing them when no NVIDIA
//! GPU is present is the correct convergence state.
//!
//! ## Important non-removals
//!
//! We deliberately do NOT remove:
//! - `linux-firmware` (kept in case the user adds an NVIDIA-firmware-needing
//!   peripheral later — it's also pulled in by base anyway).
//! - `vulkan-icd-loader` / `lib32-vulkan-icd-loader` (cross-vendor, essentials).
//! - `mesa` / `lib32-mesa` (cross-vendor, essentials).
//! - Anything matching a `*-headers` pattern (DKMS may need it).

use anyhow::{Context as _, Result};
use std::collections::HashSet;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::gpu::GpuInventory;
use crate::core::state::TweakState;
use crate::core::{Context, ExecutionMode};
use crate::utils::fs_helper::{atomic_write, ChangeReport};
use crate::utils::process::run_streaming;

/// Hard-coded refusal list. Even if a future bug (or a contributor's typo) ever
/// adds one of these names to a candidate list, this filter stops the removal.
/// Categories: kernel + base + critical userspace + dynamic linker chain.
pub const NEVER_REMOVE: &[&str] = &[
    "linux",
    "linux-lts",
    "linux-zen",
    "linux-hardened",
    "linux-rt",
    "linux-rt-lts",
    "linux-firmware",
    "base",
    "base-devel",
    "systemd",
    "udev",
    "glibc",
    "lib32-glibc",
    "mesa",
    "lib32-mesa",
    "vulkan-icd-loader",
    "lib32-vulkan-icd-loader",
    "pacman",
    "grub",
    "systemd-boot",
    "limine",
    "efibootmgr",
    "mkinitcpio",
    "dracut",
    "sbctl",
    "amd-ucode",
    "intel-ucode",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemovalCategory {
    /// Vendor-specific package on a host that lacks that vendor's GPU.
    HardwareAbsent,
    /// Upstream-deprecated or actively-broken on modern hardware.
    LegacyDeprecated,
    /// No longer exists on Arch (lingers from pre-2026 upgrades).
    DefunctPackage,
    /// Coexists with a preferred alternative.
    ConflictingChoice,
}

impl RemovalCategory {
    pub fn label(self) -> &'static str {
        match self {
            Self::HardwareAbsent => "hardware-absent",
            Self::LegacyDeprecated => "legacy/deprecated",
            Self::DefunctPackage => "defunct",
            Self::ConflictingChoice => "conflicting",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemovalCandidate {
    pub package: String,
    pub reason: String,
    pub category: RemovalCategory,
}

/// All package names this module ever considers — used to keep `pacman -Qq` queries
/// bounded and to seed unit tests with a realistic install set.
fn candidate_universe() -> Vec<&'static str> {
    let mut v = Vec::new();
    v.extend(NVIDIA_VENDOR_PACKAGES);
    v.extend(AMD_VENDOR_PACKAGES);
    v.extend(INTEL_VENDOR_PACKAGES);
    v.extend(LEGACY_PACKAGES);
    v.extend(DEFUNCT_PACKAGES);
    v.extend(AMDVLK_PACKAGES);
    v.extend(["vulkan-radeon"]); // referenced as "preferred alternative" check
    v
}

const NVIDIA_VENDOR_PACKAGES: &[&str] = &[
    "nvidia",
    "nvidia-dkms",
    "nvidia-open",
    "nvidia-open-dkms",
    "nvidia-580xx-dkms",
    "nvidia-470xx-dkms",
    "nvidia-390xx-dkms",
    "nvidia-utils",
    "lib32-nvidia-utils",
    "nvidia-580xx-utils",
    "lib32-nvidia-580xx-utils",
    "nvidia-470xx-utils",
    "lib32-nvidia-470xx-utils",
    "nvidia-390xx-utils",
    "lib32-nvidia-390xx-utils",
    "nvidia-settings",
    "nvidia-prime",
    "libva-nvidia-driver",
    "opencl-nvidia",
    "lib32-opencl-nvidia",
    "cuda",
    "cuda-tools",
];

const AMD_VENDOR_PACKAGES: &[&str] = &[
    "vulkan-radeon",
    "lib32-vulkan-radeon",
    "linux-firmware-amdgpu",
    "rocm-opencl-runtime",
    "rocm-hip-runtime",
    "rocm-opencl-sdk",
    "rocm-hip-sdk",
    "hip-runtime-amd",
    "miopen-hip",
    "python-pytorch-rocm",
];

const INTEL_VENDOR_PACKAGES: &[&str] = &[
    "vulkan-intel",
    "lib32-vulkan-intel",
    "linux-firmware-intel",
    "intel-media-driver",
    "libva-intel-driver",
    "intel-compute-runtime",
];

/// Always-stale on a modern host, regardless of which GPU is present.
const LEGACY_PACKAGES: &[&str] = &[
    "xf86-video-intel",  // wiki: actively hostile on Gen 11+; modesetting is the path
    "xf86-video-amdgpu", // legacy DDX; modesetting suffices on Wayland and modern Xorg
    "xf86-video-ati",    // ditto for radeon-class cards
    "mesa-amber",        // pre-R600 / Gen 2-4 only — shouldn't be on a modern install
    "bumblebee",         // legacy Optimus; superseded by nvidia-prime + DRM offload
    "primus",            // Optimus companion to bumblebee
    "beignet",           // Intel pre-NEO OpenCL, abandoned upstream
    "intel-opencl",      // older Intel OpenCL, abandoned upstream
    "catalyst",          // ancient AMD proprietary
    "fglrx",             // ancient AMD proprietary
    "amdapp-sdk",        // abandoned AMD APP SDK
];

/// Packages no longer published on Arch (2025+ Mesa / split-firmware era) that linger
/// only from out-of-date upgrade paths.
const DEFUNCT_PACKAGES: &[&str] = &[
    "mesa-vdpau",
    "lib32-mesa-vdpau",
    "libva-mesa-driver",
    "lib32-libva-mesa-driver",
];

/// AMDVLK pair — only flagged for removal when `vulkan-radeon` (RADV) is also installed.
const AMDVLK_PACKAGES: &[&str] = &["amdvlk", "lib32-amdvlk"];

/// Cleanup state probe. Currently unused by `auto::recommend` (Phase 28 invariant —
/// cleanup is opt-in only) but exposed for the Phase 30 GUI card, which uses it
/// to decide whether the Cleanup card shows "0 candidates / system already converged"
/// or "N packages can be removed — Preview".
#[allow(dead_code)]
pub fn check_state(ctx: &Context, gpus: &GpuInventory) -> TweakState {
    let _ = ctx;
    let Some(installed) = pacman_query_installed_set() else {
        // pacman unavailable — leave callable; apply will surface the real error.
        return TweakState::Unapplied;
    };
    let plan = compute_removal_plan(gpus, &installed);
    if plan.is_empty() {
        TweakState::Active
    } else {
        TweakState::Unapplied
    }
}

pub fn apply(
    ctx: &Context,
    gpus: &GpuInventory,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ChangeReport>> {
    let installed = pacman_query_installed_set()
        .context("querying pacman -Qq for currently installed packages")?;
    let plan = compute_removal_plan(gpus, &installed);
    if plan.is_empty() {
        return Ok(vec![ChangeReport::AlreadyApplied {
            detail: "no removal candidates — system already converged".into(),
        }]);
    }

    let mut reports = Vec::new();
    progress(&format!("[cleanup] {} package(s) to remove:", plan.len()));
    for c in &plan {
        progress(&format!(
            "[cleanup]   - {} ({}) — {}",
            c.package,
            c.category.label(),
            c.reason
        ));
    }

    if ctx.mode.is_dry_run() {
        return Ok(vec![ChangeReport::Planned {
            detail: format!(
                "would `pacman -Rns` {} package(s); pre-cleanup snapshot would be written to {}",
                plan.len(),
                ctx.paths.backup_dir.display()
            ),
        }]);
    }

    if matches!(ctx.mode, ExecutionMode::Apply) {
        // Snapshot the full installed set BEFORE we remove anything. The user can
        // restore the exact pre-cleanup state with:
        //   sudo pacman -S --needed - < <snapshot>
        let snapshot = write_pre_cleanup_snapshot(ctx, &installed)?;
        reports.push(ChangeReport::Applied {
            detail: format!("pre-cleanup snapshot written to {}", snapshot.display()),
            backup: Some(snapshot),
        });

        // Single batched pacman -Rns. -n drops configs, -s removes orphan deps.
        // -u would dry-run; we want the real thing.
        let pkgs: Vec<&str> = plan.iter().map(|c| c.package.as_str()).collect();
        let mut cmd = Command::new("pacman");
        cmd.args(["-Rns"]);
        if assume_yes {
            cmd.arg("--noconfirm");
        }
        for p in &pkgs {
            cmd.arg(p);
        }
        let detail = format!("pacman -Rns {}", pkgs.join(" "));
        progress(&format!("[pacman] {detail}"));
        let status = run_streaming(cmd, |line| progress(&format!("[pacman] {line}")))?;
        if !status.success() {
            anyhow::bail!("pacman -Rns exited with {status}");
        }
        reports.push(ChangeReport::Applied {
            detail,
            backup: None,
        });
    }

    Ok(reports)
}

/// Pure: walk the candidate universe, filter to what's installed, classify
/// against detected hardware. Output preserves discovery order: hardware-absent
/// first (most surprising), then defunct, then legacy, then conflicting.
pub fn compute_removal_plan(
    gpus: &GpuInventory,
    installed: &HashSet<String>,
) -> Vec<RemovalCandidate> {
    let mut out = Vec::new();
    let never: HashSet<&'static str> = NEVER_REMOVE.iter().copied().collect();
    let mut push = |pkg: &str, reason: String, cat: RemovalCategory| {
        if never.contains(pkg) {
            return; // defense-in-depth
        }
        if !installed.contains(pkg) {
            return;
        }
        if out.iter().any(|c: &RemovalCandidate| c.package == pkg) {
            return; // dedupe
        }
        out.push(RemovalCandidate {
            package: pkg.to_string(),
            reason,
            category: cat,
        });
    };

    // 1. Hardware-absent (vendor pkg installed but the vendor's GPU is missing).
    if !gpus.has_nvidia() {
        for p in NVIDIA_VENDOR_PACKAGES {
            push(
                p,
                "no NVIDIA GPU detected on this host".into(),
                RemovalCategory::HardwareAbsent,
            );
        }
    }
    if !gpus.has_amd() {
        for p in AMD_VENDOR_PACKAGES {
            push(
                p,
                "no AMD GPU detected on this host".into(),
                RemovalCategory::HardwareAbsent,
            );
        }
    }
    if !gpus.has_intel() {
        for p in INTEL_VENDOR_PACKAGES {
            push(
                p,
                "no Intel GPU detected on this host".into(),
                RemovalCategory::HardwareAbsent,
            );
        }
    }

    // 2. Defunct (the package literally no longer exists on Arch).
    for p in DEFUNCT_PACKAGES {
        push(
            p,
            "package no longer exists on Arch — Mesa 26 bundles VA-API/VDPAU in-tree".into(),
            RemovalCategory::DefunctPackage,
        );
    }

    // 3. Legacy / deprecated — context-aware reasons per package.
    for p in LEGACY_PACKAGES {
        let reason = match *p {
            "xf86-video-intel" => {
                "Arch Wiki: actively hostile on Gen 11+; xorg-server's `modesetting` is the modern path"
            }
            "xf86-video-amdgpu" => {
                "modesetting (in xorg-server) supersedes this DDX on Wayland and modern Xorg"
            }
            "xf86-video-ati" => "modesetting supersedes this DDX for radeon-class cards",
            "mesa-amber" => "legacy Mesa fork for pre-R600 / Gen 2-4 only — modern hardware uses mainline mesa",
            "bumblebee" | "primus" => {
                "legacy Optimus stack — superseded by nvidia-prime + DRM offload"
            }
            "beignet" | "intel-opencl" => {
                "abandoned upstream — Intel's compute story is intel-compute-runtime (NEO)"
            }
            "catalyst" | "fglrx" | "amdapp-sdk" => "abandoned proprietary AMD stack",
            _ => "legacy / deprecated",
        };
        push(p, reason.into(), RemovalCategory::LegacyDeprecated);
    }

    // 4. Conflicting choice — AMDVLK alongside RADV.
    if installed.contains("vulkan-radeon") {
        for p in AMDVLK_PACKAGES {
            push(
                p,
                "RADV (vulkan-radeon) is also installed; having two AMD Vulkan ICDs lets the \
                 wrong one win unpredictably (RADV is the Arch Wiki recommendation)"
                    .into(),
                RemovalCategory::ConflictingChoice,
            );
        }
    }

    out
}

fn pacman_query_installed_set() -> Option<HashSet<String>> {
    let out = Command::new("pacman").arg("-Qq").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let body = String::from_utf8_lossy(&out.stdout);
    let universe: HashSet<&str> = candidate_universe().into_iter().collect();
    Some(
        body.lines()
            .map(str::trim)
            .filter(|l| universe.contains(l))
            .map(str::to_string)
            .collect(),
    )
}

fn write_pre_cleanup_snapshot(
    ctx: &Context,
    installed: &HashSet<String>,
) -> Result<std::path::PathBuf> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = ctx.paths.backup_dir.join(format!("pre-cleanup-{ts}.txt"));
    // The snapshot is the FULL installed list — not just our candidate universe —
    // so it can be used as `pacman -S --needed - < <path>` to restore the host.
    // Re-run pacman -Qq here without filtering.
    let full = Command::new("pacman")
        .arg("-Qq")
        .output()
        .context("re-running pacman -Qq for snapshot")?;
    let body = if full.status.success() {
        String::from_utf8_lossy(&full.stdout).into_owned()
    } else {
        // Fallback to the filtered set we already have so we still write *something*.
        let mut sorted: Vec<&String> = installed.iter().collect();
        sorted.sort();
        sorted
            .into_iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    };
    atomic_write(&path, &body)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::{GpuInfo, GpuVendor, NvidiaGeneration};

    fn gpu(vendor: GpuVendor, device_id: u16) -> GpuInfo {
        GpuInfo {
            vendor,
            vendor_id: match vendor {
                GpuVendor::Nvidia => 0x10de,
                GpuVendor::Amd => 0x1002,
                GpuVendor::Intel => 0x8086,
                GpuVendor::Other => 0,
            },
            device_id,
            pci_address: "0000:00:02.0".into(),
            vendor_name: format!("{vendor:?}"),
            product_name: "test".into(),
            kernel_driver: None,
            is_integrated: true,
            nvidia_gen: if vendor == GpuVendor::Nvidia {
                Some(NvidiaGeneration::Ampere)
            } else {
                None
            },
        }
    }

    fn inv(gpus: Vec<GpuInfo>) -> GpuInventory {
        GpuInventory { gpus }
    }

    fn installed(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn intel_only_host_with_lingering_nvidia_stack_plans_removal() {
        let plan = compute_removal_plan(
            &inv(vec![gpu(GpuVendor::Intel, 0x64a0)]),
            &installed(&["nvidia-utils", "lib32-nvidia-utils", "nvidia-settings"]),
        );
        let names: Vec<&str> = plan.iter().map(|c| c.package.as_str()).collect();
        assert!(names.contains(&"nvidia-utils"));
        assert!(names.contains(&"lib32-nvidia-utils"));
        assert!(names.contains(&"nvidia-settings"));
        assert!(plan.iter().all(|c| c.category == RemovalCategory::HardwareAbsent));
    }

    #[test]
    fn nvidia_only_host_does_not_plan_removing_nvidia_stack() {
        let plan = compute_removal_plan(
            &inv(vec![gpu(GpuVendor::Nvidia, 0x2204)]),
            &installed(&["nvidia-utils", "lib32-nvidia-utils"]),
        );
        assert!(plan.is_empty(), "should not propose removing matching-vendor packages: {plan:?}");
    }

    #[test]
    fn defunct_packages_always_planned_regardless_of_vendor() {
        for inventory in [
            inv(vec![gpu(GpuVendor::Intel, 0x64a0)]),
            inv(vec![gpu(GpuVendor::Amd, 0x73bf)]),
            inv(vec![gpu(GpuVendor::Nvidia, 0x2204)]),
        ] {
            let plan = compute_removal_plan(
                &inventory,
                &installed(&["mesa-vdpau", "libva-mesa-driver"]),
            );
            let names: Vec<&str> = plan.iter().map(|c| c.package.as_str()).collect();
            assert!(names.contains(&"mesa-vdpau"));
            assert!(names.contains(&"libva-mesa-driver"));
        }
    }

    #[test]
    fn xf86_video_intel_always_legacy_planned() {
        let plan = compute_removal_plan(
            &inv(vec![gpu(GpuVendor::Intel, 0x64a0)]),
            &installed(&["xf86-video-intel"]),
        );
        let c = plan.iter().find(|c| c.package == "xf86-video-intel").unwrap();
        assert_eq!(c.category, RemovalCategory::LegacyDeprecated);
    }

    #[test]
    fn amdvlk_planned_only_when_vulkan_radeon_present() {
        let amd_inv = inv(vec![gpu(GpuVendor::Amd, 0x73bf)]);
        // Without RADV → user explicitly chose AMDVLK, leave it alone.
        let plan_alone = compute_removal_plan(&amd_inv, &installed(&["amdvlk"]));
        assert!(
            !plan_alone.iter().any(|c| c.package == "amdvlk"),
            "AMDVLK alone is a deliberate user choice — must not auto-remove"
        );
        // With RADV → conflict, propose removal.
        let plan_both = compute_removal_plan(
            &amd_inv,
            &installed(&["amdvlk", "lib32-amdvlk", "vulkan-radeon"]),
        );
        let amdvlk = plan_both.iter().find(|c| c.package == "amdvlk").unwrap();
        assert_eq!(amdvlk.category, RemovalCategory::ConflictingChoice);
    }

    #[test]
    fn never_remove_allowlist_blocks_kernel_and_base() {
        // Even if some bug ever queued these for removal, the filter must drop them.
        // We can't easily test the filter directly without poking at private state, so
        // instead assert that NONE of the candidate-universe lists contain a
        // NEVER_REMOVE name — that's the structural invariant.
        let never: HashSet<&'static str> = NEVER_REMOVE.iter().copied().collect();
        for cand in candidate_universe() {
            assert!(
                !never.contains(cand),
                "candidate `{cand}` overlaps NEVER_REMOVE — would be silently filtered, but the \
                 overlap itself indicates a list-construction bug"
            );
        }
    }

    #[test]
    fn empty_install_set_yields_empty_plan() {
        let plan = compute_removal_plan(
            &inv(vec![gpu(GpuVendor::Intel, 0x64a0)]),
            &installed(&[]),
        );
        assert!(plan.is_empty());
    }

    #[test]
    fn intel_compute_runtime_removed_on_amd_only_host() {
        let plan = compute_removal_plan(
            &inv(vec![gpu(GpuVendor::Amd, 0x73bf)]),
            &installed(&["intel-compute-runtime"]),
        );
        let c = plan
            .iter()
            .find(|c| c.package == "intel-compute-runtime")
            .unwrap();
        assert_eq!(c.category, RemovalCategory::HardwareAbsent);
    }

    #[test]
    fn rocm_removed_on_intel_only_host() {
        let plan = compute_removal_plan(
            &inv(vec![gpu(GpuVendor::Intel, 0x64a0)]),
            &installed(&["rocm-opencl-runtime", "rocm-hip-runtime"]),
        );
        assert!(plan.iter().any(|c| c.package == "rocm-opencl-runtime"));
        assert!(plan.iter().any(|c| c.package == "rocm-hip-runtime"));
    }

    #[test]
    fn hybrid_intel_plus_nvidia_keeps_both_vendor_stacks() {
        let plan = compute_removal_plan(
            &inv(vec![
                gpu(GpuVendor::Intel, 0x3e9b),
                gpu(GpuVendor::Nvidia, 0x25a2),
            ]),
            &installed(&[
                "nvidia-utils",
                "vulkan-intel",
                "intel-media-driver",
                "libva-nvidia-driver",
            ]),
        );
        // Hybrid host needs BOTH stacks — none of these should be flagged.
        let names: Vec<&str> = plan.iter().map(|c| c.package.as_str()).collect();
        assert!(!names.contains(&"nvidia-utils"));
        assert!(!names.contains(&"vulkan-intel"));
        assert!(!names.contains(&"intel-media-driver"));
        assert!(!names.contains(&"libva-nvidia-driver"));
    }

    #[test]
    fn deduplication_does_not_double_list_a_package() {
        // amdvlk lives in both AMDVLK_PACKAGES and is checked under conflicting-choice.
        // It must appear at most once in the final plan.
        let plan = compute_removal_plan(
            &inv(vec![gpu(GpuVendor::Amd, 0x73bf)]),
            &installed(&["amdvlk", "lib32-amdvlk", "vulkan-radeon"]),
        );
        let amdvlk_count = plan.iter().filter(|c| c.package == "amdvlk").count();
        assert_eq!(amdvlk_count, 1);
    }
}
