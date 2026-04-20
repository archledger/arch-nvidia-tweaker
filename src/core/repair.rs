// ═══════════════════════════════════════════════════════════════════════════════════════════
// Phase 20: Self-Heal on Upgrade
// ═══════════════════════════════════════════════════════════════════════════════════════════
//
// Pre-Phase-18/19 archgpu releases wrote config that is actively harmful on certain
// hardware topologies. Phase 18 and 19 GATED those writes from recurring, but they did
// nothing about legacy artifacts already on disk. Users who ran an old archgpu, then
// upgraded to Phase 19, found their systems still in the broken state — and `--diagnose`
// told them so without offering a repair.
//
// This module closes that gap. It is a dedicated, safe, idempotent repair pass that:
//
//   1. Deletes `/etc/X11/xorg.conf.d/10-archgpu-prime.conf` on desktop hybrids. That file
//      carries `Option "PrimaryGPU" "no"` on the NVIDIA OutputClass — correct for Optimus
//      laptops (iGPU drives display, dGPU offloads) but catastrophic on a desktop where
//      the monitor is physically plugged into the NVIDIA card. Deletion is backed up.
//
//   2. Removes the `nvidia-prime` package on desktop hybrids. `prime-run` offload serves
//      no purpose when the physical cable dictates the primary GPU; the package bloats
//      pacman's query surface and misleads future detection. `pacman -Rns` with
//      `--noconfirm` gated by the caller's `assume_yes`.
//
//   3. Forces a DKMS rebuild of the NVIDIA module when the running kernel has no nvidia
//      module loaded despite `nvidia-open-dkms` / `nvidia-dkms` being installed. That is
//      the exact symptom of the Phase-18 silent-failure bug (headers were missing when
//      the package was first installed → DKMS never built → nouveau blacklisted →
//      llvmpipe fallback). `dkms autoinstall` is idempotent: on a healthy host it's a
//      no-op, so re-running the repair action is safe.
//
// The scanner is split into a pure `scan_from_state` taking all environmental probes as
// inputs, and a thin runtime `scan` that collects those probes from the real system. This
// keeps every RepairAction branch unit-testable without needing a real pacman install or
// a live /sys/module tree.
//
// Wiring: `core::run_actions` dispatches `repair::apply` BEFORE gaming/power/wayland so
// the state those modules probe via their own `check_state()` routines is already
// cleaned up when they run.
// ═══════════════════════════════════════════════════════════════════════════════════════════

use anyhow::{Context as _, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::core::gpu::GpuInventory;
use crate::core::hardware::FormFactor;
use crate::core::state::TweakState;
use crate::core::{Context, ExecutionMode};
use crate::utils::fs_helper::{backup_to_dir, ChangeReport};
use crate::utils::process::run_streaming;

/// Name of the legacy PRIME Xorg OutputClass drop-in pre-Phase-19 archgpu wrote on every
/// hybrid host regardless of form factor. Living under `xorg_d` in SystemPaths.
const LEGACY_PRIME_DROPIN_FILE: &str = "10-archgpu-prime.conf";

/// Package name that should only exist on laptop-hybrid hosts under Phase 19 semantics.
const NVIDIA_PRIME_PACKAGE: &str = "nvidia-prime";

/// NVIDIA driver packages that register DKMS modules. If any of these is in pacman's
/// installed set AND `/sys/module/nvidia/` doesn't exist on the running kernel, DKMS
/// silently failed and a rebuild is needed.
///
/// Phase 24: added `nvidia-580xx-dkms` (AUR) for Maxwell/Volta/Pascal. Kept
/// `nvidia-dkms` in the list even though that package is gone from Arch's `extra` —
/// some users still have it installed from pre-Phase-24 archgpu runs and we want to
/// detect the stale DKMS module situation there too.
const NVIDIA_DKMS_PACKAGES: &[&str] = &[
    "nvidia-open-dkms",
    "nvidia-dkms",
    "nvidia-580xx-dkms",
    "nvidia-470xx-dkms",
    "nvidia-390xx-dkms",
];

/// A concrete remediation the self-heal pass can perform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairAction {
    /// Delete a stale PRIME OutputClass Xorg drop-in that pre-Phase-19 archgpu wrote on
    /// a desktop-hybrid host. Deletion is backed up under `ctx.paths.backup_dir`.
    DeleteStalePrimeDropin { path: PathBuf },

    /// Remove the `nvidia-prime` package via `pacman -Rns nvidia-prime`. Only surfaced
    /// when form == Desktop AND the package is installed. Reversible by `pacman -S nvidia-prime`.
    RemoveNvidiaPrimeOnDesktop,

    /// Force DKMS to rebuild the NVIDIA module(s) via `dkms autoinstall`. Idempotent
    /// (no-op on a healthy host). Required when a pre-Phase-18 archgpu installed a
    /// `*-dkms` driver without kernel headers and the module never built.
    ForceDkmsRebuild,

    /// Phase 22: `nomodeset` is on /proc/cmdline. Hard-forces software rendering — blocks
    /// DRM modesetting for every GPU driver. Delegates to `bootloader::apply_remove`
    /// which strips the token from the active bootloader's cmdline source
    /// (GRUB / systemd-boot / Limine / UKI) and regenerates.
    RemoveNomodesetFromCmdline,
}

impl RepairAction {
    /// Single-line human-readable summary for CLI reports and GUI warning banner.
    pub fn human_summary(&self) -> String {
        match self {
            Self::DeleteStalePrimeDropin { path } => format!(
                "Delete legacy PRIME OutputClass drop-in at {} (misroutes display on desktop hybrid)",
                path.display()
            ),
            Self::RemoveNvidiaPrimeOnDesktop => {
                "Remove `nvidia-prime` package (not needed on desktop hybrid — physical cable dictates primary GPU)"
                    .into()
            }
            Self::ForceDkmsRebuild => {
                "Force DKMS rebuild of the NVIDIA kernel module (running kernel has no nvidia module despite *-dkms driver installed)"
                    .into()
            }
            Self::RemoveNomodesetFromCmdline => {
                "Remove `nomodeset` from the kernel command line (hard-locks software rendering — blocks DRM modesetting for every GPU driver)"
                    .into()
            }
        }
    }
}

/// Pure scanner: given every environmental signal as inputs, return the list of
/// RepairActions that apply. No filesystem or process access. All unit-testable.
///
/// Parameters:
///  - `form`: detected chassis form factor (Laptop / Desktop / Unknown).
///  - `is_hybrid`: GpuInventory::is_hybrid() — NVIDIA + non-NVIDIA iGPU present.
///  - `legacy_prime_dropin`: Some(path) if `/etc/X11/xorg.conf.d/10-archgpu-prime.conf` exists.
///  - `nvidia_prime_installed`: whether `nvidia-prime` appears in `pacman -Qq`.
///  - `nvidia_dkms_package_installed`: whether ANY `*-dkms` NVIDIA driver is in `pacman -Qq`.
///  - `nvidia_module_loaded`: whether `/sys/module/nvidia/` exists (module present in running kernel).
///  - `nomodeset_in_cmdline`: whether `nomodeset` is on `/proc/cmdline` (Phase 22).
pub fn scan_from_state(
    form: FormFactor,
    is_hybrid: bool,
    legacy_prime_dropin: Option<PathBuf>,
    nvidia_prime_installed: bool,
    nvidia_dkms_package_installed: bool,
    nvidia_module_loaded: bool,
    nomodeset_in_cmdline: bool,
) -> Vec<RepairAction> {
    let mut out = Vec::new();

    // Rule 1: legacy PRIME drop-in on a non-laptop. Unknown form factor is treated as
    // "not a laptop" — same conservative default Phase 19 uses for prime::apply.
    if form != FormFactor::Laptop {
        if let Some(path) = legacy_prime_dropin {
            out.push(RepairAction::DeleteStalePrimeDropin { path });
        }
    }

    // Rule 2: nvidia-prime installed on a desktop hybrid. Non-hybrid desktops with
    // nvidia-prime are weird but benign (maybe the user has plans); only act when we
    // have a concrete mismatch (desktop + hybrid topology where prime-run is useless).
    if form == FormFactor::Desktop && is_hybrid && nvidia_prime_installed {
        out.push(RepairAction::RemoveNvidiaPrimeOnDesktop);
    }

    // Rule 3: DKMS silent failure — driver installed but module not loaded. The running
    // kernel is the authoritative signal: if `/sys/module/nvidia/` is absent while a
    // `*-dkms` driver is installed, DKMS either never built or never loaded. A rebuild
    // is harmless on healthy hosts (autoinstall is idempotent) and fixes this class.
    if nvidia_dkms_package_installed && !nvidia_module_loaded {
        out.push(RepairAction::ForceDkmsRebuild);
    }

    // Rule 4 (Phase 22): `nomodeset` on the live kernel cmdline. Universal remediation —
    // no GPU or form factor gate: `nomodeset` forces llvmpipe for EVERYONE, regardless
    // of hardware. Dispatch to bootloader::apply_remove.
    if nomodeset_in_cmdline {
        out.push(RepairAction::RemoveNomodesetFromCmdline);
    }

    out
}

/// Runtime scanner: collects the environmental signals and delegates to
/// `scan_from_state`. Safe to call from CLI `--diagnose` and from GUI's read-only
/// detection pass — no mutation.
pub fn scan(ctx: &Context, gpus: &GpuInventory, form: FormFactor) -> Vec<RepairAction> {
    let legacy_dropin = {
        let p = ctx.paths.xorg_d.join(LEGACY_PRIME_DROPIN_FILE);
        if p.exists() {
            Some(p)
        } else {
            None
        }
    };
    let installed = pacman_installed_packages();
    let nvidia_prime_installed = installed.contains(NVIDIA_PRIME_PACKAGE);
    let nvidia_dkms_installed = NVIDIA_DKMS_PACKAGES
        .iter()
        .any(|p| installed.contains(*p));
    let nvidia_module_loaded = ctx.paths.sys_module.join("nvidia").exists();
    // Phase 22: probe /proc/cmdline for the `nomodeset` token.
    let nomodeset = crate::core::rendering::check_nomodeset_in_cmdline(&ctx.paths.proc_cmdline);

    scan_from_state(
        form,
        gpus.is_hybrid(),
        legacy_dropin,
        nvidia_prime_installed,
        nvidia_dkms_installed,
        nvidia_module_loaded,
        nomodeset,
    )
}

/// Apply every repair the scanner finds. Idempotent: on a clean host `scan` returns
/// empty and this returns `Vec::with_capacity(0)`.
pub fn apply(
    ctx: &Context,
    gpus: &GpuInventory,
    form: FormFactor,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ChangeReport>> {
    let actions = scan(ctx, gpus, form);
    let mut reports = Vec::with_capacity(actions.len());
    for action in actions {
        reports.push(apply_one(ctx, &action, assume_yes, progress)?);
    }
    Ok(reports)
}

fn apply_one(
    ctx: &Context,
    action: &RepairAction,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<ChangeReport> {
    match action {
        RepairAction::DeleteStalePrimeDropin { path } => delete_with_backup(ctx, path),
        RepairAction::RemoveNvidiaPrimeOnDesktop => {
            run_pacman_remove(ctx, NVIDIA_PRIME_PACKAGE, assume_yes, progress)
        }
        RepairAction::ForceDkmsRebuild => run_dkms_autoinstall(ctx, progress),
        RepairAction::RemoveNomodesetFromCmdline => {
            crate::core::bootloader::apply_remove(ctx, &["nomodeset"], progress)
        }
    }
}

fn delete_with_backup(ctx: &Context, path: &Path) -> Result<ChangeReport> {
    let detail = format!("delete legacy PRIME drop-in {}", path.display());
    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned { detail });
    }
    let backup = backup_to_dir(path, &ctx.paths.backup_dir)?;
    std::fs::remove_file(path)
        .with_context(|| format!("removing legacy PRIME drop-in {}", path.display()))?;
    Ok(ChangeReport::Applied { detail, backup })
}

fn run_pacman_remove(
    ctx: &Context,
    pkg: &str,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<ChangeReport> {
    let detail = format!("pacman -Rns {pkg}");
    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned { detail });
    }
    if matches!(ctx.mode, ExecutionMode::Apply) {
        let mut cmd = Command::new("pacman");
        cmd.arg("-Rns");
        if assume_yes {
            cmd.arg("--noconfirm");
        }
        cmd.arg(pkg);
        progress(&format!("[pacman] {detail}"));
        let status = run_streaming(cmd, |line| progress(&format!("[pacman] {line}")))?;
        if !status.success() {
            anyhow::bail!("pacman -Rns {pkg} exited with {status}");
        }
    }
    Ok(ChangeReport::Applied {
        detail,
        backup: None,
    })
}

fn run_dkms_autoinstall(
    ctx: &Context,
    progress: &mut dyn FnMut(&str),
) -> Result<ChangeReport> {
    // `dkms autoinstall` rebuilds every registered DKMS module for every installed
    // kernel that doesn't already have one built. Idempotent.
    let detail = "dkms autoinstall (rebuild NVIDIA module for every installed kernel)".to_string();
    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned { detail });
    }
    if matches!(ctx.mode, ExecutionMode::Apply) {
        let mut cmd = Command::new("dkms");
        cmd.arg("autoinstall");
        progress(&format!("[dkms] {detail}"));
        let status = run_streaming(cmd, |line| progress(&format!("[dkms] {line}")))?;
        if !status.success() {
            anyhow::bail!("dkms autoinstall exited with {status}");
        }
    }
    Ok(ChangeReport::Applied {
        detail,
        backup: None,
    })
}

/// check_state contract:
///   Active      — `scan` returns empty → nothing to repair.
///   Unapplied   — `scan` returns at least one RepairAction.
///   Incompatible — never (repair is universally applicable; on a clean host it's Active).
pub fn check_state(ctx: &Context, gpus: &GpuInventory, form: FormFactor) -> TweakState {
    if scan(ctx, gpus, form).is_empty() {
        TweakState::Active
    } else {
        TweakState::Unapplied
    }
}

fn pacman_installed_packages() -> HashSet<String> {
    let Ok(out) = Command::new("pacman").arg("-Qq").output() else {
        return HashSet::new();
    };
    if !out.status.success() {
        return HashSet::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::{GpuInfo, GpuVendor};
    use tempfile::tempdir;

    fn desktop_hybrid() -> GpuInventory {
        GpuInventory {
            gpus: vec![
                GpuInfo {
                    vendor: GpuVendor::Amd,
                    vendor_id: 0x1002,
                    device_id: 0x1638,
                    pci_address: "0000:0a:00.0".into(),
                    vendor_name: "AMD".into(),
                    product_name: "Cezanne iGPU".into(),
                    kernel_driver: None,
                    is_integrated: true,
                    nvidia_gen: None,
                },
                GpuInfo {
                    vendor: GpuVendor::Nvidia,
                    vendor_id: 0x10de,
                    device_id: 0x2504,
                    pci_address: "0000:01:00.0".into(),
                    vendor_name: "NVIDIA".into(),
                    product_name: "RTX 3060".into(),
                    kernel_driver: None,
                    is_integrated: false,
                    nvidia_gen: None,
                },
            ],
        }
    }

    fn laptop_hybrid() -> GpuInventory {
        GpuInventory {
            gpus: vec![
                GpuInfo {
                    vendor: GpuVendor::Intel,
                    vendor_id: 0x8086,
                    device_id: 0x3e9b,
                    pci_address: "0000:00:02.0".into(),
                    vendor_name: "Intel".into(),
                    product_name: "UHD 630".into(),
                    kernel_driver: None,
                    is_integrated: true,
                    nvidia_gen: None,
                },
                GpuInfo {
                    vendor: GpuVendor::Nvidia,
                    vendor_id: 0x10de,
                    device_id: 0x25a2,
                    pci_address: "0000:01:00.0".into(),
                    vendor_name: "NVIDIA".into(),
                    product_name: "RTX 3050M".into(),
                    kernel_driver: None,
                    is_integrated: false,
                    nvidia_gen: None,
                },
            ],
        }
    }

    // ── pure scanner ──────────────────────────────────────────────────────────────────────

    #[test]
    fn scan_clean_system_returns_empty() {
        let out = scan_from_state(FormFactor::Desktop, false, None, false, false, true, false);
        assert!(out.is_empty());
    }

    #[test]
    fn scan_desktop_with_stale_prime_dropin_flags_delete() {
        let stale = PathBuf::from("/etc/X11/xorg.conf.d/10-archgpu-prime.conf");
        let out = scan_from_state(
            FormFactor::Desktop,
            true,
            Some(stale.clone()),
            false,
            false,
            true,
            false,
        );
        assert!(out
            .iter()
            .any(|a| matches!(a, RepairAction::DeleteStalePrimeDropin { path } if path == &stale)));
    }

    #[test]
    fn scan_laptop_with_prime_dropin_is_legitimate() {
        // On a laptop hybrid, `/etc/X11/xorg.conf.d/10-archgpu-prime.conf` is the CORRECT
        // output of Phase-19 `prime::apply`. Scanner must NOT flag it for deletion.
        let legit = PathBuf::from("/etc/X11/xorg.conf.d/10-archgpu-prime.conf");
        let out = scan_from_state(
            FormFactor::Laptop,
            true,
            Some(legit),
            false,
            false,
            true,
            false,
        );
        assert!(out.is_empty(),
            "laptop-hybrid PRIME drop-in must not be flagged as repair target: {out:?}");
    }

    #[test]
    fn scan_unknown_form_factor_treats_prime_dropin_as_stale() {
        // Conservative: if chassis detection failed, we still don't know if we're on a
        // desktop or laptop. Match Phase 19's decision in `prime::apply`: treat Unknown
        // as "not a laptop" and flag the drop-in for deletion. Over-eager here is safer
        // than leaving a misrouting file in place.
        let path = PathBuf::from("/etc/X11/xorg.conf.d/10-archgpu-prime.conf");
        let out = scan_from_state(
            FormFactor::Unknown,
            true,
            Some(path.clone()),
            false,
            false,
            true,
            false,
        );
        assert!(out.iter().any(|a| matches!(a, RepairAction::DeleteStalePrimeDropin { .. })));
    }

    #[test]
    fn scan_desktop_hybrid_with_nvidia_prime_flags_removal() {
        let out = scan_from_state(FormFactor::Desktop, true, None, true, false, true, false);
        assert!(out.contains(&RepairAction::RemoveNvidiaPrimeOnDesktop));
    }

    #[test]
    fn scan_laptop_hybrid_with_nvidia_prime_is_legitimate() {
        // Laptop hybrid + nvidia-prime installed is the EXPECTED state (Phase 19 adds it).
        let out = scan_from_state(FormFactor::Laptop, true, None, true, false, true, false);
        assert!(!out.contains(&RepairAction::RemoveNvidiaPrimeOnDesktop));
    }

    #[test]
    fn scan_desktop_non_hybrid_with_nvidia_prime_is_left_alone() {
        // User has nvidia-prime on a non-hybrid desktop. Weird but benign. Don't force
        // a removal — the user may have plans. We only act on the concrete mismatch.
        let out = scan_from_state(FormFactor::Desktop, false, None, true, false, true, false);
        assert!(!out.contains(&RepairAction::RemoveNvidiaPrimeOnDesktop));
    }

    #[test]
    fn scan_dkms_driver_but_no_module_loaded_flags_rebuild() {
        // The Phase-18 silent-failure symptom: nvidia-open-dkms is installed but
        // /sys/module/nvidia/ doesn't exist — module never built. Rebuild is required.
        let out = scan_from_state(FormFactor::Desktop, false, None, false, true, false, false);
        assert!(out.contains(&RepairAction::ForceDkmsRebuild));
    }

    #[test]
    fn scan_dkms_driver_with_module_loaded_is_healthy() {
        // Module is loaded → DKMS built successfully → nothing to rebuild.
        let out = scan_from_state(FormFactor::Desktop, false, None, false, true, true, false);
        assert!(!out.contains(&RepairAction::ForceDkmsRebuild));
    }

    #[test]
    fn scan_no_dkms_driver_no_rebuild_even_if_module_absent() {
        // No DKMS driver installed (e.g. AMD-only host). No rebuild to trigger — the
        // absence of /sys/module/nvidia/ is the correct state on this host.
        let out = scan_from_state(FormFactor::Desktop, false, None, false, false, false, false);
        assert!(!out.contains(&RepairAction::ForceDkmsRebuild));
    }

    #[test]
    fn scan_composite_rtx_3060_field_test_topology() {
        // The full bad state from the field-test host: Ryzen 7 5700G + RTX 3060 running
        // a pre-Phase-18/19 archgpu. Every repair rule fires at once.
        let path = PathBuf::from("/etc/X11/xorg.conf.d/10-archgpu-prime.conf");
        let out = scan_from_state(
            FormFactor::Desktop,
            true,
            Some(path),
            true,
            true,
            false,
            false,
        );
        assert_eq!(out.len(), 3, "expected all three repairs to fire, got: {out:?}");
    }

    // ── Phase 22: nomodeset repair ────────────────────────────────────────────────────────

    #[test]
    fn scan_nomodeset_flags_removal_regardless_of_gpu_or_chassis() {
        // `nomodeset` forces llvmpipe for EVERYONE — universal remediation, no gate.
        // Check it fires on all relevant combinations.
        for (form, is_hybrid) in [
            (FormFactor::Desktop, false),
            (FormFactor::Desktop, true),
            (FormFactor::Laptop, false),
            (FormFactor::Laptop, true),
            (FormFactor::Unknown, false),
        ] {
            let out = scan_from_state(form, is_hybrid, None, false, false, true, true);
            assert!(
                out.contains(&RepairAction::RemoveNomodesetFromCmdline),
                "nomodeset repair must fire on form={form:?} is_hybrid={is_hybrid}"
            );
        }
    }

    #[test]
    fn scan_nomodeset_absent_does_not_flag_removal() {
        let out = scan_from_state(FormFactor::Desktop, false, None, false, false, true, false);
        assert!(!out.contains(&RepairAction::RemoveNomodesetFromCmdline));
    }

    #[test]
    fn apply_nomodeset_removal_plans_uki_rewrite_in_dry_run() {
        // Full integration path: apply() → scan() → detects nomodeset in proc/cmdline →
        // dispatches to bootloader::apply_remove which strips it from /etc/kernel/cmdline.
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        // Seed the UKI cmdline source so bootloader detects UKI as the active
        // bootloader, and seed proc/cmdline so rendering::check_nomodeset_in_cmdline
        // returns true.
        std::fs::create_dir_all(ctx.paths.kernel_cmdline.parent().unwrap()).unwrap();
        std::fs::write(&ctx.paths.kernel_cmdline, "rw nomodeset quiet\n").unwrap();
        std::fs::create_dir_all(ctx.paths.proc_cmdline.parent().unwrap()).unwrap();
        std::fs::write(&ctx.paths.proc_cmdline, "rw nomodeset quiet\n").unwrap();
        // Seed sys/module/nvidia to silence the DKMS-rebuild branch on hosts that
        // have nvidia-open-dkms installed.
        std::fs::create_dir_all(ctx.paths.sys_module.join("nvidia")).unwrap();
        let no_gpus = GpuInventory { gpus: vec![] };

        let reports = apply(&ctx, &no_gpus, FormFactor::Desktop, false, &mut |_| {}).unwrap();
        let plan = reports
            .iter()
            .find(|r| matches!(r, ChangeReport::Planned { detail } if detail.contains("remove nomodeset")))
            .unwrap_or_else(|| panic!("nomodeset removal plan missing: {reports:#?}"));
        assert!(matches!(plan, ChangeReport::Planned { .. }));
    }

    // ── runtime apply paths ───────────────────────────────────────────────────────────────

    #[test]
    fn apply_dry_run_plans_delete_without_touching_fs() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        std::fs::create_dir_all(&ctx.paths.xorg_d).unwrap();
        let target = ctx.paths.xorg_d.join(LEGACY_PRIME_DROPIN_FILE);
        std::fs::write(&target, "# legacy bad drop-in\n").unwrap();

        let reports =
            apply(&ctx, &desktop_hybrid(), FormFactor::Desktop, false, &mut |_| {}).unwrap();

        assert!(
            !reports.is_empty(),
            "apply() must plan something on a dirty system"
        );
        assert!(
            reports
                .iter()
                .any(|r| matches!(r, ChangeReport::Planned { detail } if detail.contains("delete legacy PRIME"))),
            "dry-run must emit a Planned entry for the delete: {reports:#?}"
        );
        // File must still exist after dry-run.
        assert!(target.exists(), "dry-run must never mutate the filesystem");
    }

    #[test]
    fn apply_deletes_stale_dropin_with_backup_in_apply_mode() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        std::fs::create_dir_all(&ctx.paths.xorg_d).unwrap();
        let target = ctx.paths.xorg_d.join(LEGACY_PRIME_DROPIN_FILE);
        std::fs::write(&target, "LEGACY-MARKER\n").unwrap();

        // Isolate the delete path from the rest of the scanner. On a live Arch dev host
        // `apply()` calls `scan()` which queries REAL pacman (`pacman -Qq`) and the REAL
        // `/sys/module/nvidia` tree — meaning:
        //   - if nvidia-open-dkms is installed on the host AND the test's tempdir-rooted
        //     sys_module has no `nvidia/` dir, the scanner fires `ForceDkmsRebuild`, which
        //     runs real `dkms autoinstall` under `makepkg --check` (non-root, fails with
        //     exit 1, panics the test)
        //   - if `nvidia-prime` happens to be installed, `RemoveNvidiaPrimeOnDesktop`
        //     fires and runs real `pacman -Rns` (also fails non-root)
        // Neither of those are what this test is about. Seed `/sys/module/nvidia` to
        // suppress the DKMS rule, and pass an empty GpuInventory so `is_hybrid == false`
        // suppresses the nvidia-prime rule. What remains is the legacy-drop-in deletion,
        // which is the only thing this test covers.
        std::fs::create_dir_all(ctx.paths.sys_module.join("nvidia")).unwrap();
        let no_gpus = GpuInventory { gpus: vec![] };

        let reports = apply(&ctx, &no_gpus, FormFactor::Desktop, false, &mut |_| {}).unwrap();

        let delete = reports
            .iter()
            .find(|r| matches!(r, ChangeReport::Applied { detail, .. } if detail.contains("delete legacy PRIME")))
            .expect("Applied report for the delete is missing");
        let ChangeReport::Applied { backup, .. } = delete else {
            unreachable!();
        };
        let backup = backup.as_ref().expect("delete should produce a backup path");
        assert!(backup.exists());
        assert_eq!(std::fs::read_to_string(backup).unwrap(), "LEGACY-MARKER\n");
        assert!(!target.exists(), "target must be gone after apply");
    }

    #[test]
    fn apply_is_idempotent_on_clean_system() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        std::fs::create_dir_all(&ctx.paths.xorg_d).unwrap();
        std::fs::create_dir_all(ctx.paths.sys_module.join("nvidia")).unwrap();

        let reports =
            apply(&ctx, &laptop_hybrid(), FormFactor::Laptop, false, &mut |_| {}).unwrap();
        assert!(
            reports.is_empty(),
            "clean system must produce zero reports, got: {reports:#?}"
        );

        // Running again is still a no-op — no state was created to undo.
        let reports2 =
            apply(&ctx, &laptop_hybrid(), FormFactor::Laptop, false, &mut |_| {}).unwrap();
        assert!(reports2.is_empty());
    }

    #[test]
    fn check_state_active_on_clean_system() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        std::fs::create_dir_all(&ctx.paths.xorg_d).unwrap();
        // Seed /sys/module/nvidia so the DKMS-rebuild rule doesn't fire even if the
        // dev host happens to have a *-dkms package installed.
        std::fs::create_dir_all(ctx.paths.sys_module.join("nvidia")).unwrap();
        assert_eq!(
            check_state(&ctx, &laptop_hybrid(), FormFactor::Laptop),
            TweakState::Active
        );
    }

    #[test]
    fn check_state_unapplied_when_stale_dropin_present() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        std::fs::create_dir_all(&ctx.paths.xorg_d).unwrap();
        std::fs::create_dir_all(ctx.paths.sys_module.join("nvidia")).unwrap();
        std::fs::write(
            ctx.paths.xorg_d.join(LEGACY_PRIME_DROPIN_FILE),
            "# stale\n",
        )
        .unwrap();
        assert_eq!(
            check_state(&ctx, &desktop_hybrid(), FormFactor::Desktop),
            TweakState::Unapplied
        );
    }

    #[test]
    fn human_summary_lines_are_non_empty() {
        // Regression guard — the GUI banner calls these directly, an empty string would
        // render a zero-height bullet. Ensure every branch returns prose.
        for action in [
            RepairAction::DeleteStalePrimeDropin {
                path: PathBuf::from("/etc/X11/xorg.conf.d/10-archgpu-prime.conf"),
            },
            RepairAction::RemoveNvidiaPrimeOnDesktop,
            RepairAction::ForceDkmsRebuild,
            RepairAction::RemoveNomodesetFromCmdline,
        ] {
            let s = action.human_summary();
            assert!(!s.is_empty(), "summary empty for {action:?}");
            assert!(s.len() > 20, "summary too short for {action:?}: {s}");
        }
    }
}
