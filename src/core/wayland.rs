use anyhow::Result;
use std::path::PathBuf;

use crate::core::gpu::GpuInventory;
use crate::core::state::TweakState;
use crate::core::{prime, Context};
use crate::utils::fs_helper::{write_dropin, ChangeReport};

const PROFILE_D_FILE: &str = "99-nvidia-wayland.sh";
const MKINITCPIO_DROPIN_FILE: &str = "nvidia-modules.conf";

// Phase 16 modernization: the drop-in no longer exports GBM_BACKEND=nvidia-drm,
// __GLX_VENDOR_LIBRARY_NAME=nvidia, or LIBVA_DRIVER_NAME=nvidia globally. On modern
// NVIDIA 545+ / Mesa 23+, compositors auto-detect the right GBM/GL backend, and
// setting them GLOBALLY actively breaks hybrid (Optimus/PRIME) setups by forcing
// NVIDIA for all GL calls — defeating the purpose of `prime-run`. Per-app offload
// still works via `prime-run <cmd>`, which sets these locally. Users who ran an
// OLDER version of this tool and had those exports here will have the file
// overwritten with this modernized comment-only version on the next --apply-wayland.
const PROFILE_D_CONTENT: &str = "\
# Managed by archgpu — do not edit by hand.
#
# Historical note: this file used to export
#   GBM_BACKEND=nvidia-drm
#   LIBVA_DRIVER_NAME=nvidia
#   __GLX_VENDOR_LIBRARY_NAME=nvidia
#
# Those exports are no longer safe to set globally on NVIDIA 545+ / Mesa 23+. Modern
# compositors auto-detect the correct GBM/GL backend, and global __GLX_VENDOR_LIBRARY_NAME
# breaks hybrid (Optimus/PRIME) setups by forcing NVIDIA GL for every app — making
# prime-run useless.
#
# For per-application NVIDIA offload, use `prime-run <command>` (from nvidia-prime),
# which sets __NV_PRIME_RENDER_OFFLOAD=1 and __GLX_VENDOR_LIBRARY_NAME=nvidia for
# that invocation only.
";

// Phase 17 safety invariant (audit-verified):
//
//   1. We write to `/etc/mkinitcpio.conf.d/nvidia-modules.conf` — a DROP-IN file. We never
//      read, parse, or modify `/etc/mkinitcpio.conf` directly. The master config remains
//      exactly as the user (or pacman) left it; mkinitcpio merges our drop-in on top.
//   2. The drop-in uses `MODULES+=(...)`, which APPENDS to whatever MODULES the user
//      already set. Using `MODULES=(...)` in a drop-in would CLOBBER the user's list.
//   3. We never touch the `HOOKS=(...)` array. In particular we do NOT add or remove the
//      `kms` hook — removing it breaks modern simpledrm console handoff on systems using
//      the generic `simpledrm` early-framebuffer, which is the default on most Arch hosts.
//      Adding nvidia modules via MODULES is the correct path for early KMS without
//      disturbing the hook chain.
const MKINITCPIO_DROPIN_CONTENT: &str = "\
# Managed by archgpu — do not edit by hand.
MODULES+=(nvidia nvidia_modeset nvidia_uvm nvidia_drm)
";

/// Returns `Applied` only when BOTH the /etc/profile.d env drop-in AND the mkinitcpio modules
/// drop-in exist on disk AND the drop-in matches our current (modernized) content byte-for-byte.
/// If the file contains LEGACY exports (GBM_BACKEND=nvidia-drm, etc.), state is Unapplied so
/// the user can re-run --apply-wayland to refresh to the modern comment-only form.
pub fn check_state(ctx: &Context, gpus: &GpuInventory) -> TweakState {
    if !gpus.has_nvidia() {
        return TweakState::Incompatible;
    }
    let profile = ctx.paths.profile_d.join(PROFILE_D_FILE);
    let mkinit = ctx.paths.mkinitcpio_d.join(MKINITCPIO_DROPIN_FILE);
    if !mkinit.exists() {
        return TweakState::Unapplied;
    }
    let Ok(current) = std::fs::read_to_string(&profile) else {
        return TweakState::Unapplied;
    };
    if current == PROFILE_D_CONTENT {
        // File-only tweak — mkinitcpio drop-ins take effect on the next initramfs rebuild
        // (which `--apply-bootloader` triggers for UKI hosts via `mkinitcpio -P`). No
        // sensible kernel-level probe distinguishes "pre-reboot" from "post-reboot" here,
        // so return Active once the config matches.
        TweakState::Active
    } else {
        // File exists but contains legacy content (likely from an older version of this
        // tool). The next --apply-wayland rewrites it to the modern comment-only form.
        TweakState::Unapplied
    }
}

// ── Phase 16 Sanitation: hunt down legacy X11/Wayland configs ──────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaylandWarning {
    /// A legacy Xorg config file exists that may force X11 fallback or break modern Wayland.
    XorgConfLegacy(PathBuf),
    /// A deprecated Wayland/NVIDIA env var is exported from a `/etc/profile.d/*.sh` script
    /// that isn't owned by this tool. Examples: `WLR_NO_HARDWARE_CURSORS=1` (not needed
    /// since NVIDIA 545 + wlroots with hardware cursors), `GBM_BACKEND=nvidia-drm` (breaks
    /// hybrid setups by forcing NVIDIA GBM globally).
    LegacyEnvVar { path: PathBuf, var: String },
}

impl WaylandWarning {
    pub fn title(&self) -> String {
        match self {
            Self::XorgConfLegacy(p) => format!("Legacy X11 configuration at {}", p.display()),
            Self::LegacyEnvVar { path, var } => {
                format!("Deprecated Wayland env var `{var}` in {}", path.display())
            }
        }
    }

    pub fn detail(&self) -> String {
        match self {
            Self::XorgConfLegacy(p) => format!(
                "{}: manually authored xorg.conf drop-ins are the #1 source of broken Wayland \
                 sessions on NVIDIA. The nvidia-utils package ships its own modesetting-safe \
                 OutputClass config in /usr/share/X11/xorg.conf.d/.",
                p.display()
            ),
            Self::LegacyEnvVar { var, .. } => {
                if var.contains("WLR_NO_HARDWARE_CURSORS") {
                    "This was a 2023-era workaround for NVIDIA cursor corruption in wlroots. \
                     Modern NVIDIA drivers support hardware cursors correctly; keeping this set \
                     forces a software-cursor fallback that actually INCREASES latency on \
                     modern compositors using explicit sync."
                        .to_string()
                } else if var.contains("GBM_BACKEND") {
                    "Setting GBM_BACKEND=nvidia-drm globally forces every GL app (including the \
                     iGPU-bound ones) through the NVIDIA backend — which breaks hybrid-graphics \
                     setups by defeating prime-run. Mesa 23+ auto-detects the correct backend; \
                     this env var is no longer necessary."
                        .to_string()
                } else {
                    format!("`{var}` is legacy and no longer needed on modern drivers.")
                }
            }
        }
    }

    pub fn remediation(&self) -> String {
        match self {
            Self::XorgConfLegacy(p) => format!(
                "Rename or remove `{}` and restart your session. The shipped nvidia-utils \
                 OutputClass config handles Wayland + Xorg correctly.",
                p.display()
            ),
            Self::LegacyEnvVar { path, var } => format!(
                "Edit `{}` and delete the `{}` export; log out and back in. (Our own drop-in \
                 {PROFILE_D_FILE} is skipped by this check.)",
                path.display(),
                var
            ),
        }
    }
}

/// Scan the filesystem for legacy NVIDIA Xorg config drop-ins and deprecated Wayland/NVIDIA
/// env vars in `/etc/profile.d/*.sh` scripts we DON'T own.
pub fn sanitation_warnings(ctx: &Context) -> Vec<WaylandWarning> {
    let mut out = Vec::new();

    // X11 cruft — either a legacy /etc/X11/xorg.conf or a 20-nvidia.conf drop-in (both are
    // typical outputs of nvidia-xconfig / nvidia-settings from the pre-OutputClass era).
    let x11_candidates = [
        ctx.paths.etc_x11_xorg_conf.clone(),
        ctx.paths.xorg_d.join("20-nvidia.conf"),
    ];
    for p in x11_candidates {
        if p.exists() {
            out.push(WaylandWarning::XorgConfLegacy(p));
        }
    }

    // Legacy env vars — walk /etc/profile.d/*.sh excluding our own file, grep for deprecated
    // exports. We don't warn on our OWN file because its modernized form no longer contains
    // them; if it did, that'd be caught by check_state's byte-comparison check.
    if let Ok(rd) = std::fs::read_dir(&ctx.paths.profile_d) {
        for entry in rd.flatten() {
            let path = entry.path();
            // Skip non-shell files and our own drop-in.
            if path.extension().and_then(|s| s.to_str()) != Some("sh") {
                continue;
            }
            if path.file_name().and_then(|n| n.to_str()) == Some(PROFILE_D_FILE) {
                continue;
            }
            let Ok(body) = std::fs::read_to_string(&path) else {
                continue;
            };
            for var in ["WLR_NO_HARDWARE_CURSORS", "GBM_BACKEND=nvidia-drm"] {
                if body.contains(var) {
                    out.push(WaylandWarning::LegacyEnvVar {
                        path: path.clone(),
                        var: var.to_string(),
                    });
                }
            }
        }
    }

    out
}

pub fn apply(ctx: &Context, gpus: &GpuInventory) -> Result<Vec<ChangeReport>> {
    let dry_run = ctx.mode.is_dry_run();
    let mut reports = Vec::with_capacity(3);

    let profile_target = ctx.paths.profile_d.join(PROFILE_D_FILE);
    reports.push(write_dropin(
        &profile_target,
        PROFILE_D_CONTENT,
        &ctx.paths.backup_dir,
        dry_run,
    )?);

    let mkinitcpio_target = ctx.paths.mkinitcpio_d.join(MKINITCPIO_DROPIN_FILE);
    reports.push(write_dropin(
        &mkinitcpio_target,
        MKINITCPIO_DROPIN_CONTENT,
        &ctx.paths.backup_dir,
        dry_run,
    )?);

    // For hybrid (Optimus / PRIME) setups, also drop a PRIME outputclass Xorg config when the
    // default shipped by nvidia-utils isn't present. Pure Wayland sessions ignore the file.
    reports.push(prime::apply(ctx, gpus)?);

    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::{GpuInfo, GpuVendor};
    use crate::core::ExecutionMode;
    use tempfile::tempdir;

    fn empty_gpus() -> GpuInventory {
        GpuInventory { gpus: Vec::new() }
    }

    fn single_nvidia() -> GpuInventory {
        GpuInventory {
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Nvidia,
                vendor_id: 0x10de,
                device_id: 0x2684,
                pci_address: "0000:01:00.0".into(),
                vendor_name: "NVIDIA".into(),
                product_name: "RTX 4090".into(),
                kernel_driver: None,
                is_integrated: false,
                nvidia_gen: None,
            }],
        }
    }

    #[test]
    fn apply_writes_two_dropins_on_non_hybrid() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        let reports = apply(&ctx, &single_nvidia()).unwrap();
        assert_eq!(reports.len(), 3);
        assert!(dir
            .path()
            .join("etc/profile.d/99-nvidia-wayland.sh")
            .exists());
        assert!(dir
            .path()
            .join("etc/mkinitcpio.conf.d/nvidia-modules.conf")
            .exists());
    }

    #[test]
    fn apply_is_idempotent() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        apply(&ctx, &empty_gpus()).unwrap();
        let reports = apply(&ctx, &empty_gpus()).unwrap();
        for r in reports {
            assert!(matches!(r, ChangeReport::AlreadyApplied { .. }));
        }
    }

    #[test]
    fn dry_run_writes_nothing() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        apply(&ctx, &empty_gpus()).unwrap();
        assert!(!dir
            .path()
            .join("etc/profile.d/99-nvidia-wayland.sh")
            .exists());
    }

    // ── Phase 16 sanitation: X11 cruft + legacy env vars ──────────────────────────────────

    #[test]
    fn sanitation_no_warnings_on_clean_fs() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        assert!(sanitation_warnings(&ctx).is_empty());
    }

    #[test]
    fn sanitation_flags_legacy_etc_x11_xorg_conf() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        std::fs::create_dir_all(dir.path().join("etc/X11")).unwrap();
        std::fs::write(
            &ctx.paths.etc_x11_xorg_conf,
            "Section \"Device\"\n    Driver \"nvidia\"\nEndSection\n",
        )
        .unwrap();
        let w = sanitation_warnings(&ctx);
        assert_eq!(w.len(), 1);
        assert!(matches!(w[0], WaylandWarning::XorgConfLegacy(_)));
    }

    #[test]
    fn sanitation_flags_legacy_20_nvidia_conf() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        std::fs::create_dir_all(&ctx.paths.xorg_d).unwrap();
        std::fs::write(
            ctx.paths.xorg_d.join("20-nvidia.conf"),
            "Section \"Device\"\nEndSection\n",
        )
        .unwrap();
        let w = sanitation_warnings(&ctx);
        assert!(matches!(w[0], WaylandWarning::XorgConfLegacy(_)));
    }

    #[test]
    fn sanitation_flags_legacy_wlr_no_hardware_cursors() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        std::fs::create_dir_all(&ctx.paths.profile_d).unwrap();
        // User-authored (not our own file), still sourced by login shells.
        std::fs::write(
            ctx.paths.profile_d.join("50-user-wlr.sh"),
            "export WLR_NO_HARDWARE_CURSORS=1\n",
        )
        .unwrap();
        let w = sanitation_warnings(&ctx);
        assert!(w
            .iter()
            .any(|x| matches!(x, WaylandWarning::LegacyEnvVar { var, .. } if var.contains("WLR_NO_HARDWARE_CURSORS"))));
    }

    #[test]
    fn sanitation_flags_legacy_gbm_backend() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        std::fs::create_dir_all(&ctx.paths.profile_d).unwrap();
        std::fs::write(
            ctx.paths.profile_d.join("50-legacy-nvidia.sh"),
            "export GBM_BACKEND=nvidia-drm\n",
        )
        .unwrap();
        let w = sanitation_warnings(&ctx);
        assert!(w.iter().any(
            |x| matches!(x, WaylandWarning::LegacyEnvVar { var, .. } if var.contains("GBM_BACKEND"))
        ));
    }

    #[test]
    fn sanitation_skips_our_own_dropin() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        // Write our OWN drop-in with the modernized comment-only content — sanitation
        // must NOT flag it even though the comment text references GBM_BACKEND.
        std::fs::create_dir_all(&ctx.paths.profile_d).unwrap();
        std::fs::write(ctx.paths.profile_d.join(PROFILE_D_FILE), PROFILE_D_CONTENT).unwrap();
        assert!(sanitation_warnings(&ctx).is_empty());
    }
}
