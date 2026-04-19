use anyhow::Result;
use std::path::Path;

use crate::core::gpu::GpuInventory;
use crate::core::hardware::FormFactor;
use crate::core::Context;
use crate::utils::fs_helper::{write_dropin, ChangeReport};

const XORG_DROPIN_FILE: &str = "10-archgpu-prime.conf";

const XORG_CONTENT: &str = r#"# Managed by archgpu — PRIME render offload (hybrid graphics).
# Only consulted under Xorg sessions. Wayland ignores this file.
#
# `prime-run` from the `nvidia-prime` package wraps this by setting:
#   __NV_PRIME_RENDER_OFFLOAD=1
#   __GLX_VENDOR_LIBRARY_NAME=nvidia
#   __VK_LAYER_NV_optimus=NVIDIA_only

Section "OutputClass"
    Identifier "nvidia"
    MatchDriver "nvidia-drm"
    Driver "nvidia"
    Option "AllowEmptyInitialConfiguration"
    Option "PrimaryGPU" "no"
    ModulePath "/usr/lib/nvidia/xorg"
    ModulePath "/usr/lib/xorg/modules"
EndSection

Section "OutputClass"
    Identifier "intel"
    MatchDriver "i915|xe"
    Driver "modesetting"
EndSection

Section "OutputClass"
    Identifier "amd"
    MatchDriver "amdgpu"
    Driver "modesetting"
EndSection
"#;

/// Install a PRIME outputclass Xorg config in `/etc/X11/xorg.conf.d/`, but only when:
///  - the host has a hybrid GPU topology (NVIDIA + an integrated non-NVIDIA GPU), AND
///  - the form factor is Laptop (Phase 19: PRIME is a laptop/Optimus concept — on a
///    desktop tower the physical display cable dictates the primary GPU, so writing a
///    PRIME OutputClass actively misroutes the X server), AND
///  - the default config shipped by `nvidia-utils` (`/usr/share/X11/xorg.conf.d/10-nvidia-
///    drm-outputclass.conf`) is not already present on disk.
///
/// Under Wayland this file has no effect. On non-hybrid, desktop-hybrid, or
/// nvidia-utils-already-ships hosts we don't want to write anything.
///
/// Returns `AlreadyApplied` with an explanatory detail on every skip path, so the caller
/// (wayland::apply → run_actions → CLI/GUI) surfaces *why* PRIME was skipped rather than
/// silently dropping it.
pub fn apply(ctx: &Context, gpus: &GpuInventory, form: FormFactor) -> Result<ChangeReport> {
    if !gpus.is_hybrid() {
        return Ok(ChangeReport::AlreadyApplied {
            detail: "PRIME config skipped: no hybrid graphics on this host".into(),
        });
    }

    // Phase 19 multi-GPU desktop rule: on a desktop tower, "hybrid" means the user has
    // both an iGPU and a dGPU but the monitor is plugged into ONE of them directly.
    // PRIME render offload is unnecessary (no battery/thermals to save) and actively
    // harmful (the OutputClass can re-route display and break the physical-cable assumption).
    // Only laptops run `prime-run`-style offload; desktops and Unknown form factors skip.
    if form != FormFactor::Laptop {
        return Ok(ChangeReport::AlreadyApplied {
            detail: format!(
                "PRIME config skipped: form factor is {form:?}, not Laptop \
                 (desktop hybrids route display via the physical cable — no offload needed)"
            ),
        });
    }

    let shipped = Path::new("/usr/share/X11/xorg.conf.d/10-nvidia-drm-outputclass.conf");
    if shipped.exists() {
        return Ok(ChangeReport::AlreadyApplied {
            detail: format!(
                "PRIME outputclass already shipped by nvidia-utils ({})",
                shipped.display()
            ),
        });
    }

    let target = ctx.paths.xorg_d.join(XORG_DROPIN_FILE);
    write_dropin(
        &target,
        XORG_CONTENT,
        &ctx.paths.backup_dir,
        ctx.mode.is_dry_run(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::{GpuInfo, GpuVendor};
    use crate::core::ExecutionMode;
    use tempfile::tempdir;

    fn hybrid_inv() -> GpuInventory {
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

    fn amd_plus_nvidia_desktop() -> GpuInventory {
        // RTX 3060 + AMD iGPU (the Phase 19 field-test topology — hybrid, but on a desktop).
        GpuInventory {
            gpus: vec![
                GpuInfo {
                    vendor: GpuVendor::Amd,
                    vendor_id: 0x1002,
                    device_id: 0x1638,
                    pci_address: "0000:0a:00.0".into(),
                    vendor_name: "AMD".into(),
                    product_name: "Raphael iGPU".into(),
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

    #[test]
    fn laptop_hybrid_writes_xorg_dropin() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        let r = apply(&ctx, &hybrid_inv(), FormFactor::Laptop).unwrap();
        // The shipped-config probe hits the REAL `/usr/share/X11/xorg.conf.d/10-nvidia-drm-
        // outputclass.conf` (not ctx-rooted — that path is set by the nvidia-utils package).
        // On a dev host with nvidia-utils installed, prime::apply correctly returns
        // AlreadyApplied with "already shipped". On a clean sandbox without nvidia-utils it
        // returns Applied. Both are valid for a laptop hybrid; the thing this test MUST
        // reject is the Phase-19 desktop-skip branch.
        match r {
            ChangeReport::Applied { .. } => {
                assert!(
                    dir.path()
                        .join("etc/X11/xorg.conf.d/10-archgpu-prime.conf")
                        .exists(),
                    "Applied report should have created the drop-in"
                );
            }
            ChangeReport::AlreadyApplied { detail } => {
                assert!(
                    detail.contains("already shipped"),
                    "Laptop + hybrid must NOT take the Phase-19 desktop-skip branch; got: {detail}"
                );
            }
            other => panic!("unexpected report: {other:?}"),
        }
    }

    #[test]
    fn desktop_hybrid_does_not_write_xorg_dropin() {
        // Phase 19 core guarantee: hybrid desktop (RTX 3060 + AMD iGPU) must NOT receive
        // the PRIME OutputClass — it would break the physical-cable display routing.
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        let r = apply(&ctx, &amd_plus_nvidia_desktop(), FormFactor::Desktop).unwrap();
        match r {
            ChangeReport::AlreadyApplied { detail } => {
                assert!(
                    detail.contains("Desktop") && detail.contains("skipped"),
                    "skip message must mention Desktop and skipped: {detail}"
                );
            }
            other => panic!("expected AlreadyApplied, got {other:?}"),
        }
        assert!(
            !dir.path()
                .join("etc/X11/xorg.conf.d/10-archgpu-prime.conf")
                .exists(),
            "Phase 19 invariant violated: desktop hybrid received a PRIME OutputClass"
        );
    }

    #[test]
    fn unknown_form_factor_hybrid_does_not_write_xorg_dropin() {
        // Conservative default: if chassis detection fails, don't write PRIME. Wrong-laptop
        // is recoverable (user can install nvidia-prime manually), wrong-desktop is not
        // (broken display routing on reboot).
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        let r = apply(&ctx, &hybrid_inv(), FormFactor::Unknown).unwrap();
        assert!(matches!(r, ChangeReport::AlreadyApplied { .. }));
        assert!(!dir
            .path()
            .join("etc/X11/xorg.conf.d/10-archgpu-prime.conf")
            .exists());
    }

    #[test]
    fn non_hybrid_laptop_does_not_write_xorg_dropin() {
        // Single-GPU laptop (e.g. Intel-only ultrabook) — nothing to offload.
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        let intel_only = GpuInventory {
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Intel,
                vendor_id: 0x8086,
                device_id: 0x64a0,
                pci_address: "0000:00:02.0".into(),
                vendor_name: "Intel".into(),
                product_name: "Arc 140V".into(),
                kernel_driver: None,
                is_integrated: true,
                nvidia_gen: None,
            }],
        };
        let r = apply(&ctx, &intel_only, FormFactor::Laptop).unwrap();
        assert!(matches!(r, ChangeReport::AlreadyApplied { .. }));
        assert!(!dir
            .path()
            .join("etc/X11/xorg.conf.d/10-archgpu-prime.conf")
            .exists());
    }
}
