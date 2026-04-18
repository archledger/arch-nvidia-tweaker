use anyhow::Result;
use std::path::Path;

use crate::core::gpu::GpuInventory;
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
///  - the default config shipped by `nvidia-utils` (`/usr/share/X11/xorg.conf.d/10-nvidia-drm-
///    outputclass.conf`) is not already present on disk.
///
/// Under Wayland this file has no effect. On non-hybrid hosts we don't want to write anything.
pub fn apply(ctx: &Context, gpus: &GpuInventory) -> Result<ChangeReport> {
    if !gpus.is_hybrid() {
        return Ok(ChangeReport::AlreadyApplied {
            detail: "PRIME config skipped: no hybrid graphics on this host".into(),
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
