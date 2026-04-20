use anyhow::{Context as _, Result};
use std::process::Command;

use crate::core::gpu::GpuInventory;
use crate::core::hardware::FormFactor;
use crate::core::nvidia::NvidiaDriverVersion;
use crate::core::state::TweakState;
use crate::core::Context;
use crate::utils::fs_helper::{write_dropin, ChangeReport};

const MODPROBE_FILE: &str = "zzz-nvidia-tweaks-auto.conf";
const NOUVEAU_BLACKLIST_FILE: &str = "blacklist-nouveau.conf";

const NOUVEAU_BLACKLIST_CONTENT: &str = "\
# Managed by archgpu — prevents nouveau from loading while the proprietary (or
# open GSP) NVIDIA kernel modules are active. Remove this file to allow nouveau again.
blacklist nouveau
options nouveau modeset=0
";

/// Manual suspend/hibernate/resume systemd units. Required on driver branches 430-590
/// (where the userspace nvidia-sleep.sh script is what writes `/proc/driver/nvidia/suspend`
/// at the right lifecycle moments). On 595+ Arch's `nvidia-utils` install hook actively
/// DISABLES these — the kernel suspend notifier path handles everything — so enabling
/// them there just sets up a state-oscillation fight with pacman upgrades.
const SYSTEMD_UNITS: &[&str] = &[
    "nvidia-suspend.service",
    "nvidia-hibernate.service",
    "nvidia-resume.service",
];

/// Check state contract (Phase 24 update):
///   - no NVIDIA → Incompatible
///   - modprobe drop-in missing → Unapplied
///   - driver major < 595 (legacy era) → Active iff nvidia-suspend.service is enabled
///   - driver major >= 595 OR unknown → Active (suspend units are NOT required on 595+;
///     Arch's install hook actively disables them, so requiring them would keep this
///     tweak forever Unapplied)
pub fn check_state(ctx: &Context, gpus: &GpuInventory) -> TweakState {
    if !gpus.has_nvidia() {
        return TweakState::Incompatible;
    }
    let modprobe = ctx.paths.modprobe_d.join(MODPROBE_FILE);
    if !modprobe.exists() {
        return TweakState::Unapplied;
    }
    // Driver branch gate — only require the systemd units on 430-590.
    let driver = crate::core::nvidia::detect_driver_version_live();
    let needs_units = driver.is_some_and(|d| d.is_legacy_suspend_branch());
    if !needs_units {
        // 595+ (or unknown): kernel suspend notifiers handle resume; file-only
        // state is sufficient. Arch's install hook disables the systemd units on
        // nvidia-utils 595+, so requiring them would never converge.
        return TweakState::Active;
    }
    let enabled = Command::new("systemctl")
        .args(["is-enabled", "nvidia-suspend.service"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "enabled")
        .unwrap_or(false);
    if enabled {
        TweakState::Active
    } else {
        TweakState::Unapplied
    }
}

pub fn apply(ctx: &Context, form: FormFactor) -> Result<Vec<ChangeReport>> {
    let driver = crate::core::nvidia::detect_driver_version_live();
    apply_with_driver(ctx, form, driver)
}

/// Pure-core apply that takes the driver version as an explicit argument so tests
/// can inject any branch without hitting real pacman. `apply()` is the runtime wrapper.
pub fn apply_with_driver(
    ctx: &Context,
    form: FormFactor,
    driver: Option<NvidiaDriverVersion>,
) -> Result<Vec<ChangeReport>> {
    let mut reports = Vec::new();

    let content = modprobe_content(form, driver);
    let target = ctx.paths.modprobe_d.join(MODPROBE_FILE);
    reports.push(write_dropin(
        &target,
        &content,
        &ctx.paths.backup_dir,
        ctx.mode.is_dry_run(),
    )?);

    // nouveau blacklist (outer gating in run_actions ensures this only runs on NVIDIA hosts)
    let nouveau_target = ctx.paths.modprobe_d.join(NOUVEAU_BLACKLIST_FILE);
    reports.push(write_dropin(
        &nouveau_target,
        NOUVEAU_BLACKLIST_CONTENT,
        &ctx.paths.backup_dir,
        ctx.mode.is_dry_run(),
    )?);

    // Phase 24: the manual suspend/hibernate/resume systemd units are only for the
    // 430-590 driver era. On 595+ Arch's install hook disables them on every
    // upgrade and the kernel suspend notifier path handles everything — enabling
    // them would set up a state-oscillation fight with pacman. Report the skip
    // explicitly so users understand why three expected "enabled" lines didn't appear.
    match driver {
        Some(d) if d.is_legacy_suspend_branch() => {
            for unit in SYSTEMD_UNITS {
                reports.push(enable_unit(ctx, unit)?);
            }
        }
        Some(d) => {
            reports.push(ChangeReport::AlreadyApplied {
                detail: format!(
                    "NVIDIA driver {}.{} ({}) uses kernel suspend notifiers — nvidia-suspend/hibernate/resume.service are disabled by nvidia-utils' install hook and intentionally not enabled by this tool",
                    d.major,
                    d.minor,
                    d.branch_tag()
                ),
            });
        }
        None => {
            reports.push(ChangeReport::AlreadyApplied {
                detail: "NVIDIA driver version unknown (no nvidia-* package found in pacman) — skipping systemd suspend/hibernate/resume enable to avoid fighting the nvidia-utils 595+ install hook".into(),
            });
        }
    }

    Ok(reports)
}

/// Render the modprobe.d drop-in content for this host.
///
/// On the 430-590 driver era we include `NVreg_PreserveVideoMemoryAllocations=1` —
/// that's what actually tells the old driver to keep framebuffer contents across
/// suspend. On 595+ that flag was superseded by `NVreg_UseKernelSuspendNotifiers=1`
/// (which we always set) and no longer honored, so we omit it from modern content
/// to avoid confusing future readers.
pub fn modprobe_content(form: FormFactor, driver: Option<NvidiaDriverVersion>) -> String {
    let mut s = String::new();
    s.push_str("# Managed by archgpu — do not edit by hand.\n");
    s.push_str("# Required on all hosts to avoid Wayland wake artifacts.\n");
    s.push_str("options nvidia NVreg_UseKernelSuspendNotifiers=1\n");

    // Phase 24 gate: NVreg_PreserveVideoMemoryAllocations is the 430-590 companion
    // to nvidia-suspend.service. It was succeeded by UseKernelSuspendNotifiers in 595
    // and is a no-op there. Writing it on 595+ isn't harmful (the driver ignores it),
    // but it misleads anyone reading this drop-in and gives a false sense that we're
    // doing work that actually has no effect.
    let is_legacy_branch = driver.is_some_and(|d| d.is_legacy_suspend_branch());
    if is_legacy_branch {
        s.push_str("# NVreg_PreserveVideoMemoryAllocations is the 430-590 branch's VRAM-preservation flag.\n");
        s.push_str("# On 595+ drivers it was succeeded by UseKernelSuspendNotifiers (set above) and is no longer honored.\n");
        s.push_str("options nvidia NVreg_PreserveVideoMemoryAllocations=1\n");
    }

    if form == FormFactor::Laptop {
        s.push_str("# Hybrid-graphics dynamic power management (laptops only).\n");
        s.push_str("options nvidia NVreg_DynamicPowerManagement=0x02\n");
    }
    s
}

fn enable_unit(ctx: &Context, unit: &str) -> Result<ChangeReport> {
    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned {
            detail: format!("systemctl enable {unit}"),
        });
    }

    let status = Command::new("systemctl")
        .arg("is-enabled")
        .arg(unit)
        .output()
        .with_context(|| format!("checking systemctl is-enabled {unit}"))?;
    let out = String::from_utf8_lossy(&status.stdout);
    if out.trim() == "enabled" {
        return Ok(ChangeReport::AlreadyApplied {
            detail: format!("{unit} already enabled"),
        });
    }

    log::info!("Enabling {unit}");
    let exit = Command::new("systemctl")
        .arg("enable")
        .arg(unit)
        .status()
        .with_context(|| format!("spawning systemctl enable {unit}"))?;
    if !exit.success() {
        anyhow::bail!("systemctl enable {unit} exited with {exit}");
    }
    Ok(ChangeReport::Applied {
        detail: format!("enabled {unit}"),
        backup: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_driver() -> Option<NvidiaDriverVersion> {
        Some(NvidiaDriverVersion {
            major: 580,
            minor: 142,
        })
    }

    fn modern_driver() -> Option<NvidiaDriverVersion> {
        Some(NvidiaDriverVersion {
            major: 595,
            minor: 58,
        })
    }

    // ── modprobe_content per-branch matrix ────────────────────────────────────────────────

    #[test]
    fn legacy_branch_laptop_gets_all_three_nvreg_options() {
        let s = modprobe_content(FormFactor::Laptop, legacy_driver());
        assert!(s.contains("NVreg_UseKernelSuspendNotifiers=1"));
        assert!(
            s.contains("NVreg_PreserveVideoMemoryAllocations=1"),
            "legacy 580 branch needs the VRAM preservation flag explicitly"
        );
        assert!(s.contains("NVreg_DynamicPowerManagement=0x02"));
    }

    #[test]
    fn modern_branch_laptop_omits_preserve_vram() {
        // Phase 24: on 595+ the PreserveVideoMemoryAllocations flag is deprecated —
        // writing it is misleading (driver ignores it) even if harmless.
        let s = modprobe_content(FormFactor::Laptop, modern_driver());
        assert!(s.contains("NVreg_UseKernelSuspendNotifiers=1"));
        assert!(
            !s.contains("NVreg_PreserveVideoMemoryAllocations=1"),
            "595+ branch must not emit the deprecated PreserveVideoMemoryAllocations flag, got:\n{s}"
        );
        assert!(s.contains("NVreg_DynamicPowerManagement=0x02"));
    }

    #[test]
    fn modern_branch_desktop_omits_both_legacy_and_laptop_options() {
        let s = modprobe_content(FormFactor::Desktop, modern_driver());
        assert!(s.contains("NVreg_UseKernelSuspendNotifiers=1"));
        assert!(!s.contains("NVreg_PreserveVideoMemoryAllocations"));
        assert!(!s.contains("NVreg_DynamicPowerManagement"));
    }

    #[test]
    fn unknown_driver_treated_as_modern_for_modprobe() {
        // Conservative default when driver version is unknown: treat as modern so we
        // don't write the deprecated flag. Matches Arch's install-hook defaults and
        // avoids state oscillation.
        let s = modprobe_content(FormFactor::Laptop, None);
        assert!(s.contains("NVreg_UseKernelSuspendNotifiers=1"));
        assert!(!s.contains("NVreg_PreserveVideoMemoryAllocations"));
    }

    // ── form-factor tests preserved from earlier phases ───────────────────────────────────

    #[test]
    fn desktop_content_omits_laptop_option() {
        let s = modprobe_content(FormFactor::Desktop, legacy_driver());
        assert!(s.contains("NVreg_UseKernelSuspendNotifiers=1"));
        assert!(!s.contains("NVreg_DynamicPowerManagement"));
    }

    #[test]
    fn unknown_form_treated_as_non_laptop() {
        let s = modprobe_content(FormFactor::Unknown, legacy_driver());
        assert!(!s.contains("NVreg_DynamicPowerManagement"));
    }

    #[test]
    fn nouveau_blacklist_content_is_stable() {
        assert!(NOUVEAU_BLACKLIST_CONTENT.contains("blacklist nouveau"));
        assert!(NOUVEAU_BLACKLIST_CONTENT.contains("options nouveau modeset=0"));
    }

    // ── apply_with_driver systemd gating ──────────────────────────────────────────────────

    use crate::core::ExecutionMode;
    use tempfile::tempdir;

    #[test]
    fn modern_branch_apply_does_not_plan_systemd_enables() {
        // Dry-run with 595+: the three nvidia-suspend unit Planned entries must NOT
        // appear. Instead, one AlreadyApplied entry explains why we skipped them.
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        let reports =
            apply_with_driver(&ctx, FormFactor::Laptop, modern_driver()).unwrap();

        let systemd_plans: Vec<_> = reports
            .iter()
            .filter(|r| matches!(r, ChangeReport::Planned { detail } if detail.contains("systemctl enable")))
            .collect();
        assert!(
            systemd_plans.is_empty(),
            "595+ must not plan systemctl enable for suspend units: {reports:#?}"
        );

        let skip_note = reports.iter().any(|r| matches!(
            r,
            ChangeReport::AlreadyApplied { detail } if detail.contains("kernel suspend notifiers")
        ));
        assert!(skip_note, "595+ skip note missing from apply reports");
    }

    #[test]
    fn legacy_branch_apply_plans_three_systemd_enables() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        let reports =
            apply_with_driver(&ctx, FormFactor::Laptop, legacy_driver()).unwrap();

        let systemd_plans: Vec<_> = reports
            .iter()
            .filter(|r| matches!(r, ChangeReport::Planned { detail } if detail.contains("systemctl enable")))
            .collect();
        assert_eq!(
            systemd_plans.len(),
            3,
            "legacy branch must plan three systemctl enables, got: {reports:#?}"
        );
    }

    #[test]
    fn unknown_driver_apply_skips_systemd_conservatively() {
        // Phase 24 safety: when version is unknown (no nvidia-* installed in pacman),
        // skip the systemd enable path. Avoids fighting Arch's 595+ install hook
        // when the user simply hasn't installed any NVIDIA package yet (first run).
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        let reports = apply_with_driver(&ctx, FormFactor::Desktop, None).unwrap();

        let systemd_plans: Vec<_> = reports
            .iter()
            .filter(|r| matches!(r, ChangeReport::Planned { detail } if detail.contains("systemctl enable")))
            .collect();
        assert!(systemd_plans.is_empty());
    }
}
