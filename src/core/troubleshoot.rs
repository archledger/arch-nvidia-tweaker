//! Phase 29 — smart troubleshoot loop.
//!
//! Each `Recipe` runs a `detect → cause → fix → verify → explain` cycle:
//!
//! 1. **Detect** — does the symptom match this host right now?
//! 2. **Cause** — single human sentence explaining what produced the symptom.
//! 3. **Fix** — write the change. Skipped on dry-run; skipped if the recipe is
//!    diagnostic-only (some root causes have no safe automatic remedy).
//! 4. **Verify** — re-probe. Three outcomes:
//!     - `LiveVerified`: the symptom is gone right now (sysfs/proc/file confirms).
//!     - `PendingReboot`: config is correct on disk but the running kernel still
//!       reports the old state — the user must reboot.
//!     - `Failed`: fix ran but verify still sees the symptom — emit a clear
//!       explanation rather than claiming success.
//! 5. **Explain** — `RecipeReport.summary()` produces a one-paragraph report
//!    suitable for the CLI / GUI console: "Cause: X. Fix applied: Y. Verified: Z."
//!
//! Initial recipes (Phase 29):
//!
//!   software_rendering        diagnostic-only — too many root causes to safely
//!                             auto-fix; points at --apply-essentials and
//!                             --apply-groups, removes orphan ICDs if any
//!   nomodeset_stuck           auto-fixable — calls bootloader::apply_remove
//!   nouveau_active_with_nvidia auto-fixable — writes blacklist + mkinitcpio -P;
//!                             verify is PendingReboot until next boot
//!   dangling_vulkan_icd       auto-fixable — removes orphan JSON files (backed
//!                             up to /var/backups/archgpu/) and re-probes
//!
//! Recipes 5–9 (sg_display, runpm, abmlevel, enable_psr, CUDA+AMD ICD pinning,
//! Secure-Boot-blocking-NVIDIA) land in a Phase 31 follow-up.

use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;

use crate::core::gpu::GpuInventory;
use crate::core::rendering::{
    self, classify_renderer_output, IcdProblem, RendererState,
};
use crate::core::{bootloader, Context};
use crate::utils::fs_helper::{backup_to_dir, ChangeReport};
use crate::utils::process::run_streaming;

// ── Public types ───────────────────────────────────────────────────────────────────────────────

/// Result of re-probing after a fix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verification {
    /// The fix took effect immediately and the recheck passes now.
    LiveVerified(String),
    /// Config is correct on disk; running kernel hasn't adopted yet. Reboot required.
    PendingReboot(String),
    /// Fix attempted but the recheck still sees the symptom. Detail explains why.
    Failed(String),
    /// Recipe didn't run a fix (dry-run, or recipe is diagnostic-only).
    NotApplicable,
}

impl Verification {
    /// Short stable label for this outcome — used by the future Phase 30 GUI to
    /// pick a per-recipe badge color/icon (`#[allow(dead_code)]` until then).
    #[allow(dead_code)]
    pub fn label(&self) -> &'static str {
        match self {
            Self::LiveVerified(_) => "live-verified",
            Self::PendingReboot(_) => "pending-reboot",
            Self::Failed(_) => "verification-failed",
            Self::NotApplicable => "n/a",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecipeReport {
    pub id: &'static str,
    pub title: &'static str,
    /// `None` if the recipe didn't match this host (no symptom present).
    pub symptom: Option<String>,
    /// Human-readable cause (one sentence). Empty if the recipe didn't match.
    pub cause: String,
    /// What the fix did. `None` if no fix ran.
    pub fix_applied: Option<String>,
    pub verification: Verification,
}

impl RecipeReport {
    pub fn summary(&self) -> String {
        let Some(symptom) = self.symptom.as_deref() else {
            return format!("[{}] no symptom — recipe did not run", self.id);
        };
        let mut s = format!(
            "[{}] {}\n  symptom: {symptom}\n  cause:   {}",
            self.id, self.title, self.cause
        );
        if let Some(fix) = &self.fix_applied {
            s.push_str(&format!("\n  fix:     {fix}"));
        }
        match &self.verification {
            Verification::LiveVerified(d) => s.push_str(&format!("\n  verify:  ✓ {d}")),
            Verification::PendingReboot(d) => s.push_str(&format!("\n  verify:  ⟳ reboot required — {d}")),
            Verification::Failed(d) => s.push_str(&format!("\n  verify:  ✗ {d}")),
            Verification::NotApplicable => {}
        }
        s
    }
}

pub trait Recipe {
    fn id(&self) -> &'static str;
    fn title(&self) -> &'static str;
    /// Run the full detect → fix → verify cycle. Returns a report whether or not
    /// the symptom matched. Errors are reserved for "the tool itself broke" —
    /// "fix didn't work" goes in `Verification::Failed`.
    fn run(
        &self,
        ctx: &Context,
        gpus: &GpuInventory,
        assume_yes: bool,
        progress: &mut dyn FnMut(&str),
    ) -> Result<RecipeReport>;
}

// ── Module entry points ────────────────────────────────────────────────────────────────────────

pub fn all_recipes() -> Vec<Box<dyn Recipe>> {
    vec![
        Box::new(NomodesetStuck),
        Box::new(NouveauActiveWithNvidia),
        Box::new(DanglingVulkanIcd),
        Box::new(SoftwareRendering),
    ]
}

pub fn apply(
    ctx: &Context,
    gpus: &GpuInventory,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ChangeReport>> {
    let recipes = all_recipes();
    let mut out = Vec::new();
    let mut matched = 0usize;
    for r in &recipes {
        progress(&format!("[troubleshoot] {} — probing...", r.id()));
        let report = r.run(ctx, gpus, assume_yes, progress)?;
        if report.symptom.is_some() {
            matched += 1;
            for line in report.summary().lines() {
                progress(&format!("[troubleshoot] {line}"));
            }
        }
        out.push(report_to_change(report));
    }
    progress(&format!(
        "[troubleshoot] done — {} of {} recipes matched",
        matched,
        recipes.len()
    ));
    Ok(out)
}

fn report_to_change(r: RecipeReport) -> ChangeReport {
    match (r.symptom.is_some(), &r.verification) {
        (false, _) => ChangeReport::AlreadyApplied {
            detail: format!("[{}] no symptom", r.id),
        },
        (true, Verification::NotApplicable) => ChangeReport::Planned {
            detail: format!("[{}] {} — {}", r.id, r.title, r.cause),
        },
        (true, Verification::LiveVerified(d)) => ChangeReport::Applied {
            detail: format!("[{}] resolved — {d}", r.id),
            backup: None,
        },
        (true, Verification::PendingReboot(d)) => ChangeReport::Applied {
            detail: format!("[{}] reboot required — {d}", r.id),
            backup: None,
        },
        (true, Verification::Failed(d)) => ChangeReport::Applied {
            detail: format!("[{}] fix attempted but verify failed — {d}", r.id),
            backup: None,
        },
    }
}

// ═════════════════════════════════════════════════════════════════════════════════════════════
// Recipe 1: nomodeset_stuck
// ═════════════════════════════════════════════════════════════════════════════════════════════

pub struct NomodesetStuck;

impl Recipe for NomodesetStuck {
    fn id(&self) -> &'static str {
        "nomodeset_stuck"
    }
    fn title(&self) -> &'static str {
        "`nomodeset` is on the kernel cmdline"
    }
    fn run(
        &self,
        ctx: &Context,
        _gpus: &GpuInventory,
        _assume_yes: bool,
        progress: &mut dyn FnMut(&str),
    ) -> Result<RecipeReport> {
        let in_live = rendering::check_nomodeset_in_cmdline(&ctx.paths.proc_cmdline);
        if !in_live {
            return Ok(no_match(self.id(), self.title()));
        }
        let symptom = "`nomodeset` is in /proc/cmdline".to_string();
        let cause = "`nomodeset` blocks DRM modesetting for every GPU driver, forcing every \
                     graphics surface through software (llvmpipe). Usually added by a rescue \
                     boot entry, an outdated tutorial, or a one-shot bootloader menu choice."
            .to_string();

        if ctx.mode.is_dry_run() {
            return Ok(RecipeReport {
                id: self.id(),
                title: self.title(),
                symptom: Some(symptom),
                cause,
                fix_applied: Some("would call bootloader::apply_remove([\"nomodeset\"])".into()),
                verification: Verification::NotApplicable,
            });
        }
        // apply_remove is itself the authoritative source-check: if `nomodeset` isn't in
        // any persistent cmdline source it returns AlreadyApplied without spawning the
        // post-edit subprocess (mkinitcpio / grub-mkconfig / bootctl). If it IS in a
        // source, apply_remove strips it and runs the appropriate regeneration.
        let removal = bootloader::apply_remove(ctx, &["nomodeset"], progress)?;
        let touched_source = matches!(
            removal,
            ChangeReport::Applied { .. } | ChangeReport::Planned { .. }
        );
        // Live verify: /proc/cmdline cannot change without a reboot. Always either
        // PendingReboot (we did write the source — reboot will clear) or
        // PendingReboot (nothing to write — boot menu one-shot, reboot still clears).
        let verification = if touched_source {
            Verification::PendingReboot(
                "removed `nomodeset` from active bootloader cmdline source; reboot to clear /proc/cmdline"
                    .into(),
            )
        } else {
            Verification::PendingReboot(
                "no `nomodeset` in any persistent cmdline source — likely a one-shot boot menu choice; \
                 reboot will clear /proc/cmdline"
                    .into(),
            )
        };
        Ok(RecipeReport {
            id: self.id(),
            title: self.title(),
            symptom: Some(symptom),
            cause,
            fix_applied: Some(match removal {
                ChangeReport::Applied { detail, .. } => detail,
                ChangeReport::Planned { detail } => detail,
                ChangeReport::AlreadyApplied { detail } => detail,
            }),
            verification,
        })
    }
}

// ═════════════════════════════════════════════════════════════════════════════════════════════
// Recipe 2: nouveau_active_with_nvidia
// ═════════════════════════════════════════════════════════════════════════════════════════════

pub struct NouveauActiveWithNvidia;

const NOUVEAU_BLACKLIST_NAME: &str = "blacklist-nouveau.conf";
const NOUVEAU_BLACKLIST_BODY: &str = "\
# Managed by archgpu — recipe nouveau_active_with_nvidia. Remove this file to
# permit nouveau again (e.g. if you uninstall the proprietary NVIDIA stack).
blacklist nouveau
options nouveau modeset=0
";

impl Recipe for NouveauActiveWithNvidia {
    fn id(&self) -> &'static str {
        "nouveau_active_with_nvidia"
    }
    fn title(&self) -> &'static str {
        "nouveau driver is loaded but proprietary NVIDIA stack is installed"
    }
    fn run(
        &self,
        ctx: &Context,
        gpus: &GpuInventory,
        _assume_yes: bool,
        progress: &mut dyn FnMut(&str),
    ) -> Result<RecipeReport> {
        if !gpus.has_nvidia() {
            return Ok(no_match(self.id(), self.title()));
        }
        let nvidia_installed = pacman_pkg_installed("nvidia-utils");
        let nouveau_loaded = ctx.paths.sys_module.join("nouveau").exists();
        if !(nvidia_installed && nouveau_loaded) {
            return Ok(no_match(self.id(), self.title()));
        }
        let symptom = "nvidia-utils installed AND /sys/module/nouveau exists".to_string();
        let cause =
            "nouveau wins the boot race because it isn't blacklisted and/or the NVIDIA modules \
             aren't in the initramfs. The result is nouveau-driven mode-set (slow, no compute) \
             even though the proprietary stack is on disk."
                .to_string();

        let blacklist_path = ctx.paths.modprobe_d.join(NOUVEAU_BLACKLIST_NAME);
        if ctx.mode.is_dry_run() {
            return Ok(RecipeReport {
                id: self.id(),
                title: self.title(),
                symptom: Some(symptom),
                cause,
                fix_applied: Some(format!(
                    "would write {} + run `mkinitcpio -P`",
                    blacklist_path.display()
                )),
                verification: Verification::NotApplicable,
            });
        }
        // Write the blacklist + rebuild initramfs.
        if let Some(parent) = blacklist_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        crate::utils::fs_helper::atomic_write(&blacklist_path, NOUVEAU_BLACKLIST_BODY)?;
        progress(&format!(
            "[troubleshoot] wrote {}",
            blacklist_path.display()
        ));
        let mut cmd = Command::new("mkinitcpio");
        cmd.arg("-P");
        progress("[mkinitcpio] -P");
        let status = run_streaming(cmd, |line| progress(&format!("[mkinitcpio] {line}")))?;
        if !status.success() {
            anyhow::bail!("mkinitcpio -P exited with {status}");
        }

        // Live-verify: nouveau won't unload from a running session. Always PendingReboot.
        let still_loaded = ctx.paths.sys_module.join("nouveau").exists();
        let verification = if still_loaded {
            Verification::PendingReboot(
                "blacklist written + initramfs rebuilt; nouveau still loaded in this session — \
                 reboot to confirm it stays unloaded"
                    .into(),
            )
        } else {
            Verification::LiveVerified("nouveau is no longer loaded".into())
        };
        Ok(RecipeReport {
            id: self.id(),
            title: self.title(),
            symptom: Some(symptom),
            cause,
            fix_applied: Some(format!(
                "wrote {} and rebuilt initramfs",
                blacklist_path.display()
            )),
            verification,
        })
    }
}

fn pacman_pkg_installed(name: &str) -> bool {
    Command::new("pacman")
        .args(["-Qq", name])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ═════════════════════════════════════════════════════════════════════════════════════════════
// Recipe 3: dangling_vulkan_icd
// ═════════════════════════════════════════════════════════════════════════════════════════════

pub struct DanglingVulkanIcd;

impl Recipe for DanglingVulkanIcd {
    fn id(&self) -> &'static str {
        "dangling_vulkan_icd"
    }
    fn title(&self) -> &'static str {
        "Vulkan ICD JSON references a nonexistent .so"
    }
    fn run(
        &self,
        ctx: &Context,
        _gpus: &GpuInventory,
        _assume_yes: bool,
        progress: &mut dyn FnMut(&str),
    ) -> Result<RecipeReport> {
        let issues = rendering::check_vulkan_icds(&ctx.paths.vulkan_icd_dir);
        if issues.is_empty() {
            return Ok(no_match(self.id(), self.title()));
        }
        let symptom = format!(
            "{} ICD JSON file(s) point at missing or unparseable libraries: {}",
            issues.len(),
            issues
                .iter()
                .map(|i| i.json_path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let cause = "Vulkan loader iterates /usr/share/vulkan/icd.d/*.json at startup. A JSON \
                     pointing at a deleted .so causes the loader to either skip that ICD or — \
                     worse — surface an init error the application's fallback chain converts to \
                     llvmpipe."
            .to_string();

        if ctx.mode.is_dry_run() {
            return Ok(RecipeReport {
                id: self.id(),
                title: self.title(),
                symptom: Some(symptom),
                cause,
                fix_applied: Some(format!(
                    "would back up + remove {} orphan ICD JSON file(s)",
                    issues.len()
                )),
                verification: Verification::NotApplicable,
            });
        }

        let mut removed: Vec<PathBuf> = Vec::new();
        for issue in &issues {
            // Only auto-remove DanglingAbsolutePath. Unparseable JSON might be a temporary
            // package-install race; safer to surface it for the user to inspect.
            if !matches!(issue.problem, IcdProblem::DanglingAbsolutePath(_)) {
                continue;
            }
            let _ = backup_to_dir(&issue.json_path, &ctx.paths.backup_dir)?;
            std::fs::remove_file(&issue.json_path)?;
            progress(&format!(
                "[troubleshoot] removed orphan ICD: {}",
                issue.json_path.display()
            ));
            removed.push(issue.json_path.clone());
        }

        let still = rendering::check_vulkan_icds(&ctx.paths.vulkan_icd_dir);
        let still_dangling: Vec<&rendering::IcdIssue> = still
            .iter()
            .filter(|i| matches!(i.problem, IcdProblem::DanglingAbsolutePath(_)))
            .collect();
        let verification = if still_dangling.is_empty() {
            Verification::LiveVerified(format!("removed {} orphan ICD file(s)", removed.len()))
        } else {
            Verification::Failed(format!(
                "still {} dangling JSON(s) after fix — manual inspection required",
                still_dangling.len()
            ))
        };
        Ok(RecipeReport {
            id: self.id(),
            title: self.title(),
            symptom: Some(symptom),
            cause,
            fix_applied: Some(format!("removed {} orphan ICD JSON file(s)", removed.len())),
            verification,
        })
    }
}

// ═════════════════════════════════════════════════════════════════════════════════════════════
// Recipe 4: software_rendering (diagnostic-only)
// ═════════════════════════════════════════════════════════════════════════════════════════════

pub struct SoftwareRendering;

impl Recipe for SoftwareRendering {
    fn id(&self) -> &'static str {
        "software_rendering"
    }
    fn title(&self) -> &'static str {
        "running in llvmpipe / software rendering"
    }
    fn run(
        &self,
        _ctx: &Context,
        _gpus: &GpuInventory,
        _assume_yes: bool,
        _progress: &mut dyn FnMut(&str),
    ) -> Result<RecipeReport> {
        // Try glxinfo first; fall back to vulkaninfo. Both being missing → Unknown,
        // which means we can't make a determination — surface as no-match.
        let renderer = probe_renderer();
        let RendererState::SoftwareRendering(line) = renderer else {
            return Ok(no_match(self.id(), self.title()));
        };
        let symptom = format!("renderer probe reports software: {line}");
        let cause = "llvmpipe is the Mesa software rasterizer — the kernel can't or won't open a \
                     hardware DRM device for this session. Common causes: missing `render` group \
                     membership, missing vendor Vulkan ICD package, dangling ICD JSON, kernel \
                     modules not loaded (after pacman kernel upgrade without reboot), or \
                     `nomodeset` on the cmdline."
            .to_string();
        // Diagnostic-only — too many root causes to safely auto-fix. The other recipes
        // in this batch handle the deterministic fixes; this one points the user there.
        let next_steps = "Run, in order: `archgpu --diagnose` (see specific cause), then \
                          `archgpu --apply-essentials` (vendor ICDs), `archgpu --apply-groups` \
                          (render group), `archgpu --apply-troubleshoot` (auto-fixes other \
                          recipes detect)."
            .to_string();
        Ok(RecipeReport {
            id: self.id(),
            title: self.title(),
            symptom: Some(symptom),
            cause,
            fix_applied: Some(next_steps),
            verification: Verification::NotApplicable,
        })
    }
}

fn probe_renderer() -> RendererState {
    if let Ok(out) = Command::new("glxinfo").arg("-B").output() {
        if out.status.success() {
            let body = String::from_utf8_lossy(&out.stdout);
            return classify_renderer_output(&body);
        }
    }
    if let Ok(out) = Command::new("vulkaninfo").arg("--summary").output() {
        if out.status.success() {
            let body = String::from_utf8_lossy(&out.stdout);
            return classify_renderer_output(&body);
        }
    }
    RendererState::Unknown
}

fn no_match(id: &'static str, title: &'static str) -> RecipeReport {
    RecipeReport {
        id,
        title,
        symptom: None,
        cause: String::new(),
        fix_applied: None,
        verification: Verification::NotApplicable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::{GpuInfo, GpuVendor, NvidiaGeneration};
    use crate::core::ExecutionMode;
    use tempfile::tempdir;

    fn ctx_dry(root: &std::path::Path) -> Context {
        Context::rooted_for_test(root, ExecutionMode::DryRun)
    }
    fn ctx_apply(root: &std::path::Path) -> Context {
        Context::rooted_for_test(root, ExecutionMode::Apply)
    }

    fn nvidia_only() -> GpuInventory {
        GpuInventory {
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Nvidia,
                vendor_id: 0x10de,
                device_id: 0x2204,
                pci_address: "0000:01:00.0".into(),
                vendor_name: "NVIDIA".into(),
                product_name: "RTX 3090".into(),
                kernel_driver: None,
                is_integrated: false,
                nvidia_gen: Some(NvidiaGeneration::Ampere),
            }],
        }
    }
    fn intel_only() -> GpuInventory {
        GpuInventory {
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
        }
    }
    fn empty_inventory() -> GpuInventory {
        GpuInventory { gpus: vec![] }
    }

    fn seed(p: &std::path::Path, body: &str) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    // ── nomodeset_stuck ──────────────────────────────────────────────────────────────────────

    #[test]
    fn nomodeset_recipe_reports_no_match_on_clean_cmdline() {
        let tmp = tempdir().unwrap();
        let ctx = ctx_dry(tmp.path());
        seed(&ctx.paths.proc_cmdline, "rw quiet\n");
        let r = NomodesetStuck
            .run(&ctx, &empty_inventory(), false, &mut |_| {})
            .unwrap();
        assert!(r.symptom.is_none());
    }

    #[test]
    fn nomodeset_recipe_dry_run_reports_planned_fix() {
        let tmp = tempdir().unwrap();
        let ctx = ctx_dry(tmp.path());
        seed(&ctx.paths.proc_cmdline, "rw quiet nomodeset\n");
        let r = NomodesetStuck
            .run(&ctx, &empty_inventory(), false, &mut |_| {})
            .unwrap();
        assert!(r.symptom.is_some());
        assert!(r.fix_applied.as_deref().unwrap().contains("apply_remove"));
        assert!(matches!(r.verification, Verification::NotApplicable));
    }

    #[test]
    fn nomodeset_recipe_apply_with_no_source_returns_pending_reboot() {
        // /proc/cmdline has nomodeset but no bootloader source is detectable
        // (rooted test fixture has none of grub_default / limine / kernel_cmdline /
        // sdb_loader_conf seeded). bootloader::apply_remove returns AlreadyApplied
        // for Unknown bootloader — no subprocess spawned. Recipe still reports
        // symptom + PendingReboot because /proc/cmdline can't change without reboot.
        let tmp = tempdir().unwrap();
        let ctx = ctx_apply(tmp.path());
        seed(&ctx.paths.proc_cmdline, "rw quiet nomodeset\n");
        let r = NomodesetStuck
            .run(&ctx, &empty_inventory(), false, &mut |_| {})
            .unwrap();
        assert!(r.symptom.is_some());
        assert!(matches!(r.verification, Verification::PendingReboot(_)));
        // Fix-applied detail surfaces apply_remove's "Unknown bootloader" message
        // so the user understands what (didn't) happen.
        let fix = r.fix_applied.unwrap();
        assert!(
            fix.contains("unknown bootloader") || fix.contains("can't remove"),
            "got: {fix}"
        );
    }

    // ── nouveau_active_with_nvidia ───────────────────────────────────────────────────────────

    #[test]
    fn nouveau_recipe_no_match_on_non_nvidia_host() {
        let tmp = tempdir().unwrap();
        let ctx = ctx_dry(tmp.path());
        let r = NouveauActiveWithNvidia
            .run(&ctx, &intel_only(), false, &mut |_| {})
            .unwrap();
        assert!(r.symptom.is_none());
    }

    #[test]
    fn nouveau_recipe_no_match_when_nouveau_not_loaded() {
        let tmp = tempdir().unwrap();
        let ctx = ctx_dry(tmp.path());
        // No /sys/module/nouveau in the rooted tree.
        let r = NouveauActiveWithNvidia
            .run(&ctx, &nvidia_only(), false, &mut |_| {})
            .unwrap();
        // Even though nvidia is present, nouveau isn't loaded → no match.
        assert!(r.symptom.is_none());
    }

    // ── dangling_vulkan_icd ──────────────────────────────────────────────────────────────────

    #[test]
    fn dangling_icd_recipe_no_match_on_clean_dir() {
        let tmp = tempdir().unwrap();
        let ctx = ctx_dry(tmp.path());
        std::fs::create_dir_all(&ctx.paths.vulkan_icd_dir).unwrap();
        let r = DanglingVulkanIcd
            .run(&ctx, &empty_inventory(), false, &mut |_| {})
            .unwrap();
        assert!(r.symptom.is_none());
    }

    #[test]
    fn dangling_icd_recipe_apply_removes_orphan_and_verifies() {
        let tmp = tempdir().unwrap();
        let ctx = ctx_apply(tmp.path());
        std::fs::create_dir_all(&ctx.paths.vulkan_icd_dir).unwrap();
        let orphan = ctx.paths.vulkan_icd_dir.join("orphan.json");
        std::fs::write(
            &orphan,
            r#"{"ICD":{"library_path":"/nonexistent/libdangling.so.0"}}"#,
        )
        .unwrap();
        assert!(orphan.exists());

        let r = DanglingVulkanIcd
            .run(&ctx, &empty_inventory(), false, &mut |_| {})
            .unwrap();
        assert!(r.symptom.is_some());
        assert!(!orphan.exists(), "orphan ICD should have been removed");
        assert!(matches!(r.verification, Verification::LiveVerified(_)));
        // Backup should exist.
        let backups: Vec<_> = std::fs::read_dir(&ctx.paths.backup_dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert!(
            backups.iter().any(|n| n.to_string_lossy().contains("orphan.json")),
            "backup not found in {}",
            ctx.paths.backup_dir.display()
        );
    }

    #[test]
    fn dangling_icd_recipe_dry_run_does_not_remove() {
        let tmp = tempdir().unwrap();
        let ctx = ctx_dry(tmp.path());
        std::fs::create_dir_all(&ctx.paths.vulkan_icd_dir).unwrap();
        let orphan = ctx.paths.vulkan_icd_dir.join("orphan.json");
        std::fs::write(
            &orphan,
            r#"{"ICD":{"library_path":"/nonexistent/libdangling.so.0"}}"#,
        )
        .unwrap();
        let r = DanglingVulkanIcd
            .run(&ctx, &empty_inventory(), false, &mut |_| {})
            .unwrap();
        assert!(r.symptom.is_some());
        assert!(orphan.exists(), "dry-run must not remove files");
        assert!(matches!(r.verification, Verification::NotApplicable));
    }

    // ── report formatting ────────────────────────────────────────────────────────────────────

    #[test]
    fn recipe_report_summary_omits_fix_when_none() {
        let r = RecipeReport {
            id: "test",
            title: "test recipe",
            symptom: Some("found".into()),
            cause: "because".into(),
            fix_applied: None,
            verification: Verification::NotApplicable,
        };
        let s = r.summary();
        assert!(s.contains("symptom: found"));
        assert!(s.contains("cause:   because"));
        assert!(!s.contains("fix:"));
    }

    #[test]
    fn recipe_report_summary_no_symptom_yields_short_line() {
        let r = no_match("test", "test recipe");
        assert!(r.summary().contains("no symptom"));
    }
}
