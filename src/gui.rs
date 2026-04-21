use anyhow::Result;
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::rc::Rc;
use std::thread;

use crate::core::auto;
use crate::core::bootloader::{self, BootloaderType};
use crate::core::cleanup;
use crate::core::diagnostics::{self, Severity};
use crate::core::essentials;
use crate::core::gaming;
use crate::core::gpu::{GpuInventory, PackageSource};
use crate::core::groups;
use crate::core::hardware::{self, FormFactor};
use crate::core::power;
use crate::core::repair;
use crate::core::state::TweakState;
use crate::core::wayland;
use crate::core::{self, Actions, Context, ExecutionMode, SystemPaths};

slint::include_modules!();

pub fn run() -> Result<()> {
    let ui = MainWindow::new()?;

    populate_detection(&ui);
    append_log(
        &ui,
        "ArchGPU — ready. Click Auto-Optimize for a one-click recommendation, or tune toggles manually and hit Preview/Apply.",
    );

    // ── Refresh detection ────────────────────────────────────────────────────────────────────
    {
        let weak = ui.as_weak();
        ui.on_detect(move || {
            if let Some(ui) = weak.upgrade() {
                populate_detection(&ui);
                append_log(&ui, "Detection refreshed.");
            }
        });
    }

    // ── Auto-Optimize (hero button) ──────────────────────────────────────────────────────────
    {
        let weak = ui.as_weak();
        ui.on_auto_optimize(move || {
            let Some(ui) = weak.upgrade() else { return };
            let ctx = Context::production(ExecutionMode::DryRun);
            let form = hardware::get_chassis_type(&ctx.paths.dmi_chassis).unwrap_or(FormFactor::Unknown);
            let gpus = GpuInventory::detect().unwrap_or_default();

            let rec = auto::recommend(&ctx, form, &gpus);
            ui.set_opt_wayland(rec.wayland);
            ui.set_opt_bootloader(rec.bootloader);
            ui.set_opt_power(rec.power);
            ui.set_opt_gaming(rec.gaming);
            ui.set_opt_repair(rec.repair);
            // Phase 30: Auto-Optimize also flips on Essentials and Groups when
            // their check_state says Unapplied. Cleanup and troubleshoot stay
            // OFF — opt-in invariant from Phases 28/29.
            ui.set_opt_essentials(rec.essentials);
            ui.set_opt_groups(rec.groups);

            let names = auto::recommended_names(rec);
            if names.is_empty() {
                append_log(
                    &ui,
                    "Auto-Optimize: system is already well-configured — nothing to do.",
                );
            } else {
                append_log(
                    &ui,
                    &format!(
                        "Auto-Optimize: selected {} action(s) — {}. Review, then click {}.",
                        names.len(),
                        names.join(", "),
                        if ui.get_opt_dry_run() {
                            "Preview"
                        } else {
                            "Apply Selected"
                        }
                    ),
                );
            }
        });
    }

    // ── Run Repair Now (UI overhaul: Attention card dedicated button) ────────────────────────
    // Fires ONLY the repair pass, bypassing the opt-* toggles. Intended for the common
    // "my system is broken, just fix it" case. Honors the Dry run toggle so users can still
    // preview. After the run, populate_detection re-polls state and the Attention card
    // dismisses itself when the scanner returns empty.
    {
        let weak = ui.as_weak();
        ui.on_run_repair(move || {
            let Some(ui) = weak.upgrade() else { return };
            let mode = if ui.get_opt_dry_run() {
                ExecutionMode::DryRun
            } else {
                ExecutionMode::Apply
            };
            ui.set_is_working(true);
            append_log(
                &ui,
                &format!(
                    "--- Running repair ({}) ---",
                    if mode.is_dry_run() { "dry-run" } else { "apply" }
                ),
            );

            let thread_weak = ui.as_weak();
            thread::spawn(move || {
                let ctx = Context::production(mode);
                let form = hardware::get_chassis_type(&ctx.paths.dmi_chassis)
                    .unwrap_or(FormFactor::Unknown);
                let gpus = GpuInventory::detect().unwrap_or_default();

                let progress_weak = thread_weak.clone();
                let mut progress = move |line: &str| {
                    let weak = progress_weak.clone();
                    let line = line.to_string();
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(ui) = weak.upgrade() {
                            append_log(&ui, &line);
                        }
                    });
                };

                let actions = Actions {
                    repair: true,
                    ..Actions::default()
                };
                // Repair fires pacman -Rns + dkms autoinstall — must be non-interactive
                // under a GUI that has no TTY for [Y/n] prompts.
                let result = core::run_actions(&ctx, form, &gpus, actions, true, &mut progress);

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = thread_weak.upgrade() {
                        match result {
                            Ok(reports) => {
                                if reports.is_empty() {
                                    append_log(
                                        &ui,
                                        "No repairs needed — system is already clean.",
                                    );
                                } else {
                                    for (section, r) in &reports {
                                        append_log(&ui, &format!("[{section}] {r}"));
                                    }
                                    append_log(
                                        &ui,
                                        &format!(
                                            "Repair complete ({} action{}).",
                                            reports.len(),
                                            if reports.len() == 1 { "" } else { "s" }
                                        ),
                                    );
                                }
                            }
                            Err(e) => append_log(&ui, &format!("ERROR: {e:#}")),
                        }
                        populate_detection(&ui);
                        ui.set_is_working(false);
                    }
                });
            });
        });
    }

    // ── Diagnostics scan ─────────────────────────────────────────────────────────────────────
    {
        let weak = ui.as_weak();
        ui.on_diagnose(move || {
            let Some(ui) = weak.upgrade() else { return };
            ui.set_is_working(true);
            append_log(&ui, "--- Diagnostics scan ---");

            let thread_weak = ui.as_weak();
            thread::spawn(move || {
                let ctx = Context::production(ExecutionMode::DryRun);
                let form = hardware::get_chassis_type(&ctx.paths.dmi_chassis).unwrap_or(FormFactor::Unknown);
                let gpus = GpuInventory::detect().unwrap_or_default();
                let findings = diagnostics::scan(&ctx, &gpus, form);

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = thread_weak.upgrade() {
                        let (mut i, mut w, mut e) = (0u32, 0u32, 0u32);
                        for f in &findings {
                            match f.severity {
                                Severity::Info => i += 1,
                                Severity::Warning => w += 1,
                                Severity::Error => e += 1,
                            }
                            append_log(&ui, &format!("{} {}", f.severity.marker(), f.title));
                            if !f.detail.is_empty() {
                                append_log(&ui, &format!("       {}", f.detail));
                            }
                            if let Some(fix) = &f.fix_hint {
                                append_log(&ui, &format!("  fix: {fix}"));
                            }
                        }
                        append_log(
                            &ui,
                            &format!("Summary: {i} info / {w} warnings / {e} errors."),
                        );
                        ui.set_is_working(false);
                    }
                });
            });
        });
    }

    // ── Apply / Preview ──────────────────────────────────────────────────────────────────────
    // Phase 6 live-streaming pattern preserved: the progress callback pipes each pacman /
    // mkinitcpio / grub-mkconfig / bootctl / makepkg / yay line into the log through
    // slint::invoke_from_event_loop, while the heavy work runs on a worker thread. Phase 11
    // adds: after the run finishes, we re-poll state so the [✓ Applied] badges update.
    {
        let weak = ui.as_weak();
        ui.on_apply(move || {
            let Some(ui) = weak.upgrade() else { return };

            let actions = Actions {
                wayland: ui.get_opt_wayland(),
                bootloader: ui.get_opt_bootloader(),
                power: ui.get_opt_power(),
                gaming: ui.get_opt_gaming(),
                repair: ui.get_opt_repair(),
                essentials: ui.get_opt_essentials(),
                groups: ui.get_opt_groups(),
                cleanup: ui.get_opt_cleanup(),
                troubleshoot: ui.get_opt_troubleshoot(),
            };
            if !actions.any() {
                append_log(&ui, "No actions selected — nothing to do.");
                return;
            }

            let mode = if ui.get_opt_dry_run() {
                ExecutionMode::DryRun
            } else {
                ExecutionMode::Apply
            };
            let assume_yes = ui.get_opt_assume_yes();

            ui.set_is_working(true);
            append_log(
                &ui,
                &format!(
                    "--- Running {} actions ---",
                    if mode.is_dry_run() {
                        "dry-run"
                    } else {
                        "apply"
                    }
                ),
            );

            let thread_weak = ui.as_weak();
            thread::spawn(move || {
                let ctx = Context::production(mode);
                let form = hardware::get_chassis_type(&ctx.paths.dmi_chassis).unwrap_or(FormFactor::Unknown);
                let gpus = GpuInventory::detect().unwrap_or_default();

                let progress_weak = thread_weak.clone();
                let mut progress = move |line: &str| {
                    let weak = progress_weak.clone();
                    let line = line.to_string();
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(ui) = weak.upgrade() {
                            append_log(&ui, &line);
                        }
                    });
                };

                let result =
                    core::run_actions(&ctx, form, &gpus, actions, assume_yes, &mut progress);

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = thread_weak.upgrade() {
                        match result {
                            Ok(reports) => {
                                let mut last_section: &str = "";
                                for (section, r) in reports {
                                    if section != last_section {
                                        append_log(&ui, &format!("[{section}]"));
                                        last_section = section;
                                    }
                                    append_log(&ui, &format!("  {r}"));
                                }
                                append_log(&ui, "Done.");
                            }
                            Err(e) => {
                                append_log(&ui, &format!("ERROR: {e:#}"));
                            }
                        }
                        // Re-poll state so [✓ Applied] badges update without needing a detect click.
                        populate_detection(&ui);
                        ui.set_is_working(false);
                    }
                });
            });
        });
    }

    ui.run()?;
    Ok(())
}

fn populate_detection(ui: &MainWindow) {
    let paths = SystemPaths::production();
    let ctx = Context::production(ExecutionMode::DryRun);

    let (chassis_desc, is_laptop, form) = match hardware::get_chassis_type(&paths.dmi_chassis) {
        Ok(FormFactor::Laptop) => ("Laptop".to_string(), true, FormFactor::Laptop),
        Ok(FormFactor::Desktop) => ("Desktop".to_string(), false, FormFactor::Desktop),
        Ok(FormFactor::Unknown) => (
            "Unknown form factor".to_string(),
            false,
            FormFactor::Unknown,
        ),
        Err(e) => (format!("detection failed: {e}"), false, FormFactor::Unknown),
    };
    ui.set_chassis_text(SharedString::from(chassis_desc));
    ui.set_is_laptop(is_laptop);

    let gpus = GpuInventory::detect().unwrap_or_default();
    let has_nvidia = gpus.has_nvidia();
    ui.set_has_nvidia(has_nvidia);
    ui.set_has_amd(gpus.has_amd());
    ui.set_has_intel(gpus.has_intel());
    ui.set_has_intel_i915(gpus.has_intel_i915());
    ui.set_is_hybrid(gpus.is_hybrid());
    ui.set_gpus_text(SharedString::from(format_gpu_inventory(&gpus)));

    // Per-vendor detail strings for the new tabbed UI. Each line summarizes the detected
    // GPU + (for NVIDIA) the driver variant the tool will install. Empty when the vendor
    // isn't present — the corresponding tab is hidden anyway via has-* guards.
    ui.set_nvidia_detail_text(SharedString::from(format_nvidia_detail(&gpus)));
    ui.set_nvidia_driver_recommendation(SharedString::from(format_nvidia_driver_recommendation(&gpus)));
    ui.set_amd_detail_text(SharedString::from(format_amd_detail(&gpus)));
    ui.set_intel_detail_text(SharedString::from(format_intel_detail(&gpus)));

    // Bootloader type + source path
    let bt = bootloader::detect_active_bootloader(&ctx);
    ui.set_bootloader_type_text(SharedString::from(bt.human()));
    let bootloader_desc = describe_bootloader_source(&ctx, bt);
    ui.set_bootloader_text(SharedString::from(bootloader_desc));

    // Per-tweak state flags → drives the Switch disabled-state + [✓ Applied] / Unsupported badges.
    apply_tweak_states(ui, &ctx, &gpus, form);
    // Phase 15/16 sanitation — populate the amber warning banner.
    apply_sanitation_warnings(ui, &ctx, &gpus);

    // Pre-arm toggles from a real recommendation so startup ≡ Auto-Optimize.
    let recommended = auto::recommend(&ctx, form, &gpus);
    ui.set_opt_wayland(recommended.wayland);
    ui.set_opt_bootloader(recommended.bootloader);
    ui.set_opt_power(recommended.power);
    ui.set_opt_gaming(recommended.gaming);
    ui.set_opt_repair(recommended.repair);
}

fn apply_tweak_states(ui: &MainWindow, ctx: &Context, gpus: &GpuInventory, form: FormFactor) {
    // Phase 17: three exclusive per-tweak flags drive the Slint UI:
    //   active         → green "✓ Active" badge (kernel confirms running)
    //   pending_reboot → yellow "⟳ Reboot pending" badge (config done, kernel not yet)
    //   incompatible   → orange "Unsupported" badge (not applicable to this host)
    // When all three are false the Switch is rendered.
    let w = wayland::check_state(ctx, gpus);
    ui.set_state_wayland_applied(w.is_active());
    ui.set_state_wayland_pending_reboot(w.is_pending_reboot());
    ui.set_state_wayland_incompatible(w.is_incompatible());

    let b = bootloader::check_state(ctx, gpus);
    ui.set_state_bootloader_applied(b.is_active());
    ui.set_state_bootloader_pending_reboot(b.is_pending_reboot());
    ui.set_state_bootloader_incompatible(b.is_incompatible());

    let p = power::check_state(ctx, gpus);
    ui.set_state_power_applied(p.is_active());
    ui.set_state_power_pending_reboot(p.is_pending_reboot());
    ui.set_state_power_incompatible(p.is_incompatible());

    let g = gaming::check_state(ctx, gpus, form);
    ui.set_state_gaming_applied(g.is_active());
    ui.set_state_gaming_pending_reboot(g.is_pending_reboot());

    // Phase 30: per-tweak state for the cross-vendor Overview-tab cards.
    let e = essentials::check_state(ctx, gpus);
    ui.set_state_essentials_applied(e.is_active());

    let gr = groups::check_state(ctx);
    ui.set_state_groups_applied(gr.is_active());
    // Incompatible means "running without a detectable non-root invoking user"
    // — usermod has no target. Surfaced as the Unsupported badge.
    ui.set_state_groups_incompatible(matches!(gr, TweakState::Incompatible));

    let c = cleanup::check_state(ctx, gpus);
    ui.set_state_cleanup_applied(c.is_active());

    // troubleshoot has no Active state by design — recipes can always run; each
    // recipe reports its own per-run verification into the console.

    // Phase 30 audit M6: pre-arm the Essentials and Groups toggles at startup
    // so the GUI opens in the same state Auto-Optimize would produce. Only
    // when check_state says Unapplied — Active-state tweaks stay off so the
    // green "✓ Active" badge renders. Cleanup and troubleshoot remain off
    // (opt-in invariant from Phases 28/29).
    let rec = auto::recommend(ctx, form, gpus);
    if rec.essentials {
        ui.set_opt_essentials(true);
    }
    if rec.groups {
        ui.set_opt_groups(true);
    }

    // Phase 20 + UI overhaul: repair tweak — Active iff scanner finds nothing to heal.
    // The UI renders a green health pill in the hero when clean, a red-amber Attention
    // card below Hardware when not. `repair_summaries` feeds the bullets inside that card
    // so users see the specific issues without having to press Diagnose first.
    let rep = repair::check_state(ctx, gpus, form);
    ui.set_state_repair_applied(rep.is_active());
    let actions = repair::scan(ctx, gpus, form);
    ui.set_repair_action_count(actions.len() as i32);
    let summaries: Vec<SharedString> = actions
        .iter()
        .map(|a| SharedString::from(a.human_summary()))
        .collect();
    let summary_model: Rc<VecModel<SharedString>> = Rc::new(VecModel::from(summaries));
    ui.set_repair_summaries(ModelRc::from(summary_model));
}

/// Phase 15/16: poll the gaming + wayland sanitation scanners and push each warning as a
/// one-line summary into the GUI's `sanitation-warnings` list-model. The Slint side renders
/// them as amber bullets in a dedicated banner above the Tweaks card when non-empty.
fn apply_sanitation_warnings(ui: &MainWindow, ctx: &Context, gpus: &GpuInventory) {
    let mut lines: Vec<SharedString> = Vec::new();
    for w in gaming::sanitation_warnings(gpus) {
        // Show the title + remediation hint in a single line. Full detail goes through the
        // diagnostics scanner for users who want to dig in.
        lines.push(SharedString::from(format!(
            "{} — {}",
            w.title, w.remediation
        )));
    }
    for w in wayland::sanitation_warnings(ctx) {
        lines.push(SharedString::from(format!(
            "{} — {}",
            w.title(),
            w.remediation()
        )));
    }
    let model: Rc<VecModel<SharedString>> = Rc::new(VecModel::from(lines));
    ui.set_sanitation_warnings(ModelRc::from(model));
}

fn describe_bootloader_source(ctx: &Context, bt: BootloaderType) -> String {
    match bt {
        BootloaderType::Uki => format!("cmdline at {}", ctx.paths.kernel_cmdline.display()),
        BootloaderType::Grub => format!("cmdline at {}", ctx.paths.grub_default.display()),
        BootloaderType::SystemdBoot => format!("entries at {}", ctx.paths.sdb_entries.display()),
        BootloaderType::Limine => ctx
            .paths
            .limine_candidates
            .iter()
            .find(|p| p.exists())
            .map(|p| format!("cmdline in {}", p.display()))
            .unwrap_or_else(|| "limine.conf not found".to_string()),
        BootloaderType::Unknown => "no supported bootloader detected".to_string(),
    }
}

/// NVIDIA-specific detail line for the NVIDIA tab's detection card. Summarizes the
/// primary NVIDIA GPU's product name + architecture generation. Empty when no NVIDIA
/// GPU is detected (the NVIDIA tab is hidden in that case).
fn format_nvidia_detail(gpus: &GpuInventory) -> String {
    let Some(nv) = gpus.primary_nvidia() else {
        return String::new();
    };
    let gen = nv
        .nvidia_gen
        .map(|g| format!(" · {}", g.human()))
        .unwrap_or_default();
    format!("{}{gen}", nv.display_name())
}

/// NVIDIA driver-package recommendation line. Resolves the per-architecture driver
/// (nvidia-open-dkms for Turing+, nvidia-580xx-dkms AUR for Maxwell/Volta/Pascal, AUR legacy for
/// Maxwell/Kepler/Fermi). Empty when there's no NVIDIA GPU or no recommendation.
fn format_nvidia_driver_recommendation(gpus: &GpuInventory) -> String {
    let Some(nv) = gpus.primary_nvidia() else {
        return String::new();
    };
    let Some(rec) = nv.recommended_nvidia_package() else {
        return String::new();
    };
    let tag = match rec.source {
        PackageSource::Official => "repo",
        PackageSource::Aur => "AUR",
        PackageSource::Unsupported => "EOL",
    };
    format!("{} ({tag}) — {}", rec.package, rec.note)
}

/// AMD-specific detail line. Names every detected AMD GPU and its kernel driver so
/// users on hybrids can see exactly what amdgpu is bound to.
fn format_amd_detail(gpus: &GpuInventory) -> String {
    let amds: Vec<String> = gpus
        .gpus
        .iter()
        .filter(|g| matches!(g.vendor, crate::core::gpu::GpuVendor::Amd))
        .map(|g| {
            let driver = g.kernel_driver.as_deref().unwrap_or("(none)");
            let tag = if g.is_integrated { "iGPU" } else { "dGPU" };
            format!("{}  ·  {tag}  ·  kernel driver: {driver}", g.display_name())
        })
        .collect();
    if amds.is_empty() {
        String::new()
    } else {
        amds.join("\n")
    }
}

/// Intel-specific detail line. Shows the kernel driver (i915 vs xe) so users know
/// whether the cmdline tweak (i915-only) applies to them.
fn format_intel_detail(gpus: &GpuInventory) -> String {
    let intels: Vec<String> = gpus
        .gpus
        .iter()
        .filter(|g| matches!(g.vendor, crate::core::gpu::GpuVendor::Intel))
        .map(|g| {
            let driver = g.kernel_driver.as_deref().unwrap_or("(none)");
            let tag = if g.is_integrated { "iGPU" } else { "dGPU" };
            format!("{}  ·  {tag}  ·  kernel driver: {driver}", g.display_name())
        })
        .collect();
    if intels.is_empty() {
        String::new()
    } else {
        intels.join("\n")
    }
}

fn format_gpu_inventory(gpus: &GpuInventory) -> String {
    if gpus.gpus.is_empty() {
        return "(no display controllers detected)".to_string();
    }
    let mut lines = Vec::new();
    for g in &gpus.gpus {
        let mut line = format!(
            "• {:?}  {}{}",
            g.vendor,
            g.display_name(),
            if g.is_integrated {
                "   (integrated)"
            } else {
                ""
            }
        );
        if let Some(gen) = g.nvidia_gen {
            line.push_str(&format!("\n    arch: {}", gen.human()));
        }
        if let Some(rec) = g.recommended_nvidia_package() {
            let tag = match rec.source {
                PackageSource::Official => "repo",
                PackageSource::Aur => "AUR",
                PackageSource::Unsupported => "EOL",
            };
            line.push_str(&format!(
                "\n    driver: {} ({tag}) — {}",
                rec.package, rec.note
            ));
        }
        lines.push(line);
    }
    lines.join("\n")
}

fn append_log(ui: &MainWindow, line: &str) {
    let mut buf = ui.get_log_text().to_string();
    if !buf.is_empty() {
        buf.push('\n');
    }
    buf.push_str(line);
    ui.set_log_text(SharedString::from(buf));
    log::info!("{line}");

    // Phase 11 auto-scroll (post-bugfix):
    //
    // Primary trigger is the Slint-side `changed viewport-height` handler on the ScrollView
    // — it fires AFTER Slint's layout pass recomputes the Text's intrinsic height, so
    // viewport-height is guaranteed current and the snap lands on the true new bottom.
    //
    // Belt-and-suspenders: we ALSO explicitly invoke the public scroll function, but deferred
    // through `invoke_from_event_loop`. That pushes the call onto the next event-loop tick,
    // giving Slint a chance to process `set_log_text` and re-layout before we read
    // viewport-height in the function. Without this deferral, calling invoke_* synchronously
    // here would read the pre-append viewport-height and land one frame short of the bottom.
    let weak = ui.as_weak();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(ui) = weak.upgrade() {
            ui.invoke_scroll_console_to_end();
        }
    });
}
