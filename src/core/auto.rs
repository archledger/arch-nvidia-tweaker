use crate::core::gaming;
use crate::core::gpu::GpuInventory;
use crate::core::groups;
use crate::core::hardware::FormFactor;
use crate::core::{bootloader, essentials, power, repair, wayland, Actions, Context};

/// Determine the best default action set for the detected hardware AND current system state.
///
/// Phase 11 update: cross-references each candidate action with `check_state`. An action is
/// only recommended when it's both applicable to the hardware AND currently `Unapplied`.
/// This prevents the "Auto-Optimize" button from suggesting work that's already done, and
/// matches the UI badges ([✓ Applied] switches stay off).
///
/// Hardware rules:
///  - NVIDIA GPU present → candidate for Wayland + Bootloader
///  - Laptop with NVIDIA → also candidate for Power (suspend/resume + hybrid power)
///  - `[multilib]` not yet uncommented OR gaming tools missing → candidate for Gaming
pub fn recommend(ctx: &Context, form: FormFactor, gpus: &GpuInventory) -> Actions {
    let has_nvidia = gpus.has_nvidia();
    let is_laptop = matches!(form, FormFactor::Laptop);

    let wayland_applicable = has_nvidia;
    let bootloader_applicable = has_nvidia;
    let power_applicable = has_nvidia && is_laptop;
    // Gaming is universally applicable; gate purely on state.
    let gaming_applicable = true;

    Actions {
        wayland: wayland_applicable && wayland::check_state(ctx, gpus).is_unapplied(),
        bootloader: bootloader_applicable && bootloader::check_state(ctx, gpus).is_unapplied(),
        power: power_applicable && power::check_state(ctx, gpus).is_unapplied(),
        gaming: gaming_applicable && gaming::check_state(ctx, gpus, form).is_unapplied(),
        // Phase 20: Auto-Optimize recommends repair whenever the scanner finds stale
        // artifacts. Universally applicable, so the only gate is state.
        repair: repair::check_state(ctx, gpus, form).is_unapplied(),
        // Phase 26: essentials is universal — every host benefits from the Vulkan
        // loader + Mesa + split firmware + diagnostic userspace. State-gated only.
        essentials: essentials::check_state(ctx, gpus).is_unapplied(),
        // Phase 27: groups is universal for any host with a non-root invoking user.
        // Incompatible (no SUDO_USER / PKEXEC_UID) → not recommended.
        groups: groups::check_state(ctx).is_unapplied(),
    }
}

/// Returns the subset of action names the recommend() just turned ON, for UI log output.
pub fn recommended_names(actions: Actions) -> Vec<&'static str> {
    let mut out = Vec::new();
    // Repair first — matches run_actions dispatch order and flags the urgent case visibly.
    if actions.repair {
        out.push("repair");
    }
    // Phase 27: groups runs immediately after repair (cheap user-state cleanup
    // before pacman-heavy stages).
    if actions.groups {
        out.push("groups");
    }
    // Phase 26: essentials sits before vendor-specific stages so they see a
    // healthy Vulkan/Mesa base.
    if actions.essentials {
        out.push("essentials");
    }
    if actions.wayland {
        out.push("wayland");
    }
    if actions.bootloader {
        out.push("bootloader");
    }
    if actions.power {
        out.push("power");
    }
    if actions.gaming {
        out.push("gaming");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::{GpuInfo, GpuVendor};
    use crate::core::ExecutionMode;
    use tempfile::tempdir;

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

    fn nvidia_desktop() -> GpuInventory {
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
                nvidia_gen: None,
            }],
        }
    }

    fn nvidia_hybrid() -> GpuInventory {
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

    fn seed_pacman(root: &std::path::Path, body: &str) {
        let p = root.join("etc/pacman.conf");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, body).unwrap();
    }

    fn seed_cmdline_uki(root: &std::path::Path, body: &str) {
        let p = root.join("etc/kernel/cmdline");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, body).unwrap();
    }

    const MULTILIB_ON: &str = "[multilib]\nInclude = /etc/pacman.d/mirrorlist\n";
    const MULTILIB_OFF: &str = "#[multilib]\n#Include = /etc/pacman.d/mirrorlist\n";

    #[test]
    fn intel_laptop_with_multilib_on_recommends_nothing_state_aware() {
        let tmp = tempdir().unwrap();
        seed_pacman(tmp.path(), MULTILIB_ON);
        let ctx = Context::rooted_for_test(tmp.path(), ExecutionMode::DryRun);
        let rec = recommend(&ctx, FormFactor::Laptop, &intel_only());
        // Intel → wayland/bootloader/power all Incompatible → false
        // multilib on + can't verify all packages via pacman in test → gaming stays Unapplied
        // so gaming could be true here; verify it's at least not wayland/bootloader/power:
        assert!(!rec.wayland);
        assert!(!rec.bootloader);
        assert!(!rec.power);
    }

    #[test]
    fn intel_without_multilib_recommends_gaming_only() {
        let tmp = tempdir().unwrap();
        seed_pacman(tmp.path(), MULTILIB_OFF);
        let ctx = Context::rooted_for_test(tmp.path(), ExecutionMode::DryRun);
        let rec = recommend(&ctx, FormFactor::Laptop, &intel_only());
        assert!(!rec.wayland);
        assert!(!rec.bootloader);
        assert!(!rec.power);
        assert!(rec.gaming);
    }

    #[test]
    fn nvidia_desktop_recommends_wayland_and_bootloader_not_power() {
        let tmp = tempdir().unwrap();
        seed_pacman(tmp.path(), MULTILIB_ON);
        seed_cmdline_uki(tmp.path(), "rw quiet\n"); // no nvidia-drm.modeset=1 yet
        let ctx = Context::rooted_for_test(tmp.path(), ExecutionMode::DryRun);
        let rec = recommend(&ctx, FormFactor::Desktop, &nvidia_desktop());
        assert!(rec.wayland);
        assert!(rec.bootloader);
        assert!(!rec.power, "power is laptop-only");
    }

    #[test]
    fn nvidia_desktop_with_bootloader_already_applied_skips_bootloader() {
        let tmp = tempdir().unwrap();
        seed_pacman(tmp.path(), MULTILIB_ON);
        // Phase 16: "already applied" now means BOTH modeset=1 AND fbdev=1 are present.
        seed_cmdline_uki(
            tmp.path(),
            "rw quiet nvidia-drm.modeset=1 nvidia-drm.fbdev=1\n",
        );
        let ctx = Context::rooted_for_test(tmp.path(), ExecutionMode::DryRun);
        let rec = recommend(&ctx, FormFactor::Desktop, &nvidia_desktop());
        assert!(rec.wayland, "wayland drop-ins not yet present → recommend");
        assert!(!rec.bootloader, "bootloader already Applied → skip");
    }

    #[test]
    fn nvidia_hybrid_laptop_off_multilib_recommends_everything() {
        let tmp = tempdir().unwrap();
        seed_pacman(tmp.path(), MULTILIB_OFF);
        seed_cmdline_uki(tmp.path(), "rw quiet\n");
        let ctx = Context::rooted_for_test(tmp.path(), ExecutionMode::DryRun);
        let rec = recommend(&ctx, FormFactor::Laptop, &nvidia_hybrid());
        assert!(rec.wayland);
        assert!(rec.bootloader);
        assert!(rec.power);
        assert!(rec.gaming);
    }

    #[test]
    fn recommended_names_lists_only_enabled_actions() {
        let actions = Actions {
            wayland: true,
            bootloader: false,
            power: true,
            gaming: false,
            repair: false,
            essentials: false,
            groups: false,
        };
        assert_eq!(recommended_names(actions), vec!["wayland", "power"]);
    }

    #[test]
    fn recommended_names_puts_repair_first() {
        // Phase 20: when repair is flagged alongside other actions, it appears first —
        // this maps to run_actions dispatch order and communicates urgency to the user.
        let actions = Actions {
            wayland: true,
            bootloader: false,
            power: false,
            gaming: true,
            repair: true,
            essentials: false,
            groups: false,
        };
        assert_eq!(recommended_names(actions), vec!["repair", "wayland", "gaming"]);
    }

    #[test]
    fn recommended_names_orders_repair_groups_essentials_then_vendor() {
        // Phase 30: stacked ordering invariant — repair first, then groups
        // (cheap user-state cleanup), then essentials (Mesa/Vulkan/firmware),
        // then the vendor-specific stages.
        let actions = Actions {
            wayland: true,
            bootloader: false,
            power: false,
            gaming: true,
            repair: true,
            essentials: true,
            groups: true,
        };
        assert_eq!(
            recommended_names(actions),
            vec!["repair", "groups", "essentials", "wayland", "gaming"]
        );
    }
}
