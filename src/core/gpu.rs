use anyhow::{Context as _, Result};
use std::path::PathBuf;
use std::process::Command;

pub const NVIDIA_VENDOR_ID: u16 = 0x10de;
pub const AMD_VENDOR_ID: u16 = 0x1002;
pub const INTEL_VENDOR_ID: u16 = 0x8086;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Other,
}

impl GpuVendor {
    pub fn from_id(id: u16) -> Self {
        match id {
            NVIDIA_VENDOR_ID => Self::Nvidia,
            AMD_VENDOR_ID => Self::Amd,
            INTEL_VENDOR_ID => Self::Intel,
            _ => Self::Other,
        }
    }
}

/// Coarse NVIDIA architecture classification, by PCI device ID range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvidiaGeneration {
    BlackwellOrNewer,
    Ada,
    Hopper,
    Ampere,
    Turing,
    Volta,
    Pascal,
    Maxwell,
    Kepler,
    Fermi,
    TeslaOrOlder,
    Unknown,
}

impl NvidiaGeneration {
    pub fn from_device_id(id: u16) -> Self {
        match id {
            0x2b00..=0x2fff => Self::BlackwellOrNewer,
            0x2600..=0x2aff => Self::Ada,
            0x2330..=0x233f => Self::Hopper,
            0x2080..=0x25ff => Self::Ampere,
            0x1e00..=0x207f => Self::Turing,
            0x1d80..=0x1dff => Self::Volta,
            0x1b00..=0x1d7f => Self::Pascal,
            0x1300..=0x1aff => Self::Maxwell,
            0x0fc0..=0x12ff => Self::Kepler,
            0x06c0..=0x0fbf => Self::Fermi,
            0x0190..=0x06bf => Self::TeslaOrOlder,
            _ => Self::Unknown,
        }
    }

    /// Turing (2018) and newer support the "open" kernel modules (GSP-RM). As of the
    /// current NVIDIA 595 production branch (early 2026), the proprietary tree has
    /// dropped Pascal support, and Arch's `[extra]` repo ships only `nvidia-open` /
    /// `nvidia-open-dkms` — the old `nvidia` / `nvidia-dkms` package names are gone
    /// from the official repo (legacy branches live in AUR: `nvidia-580xx-dkms`
    /// covers Maxwell/Volta/Pascal, 470xx covers Kepler, 390xx covers Fermi).
    pub fn supports_open_modules(self) -> bool {
        matches!(
            self,
            Self::BlackwellOrNewer | Self::Ada | Self::Hopper | Self::Ampere | Self::Turing,
        )
    }

    pub fn human(self) -> &'static str {
        match self {
            Self::BlackwellOrNewer => "Blackwell / RTX 50-series or newer",
            Self::Ada => "Ada Lovelace / RTX 40-series",
            Self::Hopper => "Hopper / H100-series",
            Self::Ampere => "Ampere / RTX 30-series",
            Self::Turing => "Turing / RTX 20-series, GTX 16-series",
            Self::Volta => "Volta / Titan V, V100",
            Self::Pascal => "Pascal / GTX 10-series, Titan Xp",
            Self::Maxwell => "Maxwell / GTX 9-series, 750 Ti",
            Self::Kepler => "Kepler / GTX 6-series, 7-series",
            Self::Fermi => "Fermi / GTX 4-series, 5-series",
            Self::TeslaOrOlder => "Tesla or older (8/9/200-series)",
            Self::Unknown => "unknown architecture",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub vendor_id: u16,
    pub device_id: u16,
    pub pci_address: String,
    pub vendor_name: String,
    pub product_name: String,
    pub kernel_driver: Option<String>,
    pub is_integrated: bool,
    pub nvidia_gen: Option<NvidiaGeneration>,
}

impl GpuInfo {
    pub fn display_name(&self) -> String {
        let product = self.product_name.trim();
        if product.is_empty() {
            format!("{} device {:04x}", self.vendor_name.trim(), self.device_id)
        } else {
            format!("{} {}", self.vendor_name.trim(), product)
        }
    }

    // Phase 15: kernel-driver classification. The `kernel_driver` field is populated by
    // `read_driver_name` via `/sys/bus/pci/devices/<addr>/driver` — more reliable than
    // parsing `lspci -k` because we read a structured symlink.

    /// Intel GPU using the modern Xe kernel driver (Lunar Lake / Battlemage / Xe2+).
    pub fn uses_xe_driver(&self) -> bool {
        self.vendor == GpuVendor::Intel && self.kernel_driver.as_deref() == Some("xe")
    }
    /// Intel GPU using the classic i915 kernel driver (Gen2 … Gen12 / Alder Lake-era).
    pub fn uses_i915_driver(&self) -> bool {
        self.vendor == GpuVendor::Intel && self.kernel_driver.as_deref() == Some("i915")
    }
    /// AMD GPU using the modern amdgpu kernel driver (GCN 1.2+ / Southern Islands onward).
    pub fn uses_amdgpu_driver(&self) -> bool {
        self.vendor == GpuVendor::Amd && self.kernel_driver.as_deref() == Some("amdgpu")
    }
    /// AMD GPU using the legacy `radeon` driver (pre-GCN Terascale, or user-pinned).
    /// Modern Arch ships amdgpu.si=1/cik=1 so even some older cards default to amdgpu;
    /// seeing `radeon` in 2026 typically means a very old card or an explicit pin.
    pub fn uses_radeon_legacy_driver(&self) -> bool {
        self.vendor == GpuVendor::Amd && self.kernel_driver.as_deref() == Some("radeon")
    }

    pub fn recommended_nvidia_package(&self) -> Option<NvidiaDriverRecommendation> {
        if self.vendor != GpuVendor::Nvidia {
            return None;
        }
        let gen = self.nvidia_gen.unwrap_or(NvidiaGeneration::Unknown);
        Some(match gen {
            NvidiaGeneration::BlackwellOrNewer
            | NvidiaGeneration::Ada
            | NvidiaGeneration::Hopper
            | NvidiaGeneration::Ampere
            | NvidiaGeneration::Turing => NvidiaDriverRecommendation {
                package: "nvidia-open-dkms",
                source: PackageSource::Official,
                note: "open kernel modules (GSP-based, recommended from 560+)",
            },
            // Phase 24 correction: `nvidia-dkms` is no longer in Arch's `extra` repo
            // (only `nvidia-open-dkms` remains). Pascal / Volta / Maxwell share the
            // 580.x driver branch via the `nvidia-580xx-dkms` AUR package (maintained
            // by ventureo / CachyOS team, verified 2026-04). 580 is the last NVIDIA
            // branch with Pascal support — 595 dropped it. Open kernel modules require
            // GSP firmware which only exists from Turing onward, so these gens also
            // can't use the open variant.
            NvidiaGeneration::Volta
            | NvidiaGeneration::Pascal
            | NvidiaGeneration::Maxwell => NvidiaDriverRecommendation {
                package: "nvidia-580xx-dkms",
                source: PackageSource::Aur,
                note: "legacy 580-series driver (Maxwell/Volta/Pascal); AUR only — nvidia-dkms is no longer in official repos",
            },
            NvidiaGeneration::Kepler => NvidiaDriverRecommendation {
                package: "nvidia-470xx-dkms",
                source: PackageSource::Aur,
                note: "legacy 470-series driver (Kepler); AUR only",
            },
            NvidiaGeneration::Fermi => NvidiaDriverRecommendation {
                package: "nvidia-390xx-dkms",
                source: PackageSource::Aur,
                note: "legacy 390-series driver (Fermi); AUR only",
            },
            NvidiaGeneration::TeslaOrOlder => NvidiaDriverRecommendation {
                package: "(unsupported)",
                source: PackageSource::Unsupported,
                note: "Tesla-or-older GPUs are EOL on modern NVIDIA drivers; use nouveau.",
            },
            NvidiaGeneration::Unknown => NvidiaDriverRecommendation {
                package: "nvidia-open-dkms",
                source: PackageSource::Official,
                note: "unknown device ID — defaulting to open modules; verify manually",
            },
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageSource {
    Official,
    Aur,
    Unsupported,
}

#[derive(Debug, Clone)]
pub struct NvidiaDriverRecommendation {
    pub package: &'static str,
    pub source: PackageSource,
    pub note: &'static str,
}

#[derive(Debug, Clone, Default)]
pub struct GpuInventory {
    pub gpus: Vec<GpuInfo>,
}

impl GpuInventory {
    pub fn detect() -> Result<Self> {
        let out = Command::new("lspci")
            .args(["-mm", "-nn", "-D"])
            .output()
            .context("running lspci — install `pciutils` if missing")?;
        if !out.status.success() {
            anyhow::bail!(
                "lspci exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let text = String::from_utf8_lossy(&out.stdout);
        Ok(Self {
            gpus: parse_lspci_mm_output(&text),
        })
    }

    pub fn has_nvidia(&self) -> bool {
        self.gpus.iter().any(|g| g.vendor == GpuVendor::Nvidia)
    }

    pub fn has_intel(&self) -> bool {
        self.gpus.iter().any(|g| g.vendor == GpuVendor::Intel)
    }

    pub fn has_amd(&self) -> bool {
        self.gpus.iter().any(|g| g.vendor == GpuVendor::Amd)
    }

    // Phase 15: driver-family convenience methods, for bootloader param selection.
    pub fn has_intel_xe(&self) -> bool {
        self.gpus.iter().any(|g| g.uses_xe_driver())
    }
    pub fn has_intel_i915(&self) -> bool {
        self.gpus.iter().any(|g| g.uses_i915_driver())
    }
    pub fn has_amd_amdgpu(&self) -> bool {
        self.gpus.iter().any(|g| g.uses_amdgpu_driver())
    }
    pub fn has_amd_radeon_legacy(&self) -> bool {
        self.gpus.iter().any(|g| g.uses_radeon_legacy_driver())
    }

    pub fn nvidia_gpus(&self) -> impl Iterator<Item = &GpuInfo> {
        self.gpus.iter().filter(|g| g.vendor == GpuVendor::Nvidia)
    }

    pub fn is_hybrid(&self) -> bool {
        self.has_nvidia()
            && self
                .gpus
                .iter()
                .any(|g| g.vendor != GpuVendor::Nvidia && g.is_integrated)
    }

    pub fn primary_nvidia(&self) -> Option<&GpuInfo> {
        self.nvidia_gpus().next()
    }
}

fn parse_lspci_mm_output(text: &str) -> Vec<GpuInfo> {
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(info) = parse_line(line) {
            out.push(info);
        }
    }
    out
}

fn parse_line(line: &str) -> Option<GpuInfo> {
    let tokens = tokenize_lspci_mm(line);
    if tokens.len() < 4 {
        return None;
    }

    let address = tokens[0].clone();
    let class_field = &tokens[1];
    let class_id = extract_hex_in_brackets(class_field);
    let is_display = matches!(class_id, Some(id) if (id & 0xff00) == 0x0300);
    if !is_display {
        return None;
    }

    let (vendor_name, vendor_id) = split_name_and_hex(&tokens[2])?;
    let (product_name, device_id) = split_name_and_hex(&tokens[3])?;

    let vendor = GpuVendor::from_id(vendor_id);
    let nvidia_gen = if vendor == GpuVendor::Nvidia {
        Some(NvidiaGeneration::from_device_id(device_id))
    } else {
        None
    };

    let is_integrated = address.starts_with("0000:00:");
    let kernel_driver = read_driver_name(&address);

    Some(GpuInfo {
        vendor,
        vendor_id,
        device_id,
        pci_address: address,
        vendor_name,
        product_name,
        kernel_driver,
        is_integrated,
        nvidia_gen,
    })
}

fn tokenize_lspci_mm(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    let mut in_quotes = false;
    for ch in line.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ' ' if !in_quotes => {
                if !buf.is_empty() {
                    tokens.push(std::mem::take(&mut buf));
                }
            }
            _ => buf.push(ch),
        }
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
}

fn extract_hex_in_brackets(field: &str) -> Option<u16> {
    let open = field.rfind('[')?;
    let close = field.rfind(']')?;
    if close <= open {
        return None;
    }
    u16::from_str_radix(&field[open + 1..close], 16).ok()
}

fn split_name_and_hex(field: &str) -> Option<(String, u16)> {
    let open = field.rfind('[')?;
    let close = field.rfind(']')?;
    if close <= open {
        return None;
    }
    let id = u16::from_str_radix(&field[open + 1..close], 16).ok()?;
    let name = field[..open].trim().to_string();
    Some((name, id))
}

fn read_driver_name(pci_address: &str) -> Option<String> {
    let link = PathBuf::from("/sys/bus/pci/devices")
        .join(pci_address)
        .join("driver");
    let tgt = std::fs::read_link(&link).ok()?;
    tgt.file_name().map(|n| n.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_from_id_maps_known_vendors() {
        assert_eq!(GpuVendor::from_id(0x10de), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_id(0x1002), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_id(0x8086), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_id(0xabcd), GpuVendor::Other);
    }

    #[test]
    fn nvidia_generation_classifies_real_device_ids() {
        let cases = [
            (0x2B85, NvidiaGeneration::BlackwellOrNewer),
            (0x2684, NvidiaGeneration::Ada),
            (0x2704, NvidiaGeneration::Ada),
            (0x2330, NvidiaGeneration::Hopper),
            (0x2204, NvidiaGeneration::Ampere),
            (0x2484, NvidiaGeneration::Ampere),
            (0x1E04, NvidiaGeneration::Turing),
            (0x1F82, NvidiaGeneration::Turing),
            (0x1D81, NvidiaGeneration::Volta),
            (0x1B06, NvidiaGeneration::Pascal),
            (0x13C2, NvidiaGeneration::Maxwell),
            (0x1180, NvidiaGeneration::Kepler),
            (0x1081, NvidiaGeneration::Kepler),
            (0x0DC4, NvidiaGeneration::Fermi),
        ];
        for (id, want) in cases {
            assert_eq!(
                NvidiaGeneration::from_device_id(id),
                want,
                "device 0x{id:04x}"
            );
        }
    }

    #[test]
    fn supports_open_modules_for_turing_plus() {
        assert!(NvidiaGeneration::Ada.supports_open_modules());
        assert!(NvidiaGeneration::Ampere.supports_open_modules());
        assert!(NvidiaGeneration::Turing.supports_open_modules());
        assert!(NvidiaGeneration::BlackwellOrNewer.supports_open_modules());
        assert!(!NvidiaGeneration::Pascal.supports_open_modules());
        assert!(!NvidiaGeneration::Maxwell.supports_open_modules());
    }

    #[test]
    fn parse_line_handles_intel_arc() {
        let line = r#"0000:00:02.0 "VGA compatible controller [0300]" "Intel Corporation [8086]" "Lunar Lake [Intel Arc Graphics 130V / 140V] [64a0]" -r04 "ASUSTeK Computer Inc. [1043]" "Device [1e13]""#;
        let info = parse_line(line).expect("should parse");
        assert_eq!(info.vendor, GpuVendor::Intel);
        assert_eq!(info.vendor_id, 0x8086);
        assert_eq!(info.device_id, 0x64a0);
        assert_eq!(info.pci_address, "0000:00:02.0");
        assert!(info.product_name.contains("Intel Arc Graphics 130V"));
        assert!(info.is_integrated);
        assert_eq!(info.nvidia_gen, None);
    }

    #[test]
    fn parse_line_handles_discrete_nvidia() {
        let line = r#"0000:01:00.0 "VGA compatible controller [0300]" "NVIDIA Corporation [10de]" "GA102 [GeForce RTX 3090] [2204]" -ra1 "NVIDIA Corporation [10de]" "Device [147d]""#;
        let info = parse_line(line).expect("should parse");
        assert_eq!(info.vendor, GpuVendor::Nvidia);
        assert_eq!(info.vendor_id, 0x10de);
        assert_eq!(info.device_id, 0x2204);
        assert!(!info.is_integrated);
        assert_eq!(info.nvidia_gen, Some(NvidiaGeneration::Ampere));
    }

    #[test]
    fn parse_line_skips_non_display_devices() {
        let line = r#"0000:00:1f.3 "Audio device [0403]" "Intel Corporation [8086]" "Alder Lake PCH-P High Definition Audio [51c8]""#;
        assert!(parse_line(line).is_none());
    }

    #[test]
    fn parse_line_matches_3d_controller_class() {
        let line = r#"0000:01:00.0 "3D controller [0302]" "NVIDIA Corporation [10de]" "GA107M [GeForce RTX 3050 Mobile] [25a2]""#;
        let info = parse_line(line).expect("should parse");
        assert_eq!(info.vendor, GpuVendor::Nvidia);
        assert_eq!(info.nvidia_gen, Some(NvidiaGeneration::Ampere));
    }

    #[test]
    fn recommended_package_for_modern_nvidia() {
        let g = GpuInfo {
            vendor: GpuVendor::Nvidia,
            vendor_id: 0x10de,
            device_id: 0x2684,
            pci_address: "0000:01:00.0".into(),
            vendor_name: "NVIDIA Corporation".into(),
            product_name: "AD102 [RTX 4090]".into(),
            kernel_driver: None,
            is_integrated: false,
            nvidia_gen: Some(NvidiaGeneration::Ada),
        };
        let rec = g
            .recommended_nvidia_package()
            .expect("nvidia should recommend");
        assert_eq!(rec.package, "nvidia-open-dkms");
        assert_eq!(rec.source, PackageSource::Official);
    }

    #[test]
    fn recommended_package_for_maxwell_is_580xx_aur() {
        // Phase 24: Maxwell moved from the 470xx branch to 580xx (the branch that
        // last supported Maxwell/Volta/Pascal before NVIDIA 595 dropped them).
        let g = GpuInfo {
            vendor: GpuVendor::Nvidia,
            vendor_id: 0x10de,
            device_id: 0x13C2,
            pci_address: "0000:01:00.0".into(),
            vendor_name: "NVIDIA Corporation".into(),
            product_name: "GM204 [GTX 980]".into(),
            kernel_driver: None,
            is_integrated: false,
            nvidia_gen: Some(NvidiaGeneration::Maxwell),
        };
        let rec = g.recommended_nvidia_package().unwrap();
        assert_eq!(rec.package, "nvidia-580xx-dkms");
        assert_eq!(rec.source, PackageSource::Aur);
    }

    #[test]
    fn non_nvidia_gets_no_recommendation() {
        let g = GpuInfo {
            vendor: GpuVendor::Intel,
            vendor_id: 0x8086,
            device_id: 0x64a0,
            pci_address: "0000:00:02.0".into(),
            vendor_name: "Intel".into(),
            product_name: "Arc 140V".into(),
            kernel_driver: None,
            is_integrated: true,
            nvidia_gen: None,
        };
        assert!(g.recommended_nvidia_package().is_none());
    }

    #[test]
    fn inventory_hybrid_detection() {
        let inv = GpuInventory {
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
                    product_name: "RTX 3050 Mobile".into(),
                    kernel_driver: None,
                    is_integrated: false,
                    nvidia_gen: Some(NvidiaGeneration::Ampere),
                },
            ],
        };
        assert!(inv.has_nvidia());
        assert!(inv.has_intel());
        assert!(inv.is_hybrid());
    }

    #[test]
    fn inventory_desktop_nvidia_is_not_hybrid() {
        let inv = GpuInventory {
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
        };
        assert!(inv.has_nvidia());
        assert!(!inv.is_hybrid());
    }
}
