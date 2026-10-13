// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::fs;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CpuSnapshot {
    pub self_user_us: u64,
    pub self_sys_us: u64,
    pub children_user_us: u64,
    pub children_sys_us: u64,
    pub cgroup_usage_us: Option<u64>,
    #[serde(skip)]
    pub live_descendants_us: u64,
}

#[cfg(not(unix))]
pub fn snapshot() -> CpuSnapshot {
    CpuSnapshot::default()
}

#[cfg(unix)]
pub fn snapshot() -> CpuSnapshot {
    fn usage(who: libc::c_int) -> (u64, u64) {
        let mut value = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: getrusage initializes the provided rusage on success.
        if unsafe { libc::getrusage(who, value.as_mut_ptr()) } != 0 {
            return (0, 0);
        }
        let value = unsafe { value.assume_init() };
        let us = |t: libc::timeval| {
            (t.tv_sec as u64)
                .saturating_mul(1_000_000)
                .saturating_add(t.tv_usec as u64)
        };
        (us(value.ru_utime), us(value.ru_stime))
    }
    let (self_user_us, self_sys_us) = usage(libc::RUSAGE_SELF);
    let (children_user_us, children_sys_us) = usage(libc::RUSAGE_CHILDREN);
    let cgroup_usage_us = fs::read_to_string("/bench/cgroup/cpu.stat")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|line| {
                line.strip_prefix("usage_usec ")
                    .and_then(|v| v.trim().parse().ok())
            })
        });
    CpuSnapshot {
        self_user_us,
        self_sys_us,
        children_user_us,
        children_sys_us,
        cgroup_usage_us,
        live_descendants_us: live_cpu(),
    }
}

pub fn delta(a: &CpuSnapshot, b: &CpuSnapshot) -> (u64, u64, &'static str) {
    let harness = (b.self_user_us + b.self_sys_us).saturating_sub(a.self_user_us + a.self_sys_us);
    if let (Some(a), Some(b)) = (a.cgroup_usage_us, b.cgroup_usage_us) {
        (
            harness,
            b.saturating_sub(a).saturating_sub(harness),
            "cgroup",
        )
    } else {
        let start = a.children_user_us + a.children_sys_us + a.live_descendants_us;
        let end = b.children_user_us + b.children_sys_us + b.live_descendants_us;
        (harness, end.saturating_sub(start), "fallback")
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
fn live_cpu() -> u64 {
    0
}

#[cfg(target_os = "linux")]
fn live_cpu() -> u64 {
    use std::collections::{HashMap, HashSet};
    let mut processes = HashMap::new();
    let Ok(entries) = fs::read_dir("/proc") else {
        return 0;
    };
    for entry in entries.flatten() {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        let Ok(stat) = fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        let Some((_, tail)) = stat.rsplit_once(") ") else {
            continue;
        };
        let parts: Vec<_> = tail.split_whitespace().collect();
        if parts.len() < 15 {
            continue;
        }
        let Ok(parent) = parts[1].parse::<u32>() else {
            continue;
        };
        // utime, stime, cutime, cstime: include children reaped by a live kernel.
        let ticks = parts[11..15]
            .iter()
            .filter_map(|v| v.parse::<u64>().ok())
            .sum::<u64>();
        processes.insert(pid, (parent, ticks));
    }
    let mut descendants = HashSet::from([std::process::id()]);
    loop {
        let before = descendants.len();
        for (pid, (parent, _)) in &processes {
            if descendants.contains(parent) {
                descendants.insert(*pid);
            }
        }
        if descendants.len() == before {
            break;
        }
    }
    // SAFETY: sysconf takes a constant selector and no pointer.
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if ticks <= 0 {
        return 0;
    }
    processes
        .iter()
        .filter(|(pid, _)| **pid != std::process::id() && descendants.contains(pid))
        .map(|(_, (_, cpu))| *cpu)
        .sum::<u64>()
        .saturating_mul(1_000_000)
        / ticks as u64
}
