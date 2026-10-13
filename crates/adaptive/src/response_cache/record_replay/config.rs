// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tolerant, warning-only parsing: a malformed mode must still intercept calls.

use std::collections::HashSet;
use std::path::PathBuf;

use nemo_relay::plugin::{ConfigDiagnostic, DiagnosticLevel};
use regex::RegexBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordReplayConfig {
    pub mode: String,
    pub trace_directory: PathBuf,
    pub output_directory: PathBuf,
    pub latency_scale: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_gate: Option<StartGateConfig>,
    pub key_filters: KeyFiltersConfig,
    pub divergence: DivergenceConfig,
}

impl Default for RecordReplayConfig {
    fn default() -> Self {
        Self {
            mode: "replay".into(),
            trace_directory: PathBuf::new(),
            output_directory: PathBuf::new(),
            latency_scale: 1.0,
            start_gate: None,
            key_filters: KeyFiltersConfig::default(),
            divergence: DivergenceConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartGateConfig {
    pub path: PathBuf,
    pub timeout_seconds: u64,
}
impl Default for StartGateConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            timeout_seconds: 3600,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RedactBlock {
    pub start: String,
    pub end: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyFiltersConfig {
    pub drop: Vec<String>,
    pub redact_lines: Vec<String>,
    pub redact_blocks: Vec<RedactBlock>,
    pub tool_results: String,
}
impl Default for KeyFiltersConfig {
    fn default() -> Self {
        Self {
            drop: vec![],
            redact_lines: vec![],
            redact_blocks: vec![],
            tool_results: "keep".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceConfig {
    pub tool_names: Vec<String>,
    pub tool_fields: Vec<String>,
    pub flag_fields: Vec<String>,
    pub output_size_tolerance: f64,
    pub measure_tool_cpu_time: bool,
}
impl Default for DivergenceConfig {
    fn default() -> Self {
        Self {
            tool_names: vec![],
            tool_fields: vec!["/exit_code".into(), "/error".into()],
            flag_fields: vec!["/error".into()],
            output_size_tolerance: 0.5,
            measure_tool_cpu_time: false,
        }
    }
}

pub struct RecordReplayParse {
    pub config: Option<RecordReplayConfig>,
    pub output_directory: Option<PathBuf>,
    pub diagnostics: Vec<ConfigDiagnostic>,
}

pub fn diagnostic(code: &str, field: &str, message: &str) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level: DiagnosticLevel::Warning,
        code: format!("response_cache.record_replay_{code}"),
        component: Some("response_cache".into()),
        field: Some(field.into()),
        message: message.into(),
    }
}

pub fn valid_pointer(pointer: &str) -> bool {
    if !pointer.is_empty() && !pointer.starts_with('/') {
        return false;
    }
    let mut chars = pointer.chars();
    while let Some(c) = chars.next() {
        if c == '~' && !matches!(chars.next(), Some('0' | '1')) {
            return false;
        }
    }
    true
}

#[derive(Default)]
struct Parser {
    diagnostics: Vec<ConfigDiagnostic>,
}
impl Parser {
    fn error(&mut self, code: &str, field: &str, message: &str) {
        self.diagnostics.push(diagnostic(code, field, message));
    }
    fn table<'a>(
        &mut self,
        value: &'a Json,
        field: &str,
        keys: &[&str],
    ) -> Option<&'a Map<String, Json>> {
        let Some(map) = value.as_object() else {
            self.error("invalid_type", field, "expected a table");
            return None;
        };
        for key in map.keys() {
            if !keys.contains(&key.as_str()) {
                self.error(
                    "unknown_field",
                    &format!("{field}.{key}"),
                    "unknown record/replay field",
                );
            }
        }
        Some(map)
    }
    fn string(
        &mut self,
        map: &Map<String, Json>,
        key: &str,
        field: &str,
        required: bool,
        default: &str,
    ) -> String {
        match map.get(key) {
            None if required => {
                self.error("missing_field", field, "required field is missing");
                String::new()
            }
            None => default.into(),
            Some(Json::String(value)) => value.clone(),
            Some(_) => {
                self.error("invalid_type", field, "expected a string");
                String::new()
            }
        }
    }
    fn strings(
        &mut self,
        map: &Map<String, Json>,
        key: &str,
        field: &str,
        default: Vec<String>,
    ) -> Vec<String> {
        let Some(value) = map.get(key) else {
            return default;
        };
        let Some(items) = value.as_array() else {
            self.error("invalid_type", field, "expected a list");
            return vec![];
        };
        items
            .iter()
            .enumerate()
            .filter_map(|(i, value)| {
                if let Some(value) = value.as_str() {
                    Some(value.into())
                } else {
                    self.error(
                        "invalid_type",
                        &format!("{field}[{i}]"),
                        "expected a string",
                    );
                    None
                }
            })
            .collect()
    }
    fn number(&mut self, map: &Map<String, Json>, key: &str, field: &str, default: f64) -> f64 {
        match map.get(key) {
            None => default,
            Some(value) => match value.as_f64().filter(|v| v.is_finite()) {
                Some(value) => value,
                None => {
                    self.error("invalid_type", field, "expected a finite number");
                    default
                }
            },
        }
    }
    fn pointers(&mut self, values: &[String], field: &str, code: &str) {
        for (i, value) in values.iter().enumerate() {
            if !valid_pointer(value) {
                self.error(
                    code,
                    &format!("{field}[{i}]"),
                    "expected an RFC 6901 pointer",
                );
            }
        }
    }
}

pub fn parse_record_replay(value: &Json) -> RecordReplayParse {
    let mut p = Parser::default();
    let mut config = RecordReplayConfig::default();
    let output_directory = value
        .get("output_directory")
        .and_then(Json::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from);
    if let Some(map) = p.table(
        value,
        "record_replay",
        &[
            "mode",
            "trace_directory",
            "output_directory",
            "latency_scale",
            "start_gate",
            "key_filters",
            "divergence",
        ],
    ) {
        config.mode = p.string(map, "mode", "record_replay.mode", true, "");
        if map.get("mode").is_some_and(Json::is_string)
            && !["record", "verify", "replay"].contains(&config.mode.as_str())
        {
            p.error(
                "unsupported_mode",
                "record_replay.mode",
                "expected record, verify or replay",
            );
        }
        for key in ["trace_directory", "output_directory"] {
            let value = p.string(map, key, &format!("record_replay.{key}"), true, "");
            if map.get(key).is_some_and(Json::is_string) && value.trim().is_empty() {
                p.error(
                    "invalid_type",
                    &format!("record_replay.{key}"),
                    "expected a non-empty path",
                );
            }
            if key == "trace_directory" {
                config.trace_directory = value.into();
            } else {
                config.output_directory = value.into();
            }
        }
        config.latency_scale = p.number(map, "latency_scale", "record_replay.latency_scale", 1.0);
        if !(0.0..=1000.0).contains(&config.latency_scale) {
            p.error(
                "invalid_latency_scale",
                "record_replay.latency_scale",
                "scale must be between 0 and 1000",
            );
        }
        parse_gate(&mut p, &mut config, map);
        parse_filters(&mut p, &mut config, map);
        parse_divergence(&mut p, &mut config, map);
    }
    RecordReplayParse {
        config: p.diagnostics.is_empty().then_some(config),
        output_directory,
        diagnostics: p.diagnostics,
    }
}

fn parse_gate(
    p: &mut Parser,
    config: &mut RecordReplayConfig,
    map: &serde_json::Map<String, Json>,
) {
    if let Some(value) = map.get("start_gate")
        && let Some(gate) = p.table(
            value,
            "record_replay.start_gate",
            &["path", "timeout_seconds"],
        )
    {
        let path = p.string(gate, "path", "record_replay.start_gate.path", true, "");
        let timeout = match gate.get("timeout_seconds") {
            None => 3600,
            Some(value) => value.as_u64().unwrap_or_else(|| {
                p.error(
                    "invalid_type",
                    "record_replay.start_gate.timeout_seconds",
                    "expected a positive integer",
                );
                0
            }),
        };
        if path.trim().is_empty() || !(1..=604800).contains(&timeout) {
            p.error(
                "invalid_gate",
                "record_replay.start_gate",
                "gate requires a path and a timeout between 1 and 604800 seconds",
            );
        }
        config.start_gate = Some(StartGateConfig {
            path: path.into(),
            timeout_seconds: timeout,
        });
    }
}

fn parse_filters(
    p: &mut Parser,
    config: &mut RecordReplayConfig,
    map: &serde_json::Map<String, Json>,
) {
    if let Some(value) = map.get("key_filters")
        && let Some(filters) = p.table(
            value,
            "record_replay.key_filters",
            &["drop", "redact_lines", "redact_blocks", "tool_results"],
        )
    {
        config.key_filters.drop =
            p.strings(filters, "drop", "record_replay.key_filters.drop", vec![]);
        p.pointers(
            &config.key_filters.drop,
            "record_replay.key_filters.drop",
            "key_filters_invalid_pointer",
        );
        config.key_filters.redact_lines = p.strings(
            filters,
            "redact_lines",
            "record_replay.key_filters.redact_lines",
            vec![],
        );
        for (i, pattern) in config.key_filters.redact_lines.iter().enumerate() {
            if RegexBuilder::new(pattern).multi_line(true).build().is_err() {
                p.error(
                    "key_filters_invalid_regex",
                    &format!("record_replay.key_filters.redact_lines[{i}]"),
                    "regex does not compile",
                );
            }
        }
        config.key_filters.tool_results = p.string(
            filters,
            "tool_results",
            "record_replay.key_filters.tool_results",
            false,
            "keep",
        );
        if !["keep", "placeholder"].contains(&config.key_filters.tool_results.as_str()) {
            p.error(
                "unsupported_value",
                "record_replay.key_filters.tool_results",
                "expected keep or placeholder",
            );
        }
        if let Some(value) = filters.get("redact_blocks") {
            if let Some(blocks) = value.as_array() {
                for (i, value) in blocks.iter().enumerate() {
                    let field = format!("record_replay.key_filters.redact_blocks[{i}]");
                    if let Some(block) = p.table(value, &field, &["start", "end"]) {
                        let start = block
                            .get("start")
                            .and_then(Json::as_str)
                            .unwrap_or_default();
                        let end = block.get("end").and_then(Json::as_str).unwrap_or_default();
                        if start.is_empty() || end.is_empty() {
                            p.error(
                                "key_filters_invalid_block",
                                &field,
                                "block requires non-empty start and end literals",
                            );
                        } else {
                            config.key_filters.redact_blocks.push(RedactBlock {
                                start: start.into(),
                                end: end.into(),
                            });
                        }
                    }
                }
            } else {
                p.error(
                    "invalid_type",
                    "record_replay.key_filters.redact_blocks",
                    "expected a list",
                );
            }
        }
    }
}

fn parse_divergence(
    p: &mut Parser,
    config: &mut RecordReplayConfig,
    map: &serde_json::Map<String, Json>,
) {
    if let Some(value) = map.get("divergence")
        && let Some(div) = p.table(
            value,
            "record_replay.divergence",
            &[
                "tool_names",
                "tool_fields",
                "flag_fields",
                "output_size_tolerance",
                "measure_tool_cpu_time",
            ],
        )
    {
        config.divergence.tool_names = p.strings(
            div,
            "tool_names",
            "record_replay.divergence.tool_names",
            vec![],
        );
        let mut seen = HashSet::new();
        for name in &config.divergence.tool_names {
            if name.is_empty() || !seen.insert(name) {
                p.error(
                    "invalid_divergence",
                    "record_replay.divergence.tool_names",
                    "tool names must be non-empty and unique",
                );
            }
        }
        config.divergence.tool_fields = p.strings(
            div,
            "tool_fields",
            "record_replay.divergence.tool_fields",
            config.divergence.tool_fields.clone(),
        );
        config.divergence.flag_fields = p.strings(
            div,
            "flag_fields",
            "record_replay.divergence.flag_fields",
            config.divergence.flag_fields.clone(),
        );
        p.pointers(
            &config.divergence.tool_fields.clone(),
            "record_replay.divergence.tool_fields",
            "invalid_divergence",
        );
        p.pointers(
            &config.divergence.flag_fields.clone(),
            "record_replay.divergence.flag_fields",
            "invalid_divergence",
        );
        if config
            .divergence
            .flag_fields
            .iter()
            .any(|p| !config.divergence.tool_fields.contains(p))
        {
            p.error(
                "invalid_divergence",
                "record_replay.divergence.flag_fields",
                "flags must be a subset of outcome fields",
            );
        }
        config.divergence.output_size_tolerance = p.number(
            div,
            "output_size_tolerance",
            "record_replay.divergence.output_size_tolerance",
            0.5,
        );
        if config.divergence.output_size_tolerance < 0.0 {
            p.error(
                "invalid_divergence",
                "record_replay.divergence.output_size_tolerance",
                "tolerance must be non-negative",
            );
        }
        if let Some(value) = div.get("measure_tool_cpu_time") {
            if let Some(value) = value.as_bool() {
                config.divergence.measure_tool_cpu_time = value;
            } else {
                p.error(
                    "invalid_type",
                    "record_replay.divergence.measure_tool_cpu_time",
                    "expected a boolean",
                );
            }
        }
    }
}

fn block_schema() -> &'static nemo_relay::config_editor::EditorSchema {
    <RedactBlock as nemo_relay::config_editor::EditorConfig>::editor_schema()
}
fn block_default() -> Json {
    serde_json::json!({"start":"","end":""})
}
static BLOCK_ITEM: nemo_relay::config_editor::EditorListItemSpec =
    nemo_relay::config_editor::EditorListItemSpec {
        kind: nemo_relay::config_editor::EditorFieldKind::Section,
        schema: Some(block_schema),
        default: Some(block_default),
        tagged_union: None,
        list_item: None,
    };
nemo_relay::editor_config! {impl RedactBlock {
    start=>{label:"start",kind:String}, end=>{label:"end",kind:String},
}}
nemo_relay::editor_config! {impl StartGateConfig {
    path=>{label:"path",kind:String},timeout_seconds=>{label:"timeout_seconds",kind:Integer},
}}
nemo_relay::editor_config! {impl KeyFiltersConfig {
    drop=>{label:"drop",kind:List,list:&nemo_relay::config_editor::STRING_LIST_ITEM},
    redact_lines=>{label:"redact_lines",kind:List,list:&nemo_relay::config_editor::STRING_LIST_ITEM},
    redact_blocks=>{label:"redact_blocks",kind:List,list:&BLOCK_ITEM},
    tool_results=>{label:"tool_results",kind:Enum,values:["keep","placeholder"]},
}}
nemo_relay::editor_config! {impl DivergenceConfig {
    tool_names=>{label:"tool_names",kind:List,list:&nemo_relay::config_editor::STRING_LIST_ITEM},
    tool_fields=>{label:"tool_fields",kind:List,list:&nemo_relay::config_editor::STRING_LIST_ITEM},
    flag_fields=>{label:"flag_fields",kind:List,list:&nemo_relay::config_editor::STRING_LIST_ITEM},
    output_size_tolerance=>{label:"output_size_tolerance",kind:Float},measure_tool_cpu_time=>{label:"measure_tool_cpu_time",kind:Boolean},
}}
nemo_relay::editor_config! {impl RecordReplayConfig {
    mode=>{label:"mode",kind:Enum,values:["record","verify","replay"]},
    trace_directory=>{label:"trace_directory",kind:String},output_directory=>{label:"output_directory",kind:String},
    latency_scale=>{label:"latency_scale",kind:Float},
    start_gate=>{label:"start_gate",kind:Section,optional:true,nested:StartGateConfig,default:StartGateConfig},
    key_filters=>{label:"key_filters",kind:Section,nested:KeyFiltersConfig,default:KeyFiltersConfig},
    divergence=>{label:"divergence",kind:Section,nested:DivergenceConfig,default:DivergenceConfig},
}}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn malformed_sections_are_warnings_and_preserve_output() {
        for value in [
            Json::Null,
            json!(false),
            json!([]),
            json!({"mode": "bogus", "trace_directory": 7, "output_directory": "/tmp/out", "latency_scale": null, "key_filters": {"drop": [3, "bad~2"], "redact_lines": ["("], "unknown": true}}),
        ] {
            let parsed = parse_record_replay(&value);
            assert!(parsed.config.is_none());
            assert!(!parsed.diagnostics.is_empty());
            assert!(
                parsed
                    .diagnostics
                    .iter()
                    .all(|d| d.level == DiagnosticLevel::Warning)
            );
        }
        let parsed = parse_record_replay(&json!({"output_directory":"/tmp/out"}));
        assert_eq!(parsed.output_directory, Some(PathBuf::from("/tmp/out")));
        assert_eq!(parsed.diagnostics.len(), 2);
    }
    #[test]
    fn defaults_and_integer_scale_parse() {
        let parsed = parse_record_replay(
            &json!({"mode":"record", "trace_directory":"/tmp/trace", "output_directory":"/tmp/out", "latency_scale":0}),
        );
        assert!(parsed.diagnostics.is_empty());
        assert_eq!(parsed.config.unwrap().key_filters.tool_results, "keep");
        assert!(valid_pointer("/a~1b/~0"));
        assert!(!valid_pointer("/a~2b"));
    }
}
