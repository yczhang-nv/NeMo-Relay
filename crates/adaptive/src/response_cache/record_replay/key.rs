// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::config::KeyFiltersConfig;
use crate::config::ResponseCacheConfig;
use crate::response_cache::key::{fingerprint, record_replay_document};
use nemo_relay::api::llm::LlmRequest;
use regex::{Regex, RegexBuilder};
use serde_json::{Value as Json, json};
use std::collections::BTreeMap;

pub const REDACTED: &str = "<relay-redacted>";
pub const TOOL_RESULT: &str = "<relay-tool-result>";

pub struct KeyBuilder {
    filters: KeyFiltersConfig,
    patterns: Vec<Regex>,
    policy: Json,
    pub policy_hash: String,
}
pub struct KeyedRequest {
    pub hash: Option<String>,
    pub reason: Option<String>,
    pub filtered: Json,
    pub matches: BTreeMap<String, u64>,
    pub document: Option<Json>,
}

impl KeyBuilder {
    pub fn new(cache: &ResponseCacheConfig, filters: KeyFiltersConfig) -> Self {
        let patterns = filters
            .redact_lines
            .iter()
            .map(|p| {
                RegexBuilder::new(p)
                    .multi_line(true)
                    .build()
                    .expect("validated regex")
            })
            .collect();
        let policy = json!({"filter_version":2, "key_filters":filters, "redacted_placeholder":REDACTED,
            "tool_result_placeholder":TOOL_RESULT, "namespace":cache.namespace, "key_strategy":cache.key_strategy,
            "cache_schema_version":crate::response_cache::store::CACHE_SCHEMA_VERSION});
        let policy_hash = fingerprint(&policy).expect("validated key policy");
        Self {
            filters,
            patterns,
            policy,
            policy_hash,
        }
    }
    pub fn policy(&self) -> &Json {
        &self.policy
    }
    pub fn build(
        &self,
        operation: &str,
        request: &LlmRequest,
        cache: &ResponseCacheConfig,
    ) -> KeyedRequest {
        let mut filtered = request.content.clone();
        let mut matches = BTreeMap::new();
        for pointer in &self.filters.drop {
            matches.insert(format!("drop:{pointer}"), remove(&mut filtered, pointer));
        }
        if let Some(body) = filtered.as_object_mut() {
            body.remove("extra_headers");
            body.remove("headers");
        }
        for i in 0..self.patterns.len() {
            matches.insert(format!("redact_lines:{i}"), 0);
        }
        for i in 0..self.filters.redact_blocks.len() {
            matches.insert(format!("redact_blocks:{i}"), 0);
        }
        let mut reason = None;
        self.filter_messages(&mut filtered, &mut matches, &mut reason);
        if let Some(instructions) = filtered.get_mut("instructions") {
            self.redact_content(instructions, &mut matches, &mut reason);
        }
        let document = if reason.is_none() {
            record_replay_document(
                operation,
                &LlmRequest {
                    content: filtered.clone(),
                    ..request.clone()
                },
                cache,
                &json!(self.filters),
            )
        } else {
            None
        };
        let hash = document.as_ref().and_then(fingerprint);
        if hash.is_none() && reason.is_none() {
            reason = Some(
                if filtered.is_null() {
                    "unparseable_body"
                } else if document.is_none() {
                    "unrepresentable_number"
                } else {
                    "canonicalization_failed"
                }
                .into(),
            );
        }
        KeyedRequest {
            hash,
            reason,
            filtered,
            matches,
            document,
        }
    }
    fn filter_messages(
        &self,
        filtered: &mut Json,
        matches: &mut BTreeMap<String, u64>,
        reason: &mut Option<String>,
    ) {
        if let Some(input) = filtered.get_mut("input").filter(|input| input.is_string()) {
            self.redact_content(input, matches, reason);
        }
        for key in ["messages", "input"] {
            if let Some(messages) = filtered.get_mut(key).and_then(Json::as_array_mut) {
                for message in messages {
                    if let Some(content) = message.get_mut("content") {
                        self.redact_content(content, matches, reason);
                    }
                    if self.filters.tool_results == "placeholder" {
                        if message.get("role").and_then(Json::as_str) == Some("tool") {
                            message["content"] = json!(TOOL_RESULT);
                        }
                        if message.get("type").and_then(Json::as_str)
                            == Some("function_call_output")
                        {
                            message["output"] = json!(TOOL_RESULT);
                        }
                        if let Some(parts) = message.get_mut("content").and_then(Json::as_array_mut)
                        {
                            for part in parts {
                                if part.get("type").and_then(Json::as_str) == Some("tool_result") {
                                    part["content"] = json!(TOOL_RESULT);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    fn redact_content(
        &self,
        value: &mut Json,
        counts: &mut BTreeMap<String, u64>,
        reason: &mut Option<String>,
    ) {
        match value {
            Json::String(text) => {
                for (i, block) in self.filters.redact_blocks.iter().enumerate() {
                    let mut result = String::new();
                    let mut rest = text.as_str();
                    let mut count = 0;
                    while let Some(start) = rest.find(&block.start) {
                        result.push_str(&rest[..start]);
                        let tail = &rest[start + block.start.len()..];
                        let Some(end) = tail.find(&block.end) else {
                            *reason = Some("unterminated_block".into());
                            return;
                        };
                        result.push_str(REDACTED);
                        count += 1;
                        rest = &tail[end + block.end.len()..];
                    }
                    result.push_str(rest);
                    *text = result;
                    *counts.entry(format!("redact_blocks:{i}")).or_default() += count;
                }
                for (i, regex) in self.patterns.iter().enumerate() {
                    let count = regex.find_iter(text).count() as u64;
                    *text = regex.replace_all(text, REDACTED).into_owned();
                    *counts.entry(format!("redact_lines:{i}")).or_default() += count;
                }
            }
            Json::Array(parts) => {
                for part in parts {
                    if let Some(text) = part.get_mut("text") {
                        self.redact_content(text, counts, reason);
                    }
                }
            }
            _ => {}
        }
    }
}

fn remove(value: &mut Json, pointer: &str) -> u64 {
    if pointer.is_empty() {
        let present = !value.is_null();
        *value = Json::Null;
        return u64::from(present);
    }
    let Some(path) = pointer.strip_prefix('/') else {
        return 0;
    };
    let tokens: Vec<_> = path
        .split('/')
        .map(|token| token.replace("~1", "/").replace("~0", "~"))
        .collect();
    remove_path(value, &tokens)
}

fn remove_path(value: &mut Json, tokens: &[String]) -> u64 {
    let Some((key, rest)) = tokens.split_first() else {
        return 0;
    };
    match value {
        Json::Object(map) => {
            // An asterisk is a literal object key, just as in RFC 6901.
            if rest.is_empty() {
                u64::from(map.remove(key).is_some())
            } else {
                map.get_mut(key).map_or(0, |child| remove_path(child, rest))
            }
        }
        Json::Array(items) if key == "*" => {
            if rest.is_empty() {
                let count = items.len() as u64;
                items.clear();
                count
            } else {
                items.iter_mut().map(|child| remove_path(child, rest)).sum()
            }
        }
        Json::Array(items) => match key.parse::<usize>() {
            Ok(i) if i < items.len() && *key == i.to_string() => {
                if rest.is_empty() {
                    items.remove(i);
                    1
                } else {
                    remove_path(&mut items[i], rest)
                }
            }
            _ => 0,
        },
        _ => 0,
    }
}

pub fn first_difference(a: &Json, b: &Json) -> Option<Json> {
    fn walk(a: &Json, b: &Json, path: String) -> Option<Json> {
        if a == b {
            return None;
        }
        if let (Some(a), Some(b)) = (a.as_object(), b.as_object()) {
            let mut keys: Vec<_> = a.keys().chain(b.keys()).collect();
            keys.sort_by_cached_key(|key| key.encode_utf16().collect::<Vec<_>>());
            keys.dedup();
            for key in keys {
                let escaped = key.replace('~', "~0").replace('/', "~1");
                match (a.get(key), b.get(key)) {
                    (Some(a), Some(b)) => {
                        if let Some(diff) = walk(a, b, format!("{path}/{escaped}")) {
                            return Some(diff);
                        }
                    }
                    _ => {
                        return Some(
                            json!({"pointer":format!("{path}/{escaped}"),"recorded":a.get(key),"live":b.get(key)}),
                        );
                    }
                }
            }
        } else if let (Some(a), Some(b)) = (a.as_array(), b.as_array()) {
            for i in 0..a.len().max(b.len()) {
                match (a.get(i), b.get(i)) {
                    (Some(a), Some(b)) => {
                        if let Some(diff) = walk(a, b, format!("{path}/{i}")) {
                            return Some(diff);
                        }
                    }
                    _ => {
                        return Some(
                            json!({"pointer":format!("{path}/{i}"),"recorded":a.get(i),"live":b.get(i)}),
                        );
                    }
                }
            }
        } else if let (Some(a), Some(b)) = (a.as_str(), b.as_str()) {
            let offset = a
                .chars()
                .zip(b.chars())
                .position(|(a, b)| a != b)
                .unwrap_or(a.chars().count().min(b.chars().count()));
            return Some(
                json!({"pointer":path,"character_offset":offset,"recorded_length":a.chars().count(),"live_length":b.chars().count(),
                "recorded":a.chars().skip(offset.saturating_sub(80)).take(200).collect::<String>(),
                "live":b.chars().skip(offset.saturating_sub(80)).take(200).collect::<String>()}),
            );
        }
        fn bounded(v: &Json) -> Json {
            let text = v.to_string();
            if text.len() > 4096 {
                json!({"truncated":text.chars().take(200).collect::<String>(),"sha256":fingerprint(v)})
            } else {
                v.clone()
            }
        }
        Some(json!({"pointer":path,"recorded":bounded(a),"live":bounded(b)}))
    }
    walk(a, b, String::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn array_wildcards_preserve_object_keys_escapes_and_index_rules() {
        let mut value = json!({
            "input": [{"id":"a"}, {"id":"b"}, {}, null, [{"id":"nested"}]],
            "*": {"a/b": [{"~id":"x"}, {"~id":"y"}]},
            "object": {"*": {"id":"literal"}, "other": {"id":"keep"}},
            "list": [[0,1],[2,3]]
        });
        assert_eq!(remove(&mut value, "/input/*/id"), 2);
        assert_eq!(value["input"][4][0]["id"], "nested");
        assert_eq!(remove(&mut value, "/*/a~1b/*/~0id"), 2);
        assert_eq!(remove(&mut value, "/object/*/id"), 1);
        assert_eq!(value["object"]["other"]["id"], "keep");
        for path in ["/list/01/0", "/list/+0/0", "/list/-/0", "/list/9/0"] {
            assert_eq!(remove(&mut value, path), 0);
        }
        assert_eq!(remove(&mut value, "/list/*/0"), 2);
        assert_eq!(value["list"], json!([[1], [3]]));
        assert_eq!(remove(&mut value, "/list/*/*"), 2);
        assert_eq!(value["list"], json!([[], []]));
        assert_eq!(remove(&mut value, "/input/*/missing"), 0);
        assert_eq!(remove(&mut value, "/input/*"), 5);
        assert_eq!(value["input"], json!([]));
    }

    #[test]
    fn responses_volatile_message_fields_do_not_hide_semantic_changes() {
        let cache = ResponseCacheConfig::default();
        let key = KeyBuilder::new(
            &cache,
            KeyFiltersConfig {
                drop: vec![
                    "/input/*/id".into(),
                    "/input/*/internal_chat_message_metadata_passthrough/create_time".into(),
                    "/input/*/internal_chat_message_metadata_passthrough/turn_id".into(),
                ],
                ..Default::default()
            },
        );
        let a = LlmRequest {
            content: json!({"model":"m", "input":[
                {"id":"msg_1", "role":"user", "content":"stable", "internal_chat_message_metadata_passthrough": {
                    "create_time":123.5, "turn_id":"t1", "content_item_kinds":["text"]}},
                {"id":"msg_2", "type":"function_call", "name":"exec_command", "call_id":"call1", "arguments":"{}"}
            ]}),
            headers: Default::default(),
        };
        let mut b = a.clone();
        b.content["input"][0]["id"] = json!("msg_new");
        b.content["input"][1]["id"] = json!("msg_other");
        b.content["input"][0]["internal_chat_message_metadata_passthrough"]["create_time"] =
            json!(999.1);
        b.content["input"][0]["internal_chat_message_metadata_passthrough"]["turn_id"] =
            json!("t2");
        let recorded = key.build("responses.create", &a, &cache);
        let live = key.build("responses.create", &b, &cache);
        assert_eq!(recorded.hash, live.hash);
        assert_eq!(live.matches["drop:/input/*/id"], 2);
        assert_eq!(
            live.matches["drop:/input/*/internal_chat_message_metadata_passthrough/create_time"],
            1
        );
        assert_eq!(key.policy()["filter_version"], 2);
        for path in [
            "/input/0/content",
            "/input/0/role",
            "/input/1/call_id",
            "/input/1/arguments",
            "/input/0/internal_chat_message_metadata_passthrough/content_item_kinds",
        ] {
            let mut changed = b.clone();
            *changed.content.pointer_mut(path).unwrap() = json!("semantic change");
            assert_ne!(
                recorded.hash,
                key.build("responses.create", &changed, &cache).hash,
                "{path}"
            );
        }
    }

    #[test]
    fn drift_matches_but_tool_arguments_and_prompts_do_not() {
        let cache = ResponseCacheConfig {
            namespace: "test".into(),
            ..Default::default()
        };
        let key = KeyBuilder::new(
            &cache,
            KeyFiltersConfig {
                tool_results: "placeholder".into(),
                redact_lines: vec!["^Time: .*$".into()],
                ..Default::default()
            },
        );
        let a = LlmRequest {
            content: json!({"model":"m","temperature":1,"messages":[{"role":"system","content":"Time: a\nStable"},{"role":"tool","tool_call_id":"a","content":"output A"}]}),
            headers: Default::default(),
        };
        let mut b = a.clone();
        b.content["messages"][0]["content"] = json!("Time: b\nStable");
        b.content["messages"][1]["content"] = json!("output B");
        assert_eq!(
            key.build("op", &a, &cache).hash,
            key.build("op", &b, &cache).hash
        );
        b.content["messages"][0]["content"] = json!("Different");
        assert_ne!(
            key.build("op", &a, &cache).hash,
            key.build("op", &b, &cache).hash
        );
        assert_eq!(
            first_difference(&json!({"x":"abc"}), &json!({"x":"abd"})).unwrap()["character_offset"],
            2
        );
        let input = |time: &str| LlmRequest {
            content: json!({"model":"m","input":format!("Time: {time}\nStable")}),
            headers: Default::default(),
        };
        assert_eq!(
            key.build("op", &input("a"), &cache).hash,
            key.build("op", &input("b"), &cache).hash
        );
    }
}
