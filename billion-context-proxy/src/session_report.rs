use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

use crate::AppState;

pub(crate) async fn handler(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let Some(session) = query
        .get("session")
        .or_else(|| query.get("conversationId"))
        .filter(|value| !value.trim().is_empty())
    else {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({"error": "missing session"})),
        )
            .into_response();
    };
    let Some(status) = state.sessions.status_json(session) else {
        return (
            StatusCode::NOT_FOUND,
            axum::Json(json!({"error": "unknown session"})),
        )
            .into_response();
    };
    let langs = query.get("lang").map(String::as_str).unwrap_or("zh-CN");
    let masked = query.get("mask").map(String::as_str) != Some("off");
    let shown_id = if masked {
        "s***".to_string()
    } else {
        session.to_string()
    };
    if query.get("format").map(String::as_str) == Some("svg") {
        return svg_report(&status, &shown_id, langs).into_response();
    }
    markdown_report(&status, &shown_id).into_response()
}

fn markdown_report(status: &Value, session_id: &str) -> Response {
    let context = status
        .get("contextTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let limit = status
        .get("contextLimit")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let pct = context.saturating_mul(100).checked_div(limit).unwrap_or(0);
    let mut lines = vec![
        "# billion-context session report".to_string(),
        String::new(),
        format!("- session: {session_id}"),
        format!("- context: {context} / {limit} tokens ({pct}%)"),
        format!(
            "- requests: {} · input {} · cached {} · output {}",
            status.get("requests").and_then(Value::as_u64).unwrap_or(0),
            status
                .get("inputTokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            status
                .get("cachedTokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            status
                .get("outputTokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        ),
        format!(
            "- model: {} · blocks: {} ({} active)",
            status.get("model").and_then(Value::as_str).unwrap_or("—"),
            status
                .get("blockCount")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            status
                .get("activeBlocks")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        ),
        String::new(),
        "## compression blocks".to_string(),
    ];
    if let Some(blocks) = status.get("blocks").and_then(Value::as_array) {
        if blocks.is_empty() {
            lines.push("_no blocks_".into());
        }
        for block in blocks {
            lines.push(String::new());
            lines.push(format!(
                "### {} (tier {}, {})",
                block.get("id").and_then(Value::as_str).unwrap_or("?"),
                block.get("tier").and_then(Value::as_u64).unwrap_or(0),
                if block
                    .get("active")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    "active"
                } else {
                    "inactive"
                }
            ));
            if let Some(topic) = block.get("topic").and_then(Value::as_str) {
                lines.push(format!("- topic: {topic}"));
            }
            if let Some(summary) = block.get("summary").and_then(Value::as_str) {
                lines.push(format!("- summary: {summary}"));
            }
            if let Some(span) = block.get("span").and_then(Value::as_array) {
                let start = span.first().and_then(Value::as_str).unwrap_or("?");
                let end = span.get(1).and_then(Value::as_str).unwrap_or("?");
                lines.push(format!("- span: {start} → {end}"));
            }
            lines.push(format!(
                "- tokens: {} · direct {} · effective {}",
                block.get("tokens").and_then(Value::as_u64).unwrap_or(0),
                block
                    .get("directCount")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                block
                    .get("effectiveCount")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            ));
        }
    }
    let mut response = (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/markdown; charset=utf-8")],
        lines.join("\n"),
    )
        .into_response();
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!("attachment; filename=\"session-{session_id}.md\""))
            .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
    );
    response
}

fn svg_report(status: &Value, session_id: &str, lang: &str) -> Response {
    let context = status
        .get("contextTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let limit = status
        .get("contextLimit")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let rows = [
        format!("session: {session_id}"),
        format!("lang: {lang}"),
        format!("context: {context} / {limit}"),
        format!(
            "requests: {}",
            status.get("requests").and_then(Value::as_u64).unwrap_or(0)
        ),
        format!(
            "tokens: input {} cached {} output {}",
            status
                .get("inputTokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            status
                .get("cachedTokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            status
                .get("outputTokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        ),
        format!(
            "blocks: {} ({} active)",
            status
                .get("blockCount")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            status
                .get("activeBlocks")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        ),
    ];
    let height = 70 + rows.len() * 24;
    let text: Vec<String> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            format!(
                "<text x=\"28\" y=\"{}\" font-family=\"monospace\" font-size=\"14\" fill=\"#1f2328\">{}</text>",
                72 + index * 24,
                xml_escape(row)
            )
        })
        .collect();
    let svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"720\" height=\"{height}\" viewBox=\"0 0 720 {height}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\
         <text x=\"28\" y=\"38\" font-family=\"sans-serif\" font-size=\"20\" font-weight=\"600\" fill=\"#1f2328\">billion-context session report</text>\
         {}</svg>",
        text.join("")
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "image/svg+xml; charset=utf-8")],
        svg,
    )
        .into_response()
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
