#![allow(unsafe_code)]
#![allow(clippy::cast_precision_loss)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use infe_parsers::StreamingParser;
use infe_parsers::registry::DialectRegistry;

fn bench_hermes_single(c: &mut Criterion) {
    let open = "<tool_call>";
    let close = "</tool_call>";
    let input =
        format!("{open}{{\"name\":\"get_weather\",\"arguments\":{{\"city\":\"London\"}}}}{close}");
    c.bench_function("hermes_single_chunk", |b| {
        b.iter(|| {
            let mut p = StreamingParser::new(DialectRegistry::create("hermes").unwrap());
            let _ = black_box(p.feed(black_box(&input)));
        });
    });
}

fn bench_llama3_json_single(c: &mut Criterion) {
    let input = r#"{"name":"get_weather","parameters":{"city":"London"}}"#;
    c.bench_function("llama3_json_single_chunk", |b| {
        b.iter(|| {
            let mut p = StreamingParser::new(DialectRegistry::create("llama3_json").unwrap());
            let _ = black_box(p.feed(black_box(input)));
        });
    });
}

fn bench_plain_content(c: &mut Criterion) {
    let input = "The quick brown fox jumps over the lazy dog.";
    c.bench_function("plain_content_pass_through", |b| {
        b.iter(|| {
            let mut p = StreamingParser::new(DialectRegistry::create("hermes").unwrap());
            let _ = black_box(p.feed(black_box(input)));
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_hermes_single, bench_llama3_json_single, bench_plain_content
}
criterion_main!(benches);
