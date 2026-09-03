use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use infe_parsers::{DialectRegistry, StreamingParser};

/// A mock token stream generator inspired by inference-lab's `serve::engine`
/// `TokenEvent` pipeline. Simulates a real decode loop that emits text chunks
/// across token boundaries, including tool-call markers, JSON arguments, and
/// reasoning blocks.
struct MockTokenStream {
    chunks: Vec<&'static str>,
    cursor: usize,
}

impl MockTokenStream {
    /// Hermes tool-call stream split across token boundaries, mirroring how
    /// a real tokenizer chunks the XML-like markers and JSON body.
    fn hermes_tool_call() -> Self {
        Self {
            chunks: vec![
                "<tool",
                "_call>",
                "{\"name\"",
                ":\"get_weather\"",
                ",\"arguments\"",
                ":{\"city\"",
                ":\"London\"",
                "}}",
                "</tool",
                "_call>",
            ],
            cursor: 0,
        }
    }

    /// Llama-3 JSON tool call, streamed token by token.
    fn llama3_json_tool_call() -> Self {
        Self {
            chunks: vec![
                "{\"name\"",
                ":\"search\"",
                ",\"parameters\"",
                ":{\"q\"",
                ":\"rust streaming\"",
                ",\"limit\"",
                ":5}}",
            ],
            cursor: 0,
        }
    }

    /// `DeepSeek` reasoning block with content after the closing tag.
    fn deepseek_reasoning() -> Self {
        Self {
            chunks: vec![
                "<think>", "Let", " me", " analyze", " this", " step", " by", " step", ".",
                "</think>", "The", " answer", " is", " 42", ".",
            ],
            cursor: 0,
        }
    }

    /// Plain content with no tool calls or reasoning blocks.
    fn plain_content() -> Self {
        Self {
            chunks: vec![
                "The",
                " quick",
                " brown",
                " fox",
                " jumps",
                " over",
                " the",
                " lazy",
                " dog",
                ".",
                " This",
                " is",
                " a",
                " test",
                " of",
                " the",
                " streaming",
                " parser",
                ".",
            ],
            cursor: 0,
        }
    }

    fn next(&mut self) -> Option<&'static str> {
        if self.cursor >= self.chunks.len() {
            return None;
        }
        let chunk = self.chunks[self.cursor];
        self.cursor += 1;
        Some(chunk)
    }

    fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

fn bench_hermes_single_chunk(c: &mut Criterion) {
    let mut group = c.benchmark_group("hermes");
    group.throughput(Throughput::Elements(1));

    group.bench_function("single_tool_call", |b| {
        b.iter(|| {
            let dialect = DialectRegistry::create("hermes").unwrap();
            let mut parser = StreamingParser::new(dialect);
            let mut stream = MockTokenStream::hermes_tool_call();
            while let Some(chunk) = stream.next() {
                parser.feed(black_box(chunk));
            }
        });
    });

    group.finish();
}

fn bench_hermes_plain_content(c: &mut Criterion) {
    let mut group = c.benchmark_group("hermes_plain");
    let stream = MockTokenStream::plain_content();
    group.throughput(Throughput::Elements(stream.chunk_count() as u64));

    group.bench_function("no_tool_calls", |b| {
        b.iter(|| {
            let dialect = DialectRegistry::create("hermes").unwrap();
            let mut parser = StreamingParser::new(dialect);
            let mut stream = MockTokenStream::plain_content();
            while let Some(chunk) = stream.next() {
                parser.feed(black_box(chunk));
            }
        });
    });

    group.finish();
}

fn bench_llama3_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("llama3_json");
    group.throughput(Throughput::Elements(1));

    group.bench_function("single_tool_call", |b| {
        b.iter(|| {
            let dialect = DialectRegistry::create("llama3_json").unwrap();
            let mut parser = StreamingParser::new(dialect);
            let mut stream = MockTokenStream::llama3_json_tool_call();
            while let Some(chunk) = stream.next() {
                parser.feed(black_box(chunk));
            }
        });
    });

    group.finish();
}

fn bench_deepseek_reasoning(c: &mut Criterion) {
    let mut group = c.benchmark_group("deepseek_reasoning");
    let stream = MockTokenStream::deepseek_reasoning();
    group.throughput(Throughput::Elements(stream.chunk_count() as u64));

    group.bench_function("reasoning_block", |b| {
        b.iter(|| {
            let dialect = DialectRegistry::create("deepseek_reasoning").unwrap();
            let mut parser = StreamingParser::new(dialect);
            let mut stream = MockTokenStream::deepseek_reasoning();
            while let Some(chunk) = stream.next() {
                parser.feed(black_box(chunk));
            }
        });
    });

    group.finish();
}

/// Concurrent-stream throughput: simulates N concurrent requests being
/// parsed in a single step, as a real engine would batch them. This is
/// the metric that maps to ITL p99 in production (BRIEF §7).
fn bench_concurrent_streams(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent");

    for n in [64u64, 256, 1024] {
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(
            criterion::BenchmarkId::new("hermes_streams", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    let mut parsers: Vec<StreamingParser> = (0..n)
                        .map(|_| StreamingParser::new(DialectRegistry::create("hermes").unwrap()))
                        .collect();
                    // One step: feed one chunk to each parser
                    for parser in &mut parsers {
                        parser.feed(black_box("{\"name\":\"fn\",\"arguments\":{}}"));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hermes_single_chunk,
    bench_hermes_plain_content,
    bench_llama3_json,
    bench_deepseek_reasoning,
    bench_concurrent_streams,
);
criterion_main!(benches);
