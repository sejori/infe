#![allow(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use infe_core::timer::StepTimer;
use std::hint::black_box as bb;

fn bench_step_timer_disabled(c: &mut Criterion) {
    let timer = StepTimer::new("bench");
    // Disabled — should be near-zero overhead.
    c.bench_function("step_timer/disabled", |b| {
        b.iter(|| {
            let _guard = timer.start(bb(0));
            black_box(());
        });
    });
}

fn bench_step_timer_enabled(c: &mut Criterion) {
    let timer = StepTimer::new("bench");
    timer.set_enabled(true);

    c.bench_function("step_timer/enabled", |b| {
        b.iter(|| {
            let _guard = timer.start(bb(0));
            black_box(());
        });
    });
}

fn bench_buffer_view_construct(c: &mut Criterion) {
    use infe_core::buffer::{BufferView, DType};

    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };

    c.bench_function("buffer_view/construct_1k_f32", |b| {
        b.iter(|| {
            let view = BufferView::from_bytes(bb(bytes), DType::F32);
            black_box(view);
        });
    });
}

fn bench_buffer_view_access(c: &mut Criterion) {
    use infe_core::buffer::{BufferView, DType};

    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    let view = BufferView::from_bytes(bytes, DType::F32);

    c.bench_function("buffer_view/access_f32_1k", |b| {
        b.iter(|| {
            let slice = bb(&view).as_f32().unwrap();
            let sum: f32 = bb(slice).iter().sum();
            black_box(sum);
        });
    });
}

criterion_group!(
    benches,
    bench_step_timer_disabled,
    bench_step_timer_enabled,
    bench_buffer_view_construct,
    bench_buffer_view_access,
);
criterion_main!(benches);
