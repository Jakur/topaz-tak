use criterion::{black_box, criterion_group, criterion_main, Criterion};
use topaz_tak::{execute_moves_check_valid, perft, Board6};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("small perft", |b| {
        b.iter(|| execute_small_perft(black_box(3)))
    });
}

fn execute_small_perft(depth: usize) {
    let ptn_moves = &[
        "c2", "c3", "d3", "b3", "c4", "1c2+", "1d3<", "1b3>", "1c4-", "Cc2", "a1", "1c2+", "a2",
    ];
    let mut board = Board6::new();
    let res = execute_moves_check_valid(&mut board, ptn_moves);
    assert!(res.is_ok());

    let p_res: Vec<_> = (0..depth)
        .map(|depth| perft(&mut board, depth as u16))
        .collect();
    // assert_eq!(&p_res[..], &[1, 190, 20698]);
    assert_eq!(&p_res[..], &[1, 190, 20698]);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
