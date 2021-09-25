use criterion::{black_box, criterion_group, criterion_main, Criterion};
use topaz_tak::eval::LOSE_SCORE;
use topaz_tak::search::root_minimax;
use topaz_tak::{execute_moves_check_valid, perft, Board6, GameMove};

pub fn criterion_benchmark(c: &mut Criterion) {
    // c.bench_function("small perft", |b| {
    //     b.iter(|| execute_small_perft(black_box(3)))
    // });
    c.bench_function("small minimax", |b| b.iter(|| small_minimax(black_box(3))));
}

fn execute_small_perft(depth: usize) {
    let ptn_moves = &[
        "c2", "c3", "d3", "b3", "c4", "1c2+", "1d3<", "1b3>", "1c4-", "Cc2", "a1", "1c2+", "a2",
    ];
    let mut board = Board6::new();
    let res = execute_moves_check_valid(&mut board, ptn_moves);
    assert!(res.is_ok());

    let p_res: Vec<_> = (0..depth + 1)
        .map(|depth| perft(&mut board, depth as u16))
        .collect();
    // assert_eq!(&p_res[..], &[1, 190, 20698]);
    assert_eq!(&p_res[..], &[1, 190, 20698]);
}

fn small_minimax(depth: u16) {
    let tps = "2,1,1,1,1,2S/1,12,1,x,1C,11112/x,2,2,212,2C,11121/2,21122,x2,1,x/x3,1,1,x/x2,2,21,x,112S 1 34";
    let mut board = Board6::try_from_tps(tps).unwrap();
    let (mv, score) = root_minimax(&mut board, 2);
    assert!(score != LOSE_SCORE);
    let only_move = GameMove::try_from_ptn("c5-", &board);
    assert_eq!(mv, only_move);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
