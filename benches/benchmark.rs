use criterion::{black_box, criterion_group, criterion_main, Criterion};
use topaz_tak::board::find_placement_road;
use topaz_tak::eval::{Evaluator, Evaluator6, LOSE_SCORE};
use topaz_tak::search::root_minimax;
use topaz_tak::{execute_moves_check_valid, perft, Bitboard6, Board6, Color, GameMove};

pub fn criterion_benchmark(c: &mut Criterion) {
    // c.bench_function("small perft", |b| {
    //     b.iter(|| execute_small_perft(black_box(3)))
    // });
    let pos = get_positions();
    let eval = Evaluator6 {};
    c.bench_function("eval", |b| {
        b.iter(|| evaluate_positions(black_box(&pos), black_box(&eval)))
    });
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

fn get_positions() -> Vec<Board6> {
    ["x4,2,1/2,x2,21S,12,1/2,1,1,2,1,2/1221C,1,2,112C,1,1/2,1,2,1,2,1/2,x,2,2,1,x 1 22",
    "x,1,x4/2S,1,x,2,x2/2,121C,212S,x,2C,x/1,211121S,1S,112,11112,x/2221,x,2,1,2,1S/2222221,x,2,2,2,2 2 43",
    "x3,2,1,1/x3,21,1,21/x,2,121S,2,2,11/1,1,1,2,2,21/x2,2221C,12C,2,2/2,2,21,x,2,x 2 25",
    "2,x4,1/x,2,x,2,2S,1/x,1,2,1112C,1,1/1,2,x,2221C,x,1/2,1,1,x,12,2/x4,1,x 2 18",
    "1,x,2,112S,1,1/2,1,21,x,1,1/2,2,2,1,1,12S/2,x,2111122C,2,12,1/2,1,22221S,1,2,211112S/21,2221C,1,1,2,x 1 42",
    "2,2,221S,121S,x,12/x,2S,2,x,12S,12S/1,2,112C,22121C,1,1/x2,2,1,1,x/1,1,12,x,1,x/x2,2,x,1,x 1 28"]
        .iter()
        .map(|s| Board6::try_from_tps(s).unwrap())
        .collect()
}

fn evaluate_positions<E: Evaluator<Game = Board6>>(positions: &[Board6], eval: &E) -> i32 {
    let mut sum = 0;
    for pos in positions.iter() {
        sum += eval.evaluate(pos, 1);
    }
    return sum;
}

fn small_minimax(depth: u16) {
    let tps = "2,1,1,1,1,2S/1,12,1,x,1C,11112/x,2,2,212,2C,11121/2,21122,x2,1,x/x3,1,1,x/x2,2,21,x,112S 1 34";
    let mut board = Board6::try_from_tps(tps).unwrap();
    let eval = Evaluator6 {};
    let (mv, score) = root_minimax(&mut board, &eval, 2);
    assert!(score != LOSE_SCORE);
    let only_move = GameMove::try_from_ptn("c5-", &board);
    assert_eq!(mv, only_move);
}

fn placement_road(_x: ()) {
    // "2,x,2,2,2C,1/1,2212,1,1,x,121/12,222,1S,2S,2,x/2,x,2,x2,1/1,221,22,121,121,11C/x,2,2,2,x,1 2 32";
    let rp = Bitboard6::new(18144415765381120);
    let empty = Bitboard6::new(9570373622301696);
    let color = Color::White;
    assert_eq!(
        find_placement_road(color, rp, empty).map(|s| s.to_ptn()),
        Some("f4".to_string())
    );
    // "x,1,x4/2,2,1,1,1,1/2221,x,1,21C,x2/2,2,2C,1,2,x/2,2,1,1,1,2/2,x2,2,x,1 2 18"
    let rp = Bitboard6::new(5143712963624960);
    let empty = Bitboard6::new(12385175530928640);
    let color = Color::Black;
    assert_eq!(find_placement_road(color, rp, empty), None);
    // "1,1,1,1,1112C,1/x,121C,x,2S,21,1/1,2,x,12221S,1S,x/x,2,2,1,x,2/x3,2S,x2/2,2,x,122121,x2 1 29"
    let rp = Bitboard6::new(4503668386979328);
    let empty = Bitboard6::new(29394491094466560);
    let color = Color::White;
    assert_eq!(find_placement_road(color, rp, empty), None);
    // "2,x,1,x,2S,112C/2,x,1,121,x2/x2,1S,x,22221,x/x2,1,x,112222221C,x/x,112S,12,221S,21,21/1,1,x,2,x,1 1 44"
    let rp = Bitboard6::new(19808973823150080);
    let empty = Bitboard6::new(11261568908268544);
    let color = Color::White;
    assert!(find_placement_road(color, rp, empty).is_some());
    // "2,x4,1/x4,1,x/x,2,12C,1,1,x/x,1,2,21C,x2/x,2,2,x3/x2,2,1,x2 1 10"
    let rp = Bitboard6::new(4503686334136320);
    let empty = Bitboard6::new(28836213970320384);
    let color = Color::White;

    assert_eq!(find_placement_road(color, rp, empty), None);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
