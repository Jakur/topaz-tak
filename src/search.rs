use crate::eval::Evaluate;
use crate::move_gen::GameMove;

pub struct SearchInfo {
    pub max_depth: usize,
}

pub fn search<E: Evaluate>(board: &mut E, info: SearchInfo) {
    let mut best_move = GameMove::null_move();
    let mut best_score = i32::MIN;

    let pv_moves = 0;
    let pv_numv = 0;

    for depth in 1..info.max_depth {
        best_score = alpha_beta(i32::MIN, i32::MAX, depth, board, &info, true);
    }
}

fn alpha_beta<E>(
    alpha: i32,
    beta: i32,
    depth: usize,
    board: &mut E,
    info: &SearchInfo,
    null_move: bool,
) -> i32
where
    E: Evaluate,
{
    todo!()
}
