use super::{Color, GameResult};
use crate::eval::Evaluate;
use crate::eval::{LOSE_SCORE, WIN_SCORE};
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

pub fn naive_minimax<E: Evaluate>(board: &mut E, depth: u16) -> i32 {
    match board.game_result() {
        Some(GameResult::WhiteWin) => {
            if let Color::White = board.side_to_move() {
                return WIN_SCORE;
            }
            return LOSE_SCORE;
        }
        Some(GameResult::BlackWin) => {
            if let Color::White = board.side_to_move() {
                return LOSE_SCORE;
            }
            return WIN_SCORE;
        }
        Some(GameResult::Draw) => return 0,
        None => (),
    }
    if depth == 0 {
        board.evaluate()
    } else {
        let side_to_move = board.side_to_move();
        let mut moves = vec![];
        board.generate_moves(&mut moves);
        let child_evaluations = moves.into_iter().map(|mv| {
            let reverse_move = board.do_move(mv);
            let eval = -1 * naive_minimax(board, depth - 1);
            board.reverse_move(reverse_move);
            eval
        });
        child_evaluations
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::Board6;
    // #[test]
    // fn small_minimax() {
    //     let tps = "2,1,1,1,1,2S/1,12,1,x,1C,11112/x,2,2,212,2C,11121/2,21122,x2,1,x/x3,1,1,x/x2,2,21,x,112S 1 34";
    //     let mut board = Board6::try_from_tps(tps).unwrap();
    //     let score = naive_minimax(&mut board, 3);
    //     assert_eq!(score, LOSE_SCORE);
    // }
}
