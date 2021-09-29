use super::{Color, GameResult};
use crate::eval::Evaluate;
use crate::eval::{LOSE_SCORE, WIN_SCORE};
use crate::move_gen::GameMove;
use std::collections::HashMap;

const NULL_REDUCTION: usize = 2;

pub struct SearchInfo {
    pub max_depth: usize,
    nodes: usize,
    pv_table: HashMap<u64, GameMove>, // Todo replace with LRU cache
    killer_moves: Vec<KillerMoves>,
    fail_high_first: usize,
    fail_high: usize,
}

impl SearchInfo {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            nodes: 0,
            pv_table: HashMap::new(),
            killer_moves: vec![KillerMoves::new(); max_depth + 1],
            fail_high_first: 0,
            fail_high: 0,
        }
    }
    fn store_pv_move<E: Evaluate>(&mut self, position: &E, mv: GameMove) {
        self.pv_table.insert(position.hash(), mv);
    }
    pub fn pv_move<E: Evaluate>(&self, position: &E) -> Option<GameMove> {
        self.pv_table.get(&position.hash()).copied()
    }
    fn full_pv<E: Evaluate>(&self, position: &mut E) -> Vec<GameMove> {
        let mut forward = Vec::new();
        let mut backward = Vec::new();
        while let Some(m) = self.pv_move(position) {
            dbg!(m.to_ptn());
            if !position.legal_move(m) {
                dbg!("Illegal Move in Pv");
                break;
            }
            let rev = position.do_move(m);
            backward.push(rev);
            forward.push(m);
        }
        for rev_m in backward.into_iter().rev() {
            position.reverse_move(rev_m);
        }
        forward
    }
}

#[derive(Clone)]
struct KillerMoves {
    killer1: GameMove,
    killer2: GameMove,
}

impl KillerMoves {
    pub fn new() -> Self {
        KillerMoves {
            killer1: GameMove::null_move(),
            killer2: GameMove::null_move(),
        }
    }
    pub fn add(&mut self, game_move: GameMove) {
        self.killer2 = self.killer1;
        self.killer1 = game_move;
    }
    pub fn score(&self, game_move: GameMove) -> i32 {
        if self.killer1 == game_move {
            90
        } else if self.killer2 == game_move {
            80
        } else {
            0
        }
    }
}

pub fn search<E: Evaluate>(board: &mut E, info: &mut SearchInfo) {
    let mut best_score = i32::MIN;

    let mut pv_moves = Vec::new();
    // let pv_numv = 0;

    for depth in 1..=info.max_depth {
        best_score = alpha_beta(-1_000_000, 1_000_000, depth, board, info, true);
        pv_moves = info.full_pv(board);
        print!(
            "Depth: {} Score: {} Nodes: {} PV: ",
            depth, best_score, info.nodes
        );
        for ptn in pv_moves.into_iter().map(|m| m.to_ptn()) {
            print!("{} ", ptn);
        }
        println!("");
    }
}

fn alpha_beta<E>(
    mut alpha: i32,
    beta: i32,
    depth: usize,
    board: &mut E,
    info: &mut SearchInfo,
    null_move: bool,
) -> i32
where
    E: Evaluate,
{
    info.nodes += 1;
    match board.game_result() {
        Some(GameResult::WhiteWin) => {
            if let Color::White = board.side_to_move() {
                return WIN_SCORE - depth as i32;
            }
            return LOSE_SCORE + depth as i32;
        }
        Some(GameResult::BlackWin) => {
            if let Color::White = board.side_to_move() {
                return LOSE_SCORE - depth as i32;
            }
            return WIN_SCORE + depth as i32;
        }
        Some(GameResult::Draw) => return 0,
        None => {}
    }
    if depth == 0 {
        return board.evaluate();
    }
    if null_move && depth >= 1 + NULL_REDUCTION {
        // Check if our position is so good that passing still gives opp a bad pos
        board.null_move();
        let score = -1
            * alpha_beta(
                -beta,
                -beta + 1,
                depth - 1 - NULL_REDUCTION,
                board,
                info,
                false,
            );
        board.rev_null_move();
        // If we beta cutoff from the null move, then we can stop searching
        if score >= beta {
            return beta;
        }
    }
    let mut moves = Vec::new();
    board.generate_moves(&mut moves);
    let mut best_move = None;
    if let Some(pv_move) = info.pv_move(board) {
        if let Some(found) = moves.iter_mut().find(|m| **m == pv_move) {
            found.add_score(100);
        }
    }
    for count in 0..moves.len() {
        let (idx, m) = moves
            .iter()
            .enumerate()
            .max_by_key(|(_i, &m)| m.score() + info.killer_moves[depth].score(m))
            .unwrap();
        let m = *m;
        moves.swap_remove(idx);
        let rev_move = board.do_move(m);
        let score = -1 * alpha_beta(-beta, -alpha, depth - 1, board, info, true);
        board.reverse_move(rev_move);
        if score > alpha {
            if score >= beta {
                if count == 1 {
                    info.fail_high_first += 1;
                }
                info.fail_high += 1;
                info.killer_moves[depth].add(m);
                return beta;
            }
            alpha = score;
            best_move = Some(m);
        }
    }

    if let Some(best) = best_move {
        info.store_pv_move(board, best);
    }
    alpha
}

pub fn root_minimax<E>(board: &mut E, depth: u16) -> (Option<GameMove>, i32)
where
    E: Evaluate,
{
    let mut moves = vec![];
    board.generate_moves(&mut moves);
    let mut best_eval = i32::MIN;
    let mut best_move = None;
    let child_evaluations = moves.into_iter().map(|mv| {
        let reverse_move = board.do_move(mv);
        let eval = -1 * naive_minimax(board, depth - 1);
        board.reverse_move(reverse_move);
        (mv, eval)
    });
    for (mv, eval) in child_evaluations {
        if eval > best_eval {
            best_eval = eval;
            best_move = Some(mv);
        }
    }
    (best_move, best_eval)
}

fn naive_minimax<E: Evaluate>(board: &mut E, depth: u16) -> i32 {
    match board.game_result() {
        Some(GameResult::WhiteWin) => {
            // dbg!("Found a winning position for white");
            if let Color::White = board.side_to_move() {
                return WIN_SCORE;
            }
            return LOSE_SCORE;
        }
        Some(GameResult::BlackWin) => {
            // dbg!("Found a winning position for black");
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
        let mut moves = vec![];
        board.generate_moves(&mut moves);
        let child_evaluations = moves.into_iter().map(|mv| {
            let reverse_move = board.do_move(mv);
            let eval = -1 * naive_minimax(board, depth - 1);
            board.reverse_move(reverse_move);
            eval
        });
        child_evaluations.max().unwrap()
    }
}
#[cfg(test)]
mod test {
    use super::*;
    use crate::eval::LOSE_SCORE;
    use crate::Board6;
    #[test]
    fn small_minimax() {
        let tps = "2,1,1,1,1,2S/1,12,1,x,1C,11112/x,2,2,212,2C,11121/2,21122,x2,1,x/x3,1,1,x/x2,2,21,x,112S 1 34";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let (mv, score) = root_minimax(&mut board, 2);
        assert!(score != LOSE_SCORE);
        let only_move = GameMove::try_from_ptn("c5-", &board);
        assert_eq!(mv, only_move);
    }

    #[test]
    fn small_alpha_beta() {
        let tps = "2,1,1,1,1,2S/1,12,1,x,1C,11112/x,2,2,212,2C,11121/2,21122,x2,1,x/x3,1,1,x/x2,2,21,x,112S 1 34";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let mut info = SearchInfo::new(4);
        search(&mut board, &mut info);
    }
    // #[test]
    // fn unk_puzzle() {
    //     let tps = "x2,1,21,2,2/1,2,21,1,21,2/1S,2,2,2C,2,2/21S,1,121C,x,1,12/2,2,121,1,1,1/2,2,x3,22S 1 27";
    //     let mut board = Board6::try_from_tps(tps).unwrap();
    //     let mut info = SearchInfo::new(6);
    //     search(&mut board, &mut info);
    // }
}
