use super::{Color, GameResult};
use crate::board::TakBoard;
use crate::eval::Evaluator;
use crate::eval::{LOSE_SCORE, WIN_SCORE};
use crate::move_gen::{generate_all_moves, GameMove, KillerMoves, SmartMoveBuffer};
use crate::TeiCommand;
use crossbeam_channel::Receiver;
use lru::LruCache;
use std::marker::PhantomData;
use std::time::Instant;

pub mod proof;

const NULL_REDUCTION: usize = 2;

pub struct SearchInfo {
    pub max_depth: usize,
    pub nodes: usize,
    pv_table: LruCache<u64, HashEntry>, // Todo replace with LRU cache
    pub killer_moves: Vec<KillerMoves>,
    fail_high_first: usize,
    fail_high: usize,
    stopped: bool,
    input: Option<Receiver<TeiCommand>>,
    max_time: u64,
    start_time: Instant,
    start_ply: usize,
    transposition_cutoffs: u64,
}

impl SearchInfo {
    pub fn new(max_depth: usize, pv_size: usize) -> Self {
        Self {
            max_depth,
            pv_table: LruCache::new(pv_size),
            killer_moves: vec![KillerMoves::new(); max_depth + 1],
            fail_high_first: 0,
            fail_high: 0,
            transposition_cutoffs: 0,
            nodes: 0,
            stopped: false,
            input: None,
            max_time: 120, // Some large but not enormous default
            start_time: Instant::now(),
            start_ply: 0,
        }
    }
    pub fn print_cuts(&self) {
        println!(
            "Fail High: {} Fail High First: {} Transposition Hits: {} ",
            self.fail_high, self.fail_high_first, self.transposition_cutoffs
        );
    }
    pub fn take_table(mut self, other: &mut Self) -> Self {
        std::mem::swap(&mut self.pv_table, &mut other.pv_table);
        self
    }
    pub fn set_start_ply(&mut self, start_ply: usize) {
        self.start_ply = start_ply;
    }
    pub fn input_stream(mut self, r: Receiver<TeiCommand>) -> Self {
        self.input = Some(r);
        self
    }
    pub fn max_time(mut self, time: u64) -> Self {
        self.max_time = time;
        self
    }
    pub fn start_search(&mut self) {
        self.stopped = false;
        self.nodes = 0;
        self.start_time = Instant::now();
    }
    pub fn take_input_stream(&mut self) -> Option<Receiver<TeiCommand>> {
        self.input.take()
    }
    pub fn check_stop(&mut self) {
        let secs = self.start_time.elapsed().as_secs();
        if secs >= self.max_time {
            self.stopped = true;
        }
        if let Some(ref r) = self.input {
            if r.try_recv().is_ok() {
                self.stopped = true;
            }
        }
    }
    fn store_move<E: TakBoard>(&mut self, position: &E, entry: HashEntry) {
        self.pv_table.put(position.hash(), entry);
    }
    fn lookup_move<E: TakBoard>(&mut self, position: &E) -> Option<&HashEntry> {
        self.pv_table.get(&position.hash())
    }
    pub fn pv_move<E: TakBoard>(&mut self, position: &E) -> Option<GameMove> {
        self.pv_table.get(&position.hash()).map(|e| e.game_move)
    }
    fn full_pv<E: TakBoard>(&mut self, position: &mut E) -> Vec<GameMove> {
        let mut forward = Vec::new();
        let mut backward = Vec::new();
        while let Some(m) = self.pv_move(position) {
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
    fn ply_depth<E: TakBoard>(&self, position: &E) -> usize {
        position.ply() - self.start_ply
    }
}

enum ScoreCutoff {
    Alpha(i32),
    Beta(i32),
    Exact(i32),
}

struct HashEntry {
    game_move: GameMove,
    score: ScoreCutoff,
    depth: usize,
}

impl HashEntry {
    fn new(game_move: GameMove, score: ScoreCutoff, depth: usize) -> Self {
        Self {
            game_move,
            score,
            depth,
        }
    }
}

pub struct SearchOutcome<T> {
    score: i32,
    time: u128,
    pv: Vec<GameMove>,
    nodes: usize,
    depth: usize,
    t_cuts: u64,
    phantom: PhantomData<T>,
}

impl<T> SearchOutcome<T>
where
    T: TakBoard,
{
    pub fn new(score: i32, pv: Vec<GameMove>, depth: usize, search_info: &SearchInfo) -> Self {
        let nodes = search_info.nodes;
        let time = search_info.start_time.elapsed().as_millis();
        // let t_cuts = search_info.transposition_cutoffs;
        let t_cuts = search_info.transposition_cutoffs;
        Self {
            score,
            pv,
            nodes,
            time,
            depth,
            t_cuts,
            phantom: PhantomData,
        }
    }
    pub fn best_move(&self) -> Option<String> {
        self.pv.get(0).map(|m| m.to_ptn::<T>())
    }
}

impl<T> std::fmt::Display for SearchOutcome<T>
where
    T: TakBoard,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let mut pv_string = String::new();
        for m in self.pv.iter() {
            pv_string.push_str(&m.to_ptn::<T>());
            pv_string.push_str(" ");
        }
        pv_string.pop();
        let nps = if self.time > 0 {
            (self.nodes as u128) / self.time * 1000
        } else {
            0
        };
        write!(
            f,
            "score cp {} time {} pv {} nodes {} nps {} depth {} tcut {}",
            self.score, self.time, pv_string, self.nodes, nps, self.depth, self.t_cuts
        )
    }
}

pub fn search<T, E>(board: &mut T, eval: &E, info: &mut SearchInfo) -> Option<SearchOutcome<T>>
where
    T: TakBoard,
    E: Evaluator<Game = T>,
{
    let mut outcome = None;
    let mut node_counts = vec![1];
    info.set_start_ply(board.ply());
    for depth in 1..=info.max_depth {
        // Abort if we are unlikely to finish the search at this depth
        if depth >= 6 {
            let mut est_branch = node_counts[depth - 2] as f64 / node_counts[depth - 3] as f64;
            if est_branch < 3.0 || est_branch > 100.0 {
                // Transposition hits causing instability, just guess
                if depth % 2 == 0 {
                    // Even nodes are cheaper, see even-odd effect
                    est_branch = 5.0;
                } else {
                    est_branch = 10.0;
                }
            }
            let est_nodes = node_counts[depth - 1] as f64 * est_branch;
            let elapsed = info.start_time.elapsed().as_secs_f64();
            let nps = node_counts.iter().copied().sum::<usize>() as f64 / elapsed;
            let remaining = info.max_time as f64 - elapsed;
            if est_nodes / nps > remaining {
                break;
            }
        }
        let best_score = alpha_beta(-1_000_000, 1_000_000, depth, board, eval, info, true);
        node_counts.push(info.nodes);
        let pv_moves = info.full_pv(board);
        // If we had an incomplete depth search, use the previous depth's vals
        if info.stopped {
            break;
        }
        print!(
            "Depth: {} Score: {} Nodes: {} PV: ",
            depth, best_score, info.nodes
        );
        outcome = Some(SearchOutcome::new(
            best_score,
            pv_moves.clone(),
            depth,
            info,
        ));
        for ptn in pv_moves.iter().map(|m| m.to_ptn::<T>()) {
            print!("{} ", ptn);
        }
        println!("");
        // Stop wasting time
        if best_score > WIN_SCORE - 10 || best_score < LOSE_SCORE + 10 {
            return Some(SearchOutcome::new(best_score, pv_moves, depth, info));
        }
    }
    outcome
}

fn alpha_beta<T, E>(
    mut alpha: i32,
    beta: i32,
    depth: usize,
    board: &mut T,
    evaluator: &E,
    info: &mut SearchInfo,
    null_move: bool,
) -> i32
where
    T: TakBoard,
    E: Evaluator<Game = T>,
{
    info.nodes += 1;
    const FREQ: usize = (1 << 16) - 1; // Per 65k nodes
    if (info.nodes & FREQ) == FREQ {
        info.check_stop();
    }
    match board.game_result() {
        Some(GameResult::WhiteWin) => {
            if let Color::White = board.side_to_move() {
                return WIN_SCORE - board.ply() as i32 + info.start_ply as i32;
            }
            return LOSE_SCORE + board.ply() as i32 - info.start_ply as i32;
        }
        Some(GameResult::BlackWin) => {
            if let Color::White = board.side_to_move() {
                return LOSE_SCORE + board.ply() as i32 - info.start_ply as i32;
            }
            return WIN_SCORE - board.ply() as i32 + info.start_ply as i32;
        }
        Some(GameResult::Draw) => return 0,
        None => {}
    }
    // let mut road_move = None;
    if depth == 0 {
        let ply_depth = info.ply_depth(board);
        return evaluator.evaluate(board, ply_depth);
        // let mut road_check = Vec::new();
        // road_move = board.can_make_road(&mut road_check);
        // if road_move.is_some() {
        //     depth += 1;
        // } else {
        //     return board.evaluate();
        // }
        // road_check.clear();
    }

    if let Some(data) = info.lookup_move(board) {
        if data.depth >= depth {
            match data.score {
                ScoreCutoff::Alpha(score) => {
                    if score <= alpha {
                        info.transposition_cutoffs += 1;
                        return alpha;
                    }
                }
                ScoreCutoff::Beta(score) => {
                    if score >= beta {
                        info.transposition_cutoffs += 1;
                        return beta;
                    }
                }
                ScoreCutoff::Exact(score) => {
                    info.transposition_cutoffs += 1;
                    return score;
                }
            }
        }
    }

    if null_move && depth >= 1 + NULL_REDUCTION {
        // && road_move.is_none() {
        // Check if our position is so good that passing still gives opp a bad pos
        board.null_move();
        let score = -1
            * alpha_beta(
                -beta,
                -beta + 1,
                depth - 1 - NULL_REDUCTION,
                board,
                evaluator,
                info,
                false,
            );
        board.rev_null_move();
        // If we beta cutoff from the null move, then we can stop searching
        if score >= beta {
            return beta;
        }
    }
    let mut moves = SmartMoveBuffer::new();
    generate_all_moves(board, &mut moves);
    let mut best_move = None;
    let mut best_score = None;
    if let Some(pv_move) = info.pv_move(board) {
        moves.score_pv_move(pv_move);
    }
    moves.score_stack_moves(board);
    // Do a slower, more thorough move ordering at the root
    // if info.ply_depth(board) == 0 {
    //     // DEBUG
    //     // moves.sort_by(|x, y| x.score().cmp(&y.score()));
    //     // println!("MOVE SEARCH ORDER: ");
    //     // for m in moves.iter() {
    //     //     println!("{}", m.to_ptn::<crate::Board6>());
    //     // }
    //     let tak_threats = board.get_tak_threats(&moves, None);
    //     if tak_threats.len() > 0 {
    //         for m in moves.iter_mut() {
    //             if tak_threats.contains(m) {
    //                 m.add_score(50);
    //             }
    //         }
    //     }
    // }
    let old_alpha = alpha;
    for count in 0..moves.len() {
        let m = moves.get_best(depth, info);
        let rev_move = board.do_move(m);
        let score = -1 * alpha_beta(-beta, -alpha, depth - 1, board, evaluator, info, true);
        board.reverse_move(rev_move);
        if info.stopped {
            return 0;
        }
        if score > alpha {
            if score >= beta {
                if count == 1 {
                    info.fail_high_first += 1;
                }
                info.fail_high += 1;
                if m.is_place_move() {
                    info.killer_moves[depth].add(m);
                }
                info.store_move(board, HashEntry::new(m, ScoreCutoff::Beta(beta), depth));
                return beta;
            }
            alpha = score;
            best_move = Some(m);
            best_score = Some(score);
        }
    }

    if let Some(best) = best_move {
        if let Some(best_score) = best_score {
            if alpha != old_alpha {
                info.store_move(
                    board,
                    HashEntry::new(best, ScoreCutoff::Exact(best_score), depth),
                );
            } else {
                info.store_move(
                    board,
                    HashEntry::new(best, ScoreCutoff::Alpha(alpha), depth),
                )
            }
        }
    }
    alpha
}

/// A naive minimax function without pruning used for debugging and benchmarking
pub fn root_minimax<T, E>(board: &mut T, eval: &E, depth: u16) -> (Option<GameMove>, i32)
where
    T: TakBoard,
    E: Evaluator<Game = T>,
{
    let mut moves = vec![];
    board.generate_moves(&mut moves);
    let mut best_eval = i32::MIN;
    let mut best_move = None;
    let child_evaluations = moves.into_iter().map(|mv| {
        let reverse_move = board.do_move(mv);
        let eval = -1 * naive_minimax(board, eval, depth - 1);
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

fn naive_minimax<T: TakBoard, E: Evaluator<Game = T>>(board: &mut T, eval: &E, depth: u16) -> i32 {
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
        eval.evaluate(board, 0)
    } else {
        let mut moves = vec![];
        board.generate_moves(&mut moves);
        let child_evaluations = moves.into_iter().map(|mv| {
            let reverse_move = board.do_move(mv);
            let eval = -1 * naive_minimax(board, eval, depth - 1);
            board.reverse_move(reverse_move);
            eval
        });
        child_evaluations.max().unwrap()
    }
}
#[cfg(test)]
mod test {
    use super::*;
    use crate::board::Board6;
    use crate::eval::{Evaluator6, LOSE_SCORE};
    #[test]
    fn small_minimax() {
        let tps = "2,1,1,1,1,2S/1,12,1,x,1C,11112/x,2,2,212,2C,11121/2,21122,x2,1,x/x3,1,1,x/x2,2,21,x,112S 1 34";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let eval = Evaluator6 {};
        let (mv, score) = root_minimax(&mut board, &eval, 2);
        assert!(score != LOSE_SCORE);
        let only_move = GameMove::try_from_ptn("c5-", &board);
        assert_eq!(mv, only_move);
    }

    #[test]
    fn small_alpha_beta() {
        let tps = "2,1,1,1,1,2S/1,12,1,x,1C,11112/x,2,2,212,2C,11121/2,21122,x2,1,x/x3,1,1,x/x2,2,21,x,112S 1 34";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let mut info = SearchInfo::new(4, 50000);
        let eval = Evaluator6 {};
        search(&mut board, &eval, &mut info);
    }
    #[test]
    fn unk_puzzle() {
        let tps = "x2,1,21,2,2/1,2,21,1,21,2/1S,2,2,2C,2,2/21S,1,121C,x,1,12/2,2,121,1,1,1/2,2,x3,22S 1 27";
        let mut board = Board6::try_from_tps(tps).unwrap();
        dbg!(board.ply());
        let eval = Evaluator6 {};
        let mut info = SearchInfo::new(5, 100000);
        search(&mut board, &eval, &mut info);
    }
}
