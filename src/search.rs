use super::{Color, GameResult};
use crate::board::TakBoard;
use crate::eval::Evaluator;
use crate::eval::{LOSE_SCORE, WIN_SCORE};
use crate::move_gen::{
    generate_aggressive_place_moves, generate_all_stack_moves, GameMove, HistoryMoves, KillerMoves,
    MoveBuffer, RevGameMove, SmartMoveBuffer,
};
use crate::TeiCommand;
use crossbeam_channel::Receiver;
use lru::LruCache;
use std::marker::PhantomData;
use std::time::Instant;

pub mod proof;

const NULL_REDUCTION: usize = 2;

// late move reduction parameters
const LMR_FULL_SEARCH_MOVES: usize = 8; // number of moves that don't get reduced
const LMR_DEPTH_LIMIT: usize = 3;       // don't reduce low depth searches up to this depth

// transposition table parameters
const TT_DISCARD_AGE: u8 = 5; // adjusts which HashEntries are considered outdated


pub struct SearchInfo {
    pub max_depth: usize,
    pub nodes: usize,
    pv_table: HashTable,
    pub killer_moves: Vec<KillerMoves>,
    pub hist_moves: HistoryMoves,
    stopped: bool,
    input: Option<Receiver<TeiCommand>>,
    max_time: u64,
    start_time: Instant,
    start_ply: usize,
    estimate_time: bool,
    pub stats: SearchStats,
}

impl SearchInfo {
    pub fn new(max_depth: usize, pv_size: usize) -> Self {
        Self {
            max_depth,
            pv_table: HashTable::new(pv_size),
            killer_moves: vec![KillerMoves::new(); max_depth + 1],
            hist_moves: HistoryMoves::new(6), // Todo init in search
            nodes: 0,
            stopped: false,
            input: None,
            max_time: 120, // Some large but not enormous default
            start_time: Instant::now(),
            start_ply: 0,
            estimate_time: true,
            stats: SearchStats::new(16),
        }
    }
    pub fn print_cuts(&self) {
        println!(
            "Fail High: {} Fail High First: {} Transposition Hits: {} ",
            self.stats.fail_high, self.stats.fail_high_first, self.stats.transposition_cutoffs
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
                dbg!("Illegal Move in Pv {}", m.to_ptn::<E>());
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

#[derive(Debug)]
pub struct SearchStats {
    fail_high: u64,
    fail_high_first: u64,
    transposition_cutoffs: u64,
    ordering_cut: Vec<usize>,
    ordering_alpha: Vec<usize>,
    bad_search: u64,
}

impl SearchStats {
    pub fn new(ordering_size: usize) -> Self {
        Self {
            fail_high: 0,
            fail_high_first: 0,
            transposition_cutoffs: 0,
            ordering_cut: vec![0; ordering_size],
            ordering_alpha: vec![0; ordering_size],
            bad_search: 0,
        }
    }
    fn add_cut(&mut self, order: usize) {
        if order >= self.ordering_cut.len() {
            *self.ordering_cut.last_mut().unwrap() += 1;
        } else {
            self.ordering_cut[order] += 1;
        }
    }
    fn add_alpha(&mut self, order: usize) {
        if order >= self.ordering_alpha.len() {
            *self.ordering_alpha.last_mut().unwrap() += 1;
        } else {
            self.ordering_alpha[order] += 1;
        }
    }
}

#[derive(Clone, Copy)]
enum ScoreCutoff {
    Alpha(i32),
    Beta(i32),
    Exact(i32),
}

// drop-in replacement for LruCache<u64, hashEntry>
//
// for every slot, there are two entries consecutive in memory.
// first entry is the "depth table" entry, always replace on depth.
// second entry is the "recency table" entry, always replace if not in depth table.
// this way, depth table and recency table entries lie consecutively in memory and will
// hopefully be fetched in a single cache line.
//
// possible improvements:
// - Buckets:  instead of having 2 entries with different replacement schemes for every slot,
//             have 4 entries with some fancy common replacement scheme.
// - Bleeding: have put and get also "bleed over" to next slot if that is free,
//             so that a better table utilization can be reached.
struct HashTable {
    size: usize,
    two_way_table: Vec<HashEntry>,
}

impl HashTable {
    fn new(size: usize) -> Self {
        Self {
            size: size / 2,
            two_way_table: vec![HashEntry::new(1, GameMove::null_move(), ScoreCutoff::Alpha(0), 0, 0); size],
        }
    }

    fn put(&mut self, hash: u64, entry: HashEntry) {
        let slot: usize = (hash as usize % self.size) * 2;
        let depth_entry = self.two_way_table[slot];
        if entry.depth > depth_entry.depth
            || entry.depth + entry.ply - TT_DISCARD_AGE > depth_entry.depth + depth_entry.ply
            {
                self.two_way_table[slot] = entry;
            } else {
                self.two_way_table[slot+1] = entry;
            }
    }

    fn get(&self, hash: &u64) -> Option<&HashEntry> {
        let slot: usize = (*hash as usize % self.size) * 2;
        let hash_hi: u32 = (*hash >> 32) as u32;
        if self.two_way_table[slot].hash_hi == hash_hi {
            return Some(&self.two_way_table[slot]);
        }
        if self.two_way_table[slot+1].hash_hi == hash_hi {
            return Some(&self.two_way_table[slot+1]);
        }
        None
    }
}

// right now this totals to 22 bytes, extended to 24 bytes for 8 byte memory alignment.
// it would be nice to get this into 16 bytes!
// this would be not only mean less memory usage, but also be much faster because of
// cache utilization. each slot (2 entries) would be loaded with a single cache line.
#[derive(Clone, Copy)]
struct HashEntry {
    hash_hi: u32,        // 4 bytes (~3 low bytes are encoded in the HashTable idx)
    game_move: GameMove, // 8 bytes
    score: ScoreCutoff,  // 8 bytes (can this be 2 bytes?)
    depth: u8,           // 1 byte
    ply: u8,             // 1 byte
}

impl HashEntry {
    fn new(hash: u64, game_move: GameMove, score: ScoreCutoff, depth: usize, ply: usize) -> Self {
        Self {
            hash_hi: (hash >> 32) as u32,
            game_move,
            score,
            depth: depth as u8,
            ply: ply as u8,
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
        let t_cuts = search_info.stats.transposition_cutoffs;
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
        if info.estimate_time && depth >= 6 {
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
        let best_score = alpha_beta(
            board,
            eval,
            info,
            SearchData::new(-1_000_000, 1_000_000, depth, true, None, 0, TakHistory(0)),
        );
        node_counts.push(info.nodes);
        let pv_moves = info.full_pv(board);
        // If we had an incomplete depth search, use the previous depth's vals
        if info.stopped {
            print!(
                "Aborted Depth: {} Score: {} Nodes: {} PV: ",
                depth, best_score, info.nodes
            );
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

#[derive(Clone, Copy)]
struct TakHistory(u32);

impl TakHistory {
    fn add(self, depth: usize) -> Self {
        Self(self.0 | 1 << depth)
    }
    fn consecutive(self, depth: usize) -> bool {
        if depth < 2 {
            return false;
        }
        let mask = (1 << depth) | (1 << depth - 2);
        (self.0 & mask) == mask
    }
}

struct SearchData {
    alpha: i32,
    beta: i32,
    depth: usize,
    null_move: bool,
    last_move: Option<RevGameMove>,
    extensions: u8,
    tak_history: TakHistory,
}

impl SearchData {
    fn new(
        alpha: i32,
        beta: i32,
        depth: usize,
        null_move: bool,
        last_move: Option<RevGameMove>,
        extensions: u8,
        tak_history: TakHistory,
    ) -> Self {
        Self {
            alpha,
            beta,
            depth,
            null_move,
            last_move,
            extensions,
            tak_history,
        }
    }
}

fn alpha_beta<T, E>(board: &mut T, evaluator: &E, info: &mut SearchInfo, data: SearchData) -> i32
where
    T: TakBoard,
    E: Evaluator<Game = T>,
{
    let SearchData {
        mut alpha,
        beta,
        depth,
        null_move,
        last_move,
        extensions,
        mut tak_history,
    } = data;
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
        // let critical = board.bits().road_pieces(board.side_to_move());
        // if critical & board.bits().empty() != T::Bits::ZERO {
        //     return WIN_SCORE - board.ply() as i32 + info.start_ply as i32 - 1;
        // }

        // let opp_critical = board.bits().road_pieces(!board.side_to_move()) & board.bits().empty();
        // // Opponent has two winning placements, we must move a stack
        // if extensions == 0 && opp_critical.pop_count() >= 2 {
        //     return alpha_beta(
        //         board,
        //         evaluator,
        //         info,
        //         SearchData::new(alpha, beta, 2, false, last_move, 1, true),
        //     );
        // }
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

    if let Some(entry) = info.lookup_move(board) {
        if entry.depth as usize >= depth {
            match entry.score {
                ScoreCutoff::Alpha(score) => {
                    if score <= alpha {
                        info.stats.transposition_cutoffs += 1;
                        return alpha;
                    }
                }
                ScoreCutoff::Beta(score) => {
                    if score >= beta {
                        info.stats.transposition_cutoffs += 1;
                        return beta;
                    }
                }
                ScoreCutoff::Exact(score) => {
                    info.stats.transposition_cutoffs += 1;
                    return score;
                }
            }
        }
    }
    if null_move && depth >= 1 + NULL_REDUCTION {
        // && road_move.is_none() {

        board.null_move();
        // Check if we are in Tak
        let one_depth_score = -1
            * alpha_beta(
                board,
                evaluator,
                info,
                SearchData::new(
                    -1_000_000,
                    1_000_000,
                    1,
                    false,
                    None,
                    extensions,
                    tak_history,
                ),
            );
        if one_depth_score <= LOSE_SCORE + 100 {
            tak_history = tak_history.add(info.ply_depth(board));
            board.rev_null_move();
        } else {
            // Check if our position is so good that passing still gives opp a bad pos
            let score = -1
                * alpha_beta(
                    board,
                    evaluator,
                    info,
                    SearchData::new(
                        -beta,
                        -beta + 1,
                        depth - 1 - NULL_REDUCTION,
                        false,
                        None,
                        extensions,
                        tak_history,
                    ),
                );
            board.rev_null_move();
            // If we beta cutoff from the null move, then we can stop searching
            if score >= beta {
                return beta;
            }
        }
    }
    let mut moves = SmartMoveBuffer::new();
    // Do a slower, more thorough move ordering near the root
    if depth > 3 && board.ply() >= 6 {
        moves.gen_score_place_moves(board);
        let mut check_moves = Vec::new();
        if let Some(mv) = board.can_make_road(&mut check_moves, None) {
            let data = &[mv];
            moves.add_move(mv);
            moves.score_tak_threats(data);
        } else {
            check_moves.clear();
            generate_aggressive_place_moves(board, &mut check_moves);
            let tak_threats = board.get_tak_threats(&check_moves, None);
            moves.score_tak_threats(&tak_threats);
            if board.ply() >= 4 {
                generate_all_stack_moves(board, &mut moves);
                moves.score_stack_moves(board, last_move.filter(|x| x.game_move.is_stack_move()));
            }
        }
    } else {
        if board.ply() >= 4 {
            generate_all_stack_moves(board, &mut moves);
            moves.score_stack_moves(board, last_move.filter(|x| x.game_move.is_stack_move()));
        }
        moves.gen_score_place_moves(board);
    }
    // moves.gen_score_place_moves(board);
    // if !must_capture {
    //     // Note maybe it is possible to construct a situation with 0 legal moves
    //     // Though it should not arise from real gameplay
    //     moves.gen_score_place_moves(board);
    // }
    let mut best_move = None;
    let mut best_score = None;
    if let Some(pv_move) = info.pv_move(board) {
        moves.score_pv_move(pv_move);
    }

    let old_alpha = alpha;
    for count in 0..moves.len() {
        let m = moves.get_best(depth, info);
        // if depth == 6 {
        //     println!("{}", m.to_ptn::<T>());
        // }
        // if depth == 5 && last_move.is_some() {
        //     println!("    {}", m.to_ptn::<T>());
        // }
        // let pv_move = info.pv_move(board);
        // let side = board.side_to_move();
        // let flat_diff = board.flat_diff(side);
        let rev_move = board.do_move(m);
        let mut next_extensions = extensions;
        // Extend if the pv is to make a "bad capture"
        // if let Some(pv_move) = pv_move {
        //     if m == pv_move && m.is_stack_move() {
        //         next_extensions += 1;
        //         // if let Some(last_move) = last_move {
        //         //     if last_move.dest_sq == rev_move.dest_sq {
        //         //         next_extensions += 1;
        //         //     }
        //         // }
        //         // let new_flat_diff = board.flat_diff(side);
        //         // if new_flat_diff - flat_diff <= 1 {
        //         //     next_extensions += 1;
        //         // }
        //     } else {
        //         next_extensions = 0;
        //     }
        //     // } else {
        //     //     next_extensions = 0;
        //     // }
        // }
        let next_depth = if tak_history.consecutive(info.ply_depth(board)) && extensions < 2 {
            // dbg!("extending!");
            next_extensions += 1;
            depth - 1
        } else {
            depth - 1
        };
        // if next_extensions >= 3 {
        //     // dbg!(info.ply_depth(board));
        //     next_extensions = 0;
        //     next_depth += 1;
        // }
        // match last_move {
        //     Some(last_rev) if last_rev.game_move.is_stack_move() => {
        //         if extensions <= 2 && last_rev.dest_sq == rev_move.dest_sq {
        //             // Must be a capture, I think
        //             extensions += 1;
        //             next_depth += 1;
        //         }
        //     }
        //     _ => {}
        // }
        //         value = PVS(-(alpha+1),-alpha)
        // if(value > alpha && value < beta) {
        //   value = PVS(-beta,-alpha);
        // }

        let mut score = 0;

        // late move reduction
        if depth > LMR_DEPTH_LIMIT  && count >= LMR_FULL_SEARCH_MOVES {
            score = -1 * alpha_beta(
                    board,
                    evaluator,
                    info,
                    SearchData::new(
                        -beta,
                        -alpha,
                        next_depth - 2,
                        true,
                        Some(rev_move),
                        next_extensions,
                        tak_history,
                    ),
                );
            // re-search full depth moves and those that don't fail low
            if score > alpha {
                score = -1 * alpha_beta(
                        board,
                        evaluator,
                        info,
                        SearchData::new(
                            -beta,
                            -alpha,
                            next_depth,
                            true,
                            Some(rev_move),
                            next_extensions,
                            tak_history,
                        ),
                    );
            }
        } else {
            score = -1 * alpha_beta(
                    board,
                    evaluator,
                    info,
                    SearchData::new(
                        -beta,
                        -alpha,
                        next_depth,
                        true,
                        Some(rev_move),
                        next_extensions,
                        tak_history,
                    ),
                );
        }

        board.reverse_move(rev_move);
        if info.stopped {
            return 0;
        }
        if score > alpha {
            if score >= beta {
                if count == 0 {
                    info.stats.fail_high_first += 1;
                }
                info.stats.fail_high += 1;
                info.stats.add_cut(count);
                if m.is_place_move() {
                    info.killer_moves[depth].add(m);
                } else {
                    // const WINNING: i32 = crate::eval::WIN_SCORE - 100;
                    // // Cannot be null?
                    // if score >= WINNING {
                    //     info.hist_moves.update(depth, m);
                    // }
                }
                info.store_move(board, HashEntry::new(board.hash(), m, ScoreCutoff::Beta(beta), depth, board.ply()));
                return beta;
            }
            info.stats.add_alpha(count);
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
                    HashEntry::new(board.hash(), best, ScoreCutoff::Exact(best_score), depth, board.ply()),
                );
            } else {
                info.store_move(
                    board,
                    HashEntry::new(board.hash(), best, ScoreCutoff::Alpha(alpha), depth, board.ply()),
                )
            }
        }
    } else {
        info.stats.bad_search += 1;
        // Waste of time?
    }
    alpha
}

// fn q_search<T, E>(board: &mut T, evaluator: &E, info: &mut SearchInfo, data: SearchData) -> i32
// where
//     T: TakBoard,
//     E: Evaluator<Game = T>,
// {
//     let opp_critical = board.bits().road_pieces(!board.side_to_move()) & board.bits().empty();
//     // Opponent has two winning placements, we must move a stack
//     if opp_critical.pop_count() >= 2 {

//     }
//     0
// }

fn road_at_a_glance() {
    // I
    todo!();
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
