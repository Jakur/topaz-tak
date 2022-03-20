use super::{Color, GameResult};
use crate::board::TakBoard;
use crate::eval::Evaluator;
use crate::eval::{LOSE_SCORE, WIN_SCORE};
use crate::move_gen::{
    generate_aggressive_place_moves, generate_all_stack_moves, GameMove, HistoryMoves, KillerMoves,
    MoveBuffer, RevGameMove, SmartMoveBuffer,
};
use crate::transposition_table::{HashEntry, HashTable, ScoreCutoff};
use crate::TeiCommand;
use crossbeam_channel::Receiver;
use instant::Instant;
use std::marker::PhantomData;
// use std::time::Instant;

#[cfg(feature = "cli")]
pub mod proof;

const NULL_REDUCTION_ENABLED: bool = true;
const NULL_REDUCE_PV: bool = true; // probably shouldn't
const NULL_REDUCTION: usize = 2;

// late move reduction parameters
const LMR_ENABLED: bool = true;
const LMR_FULL_SEARCH_MOVES: usize = 4; // number of moves that don't get reduced
const LMR_DEPTH_LIMIT: usize = 2; // don't reduce low depth searches up to this depth
const LMR_REDUCE_PV: bool = true; // probably shouldn't
const LMR_REDUCE_ROOT: bool = true; // probably shouldn't

const PV_SEARCH_ENABLED: bool = true; // no speedup, worse playing strength
const PV_RE_SEARCH_NON_PV: bool = true; // stockfish doesn't... ONLY DISABLE WHEN SOFT CUTOFF

// aspiration window parameters
// CAN CAUSE CUT-OFF ON ROOT WHEN USED WITH PV_SEARCH!!!
const ASPIRATION_ENABLED: bool = false; // TERRIBLE performance, search not stable enough
const ASPIRATION_WINDOW: i32 = 55;

// move generation and ordering parameters
const GEN_THOROUGH_ORDER_DEPTH: usize = 1; // where to stop bothering with accurate move ordering

// internal iterative deepening parameters TODO not tuned yet
// doesn't have much cost attached to it, so why not
const IID_ENABLED: bool = true;
const IID_NON_PV: bool = true;
const IID_MIN_DEPTH: usize = 5;
const IID_REDUCTION: usize = 3;
const IID_DIVISION: usize = 2;

pub struct SearchInfo {
    pub max_depth: usize,
    pub nodes: usize,
    pv_table: HashTable,
    pub killer_moves: Vec<KillerMoves>,
    pub hist_moves: HistoryMoves,
    pub stack_win_kill: Vec<KillerMoves>,
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
            stack_win_kill: vec![KillerMoves::new(); max_depth + 1],
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
        let mut counter = 0;
        while let Some(m) = self.pv_move(position) {
            if counter >= 32 {
                break; // Cycle
            }
            if !position.legal_move(m) {
                dbg!("Illegal pv move in {:?}", &position);
                dbg!("Illegal Move in Pv {}", m.to_ptn::<E>());
                break;
            }
            let rev = position.do_move(m);
            backward.push(rev);
            forward.push(m);
            counter += 1;
        }
        for rev_m in backward.into_iter().rev() {
            position.reverse_move(rev_m);
        }
        forward
    }
    fn ply_depth<E: TakBoard>(&self, position: &E) -> usize {
        position.ply() - self.start_ply
    }
    pub fn clear_tt(&mut self) {
        self.pv_table.clear();
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
    pub fn pretty_string(&self) -> String {
        let mut string = format!(
            "info depth {} score cp {} nodes {} pv ",
            self.depth, self.score, self.nodes
        );
        for mv in self.pv.iter().map(|x| x.to_ptn::<T>()) {
            string.push_str(&mv);
            string.push_str(" ");
        }
        string.pop();
        string
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
    let mut alpha = -1_000_000;
    let mut beta = 1_000_000;
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
        let mut best_score = alpha_beta(
            board,
            eval,
            info,
            SearchData::new(alpha, beta, depth, true, None, 0, TakHistory(0), true, true),
        );
        if ASPIRATION_ENABLED {
            if (best_score <= alpha) || (best_score >= beta) {
                best_score = alpha_beta(
                    board,
                    eval,
                    info,
                    SearchData::new(
                        -1_000_000,
                        1_000_000,
                        depth,
                        true,
                        None,
                        0,
                        TakHistory(0),
                        true,
                        true,
                    ),
                )
            }
            alpha = best_score - ASPIRATION_WINDOW;
            beta = best_score + ASPIRATION_WINDOW;
        }
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
            "info depth {} score cp {} nodes {} hashfull {} pv ",
            depth,
            best_score,
            info.nodes,
            info.pv_table.occupancy()
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
    fn check(self, depth: usize) -> bool {
        self.0 & (1 << depth) != 0
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
    is_pv: bool,
    is_root: bool,
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
        is_pv: bool,
        is_root: bool,
    ) -> Self {
        Self {
            alpha,
            beta,
            depth,
            null_move,
            last_move,
            extensions,
            tak_history,
            is_pv,
            is_root,
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
        is_pv,
        is_root,
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
        // // Todo make this work with pv?
        // let side = board.side_to_move();
        // if board.pieces_reserve(side) == 1 && board.caps_reserve(side) == 0 {
        //     if board.flat_diff(side) >= 0 {
        //         return WIN_SCORE - board.ply() as i32 + info.start_ply as i32 - 1;
        //     }
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

    let mut pv_entry: Option<HashEntry> = None;
    let pv_entry_foreign = info.lookup_move(board);

    if let Some(entry) = pv_entry_foreign {
        pv_entry = Some(entry.clone()); // save for move lookup
    }

    if let Some(entry) = pv_entry {
        if entry.depth() as usize >= depth {
            match entry.score() {
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
    if NULL_REDUCTION_ENABLED
        && (!data.is_pv || NULL_REDUCE_PV)
        && null_move
        && depth >= 1 + NULL_REDUCTION
    {
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
                    false,
                    false,
                ),
            );
        if one_depth_score <= LOSE_SCORE + 100 {
            board.rev_null_move();
            tak_history = tak_history.add(info.ply_depth(board));
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
                        false,
                        false,
                    ),
                );
            board.rev_null_move();
            // If we beta cutoff from the null move, then we can stop searching
            if score >= beta {
                return beta;
            }
        }
    }

    // internal iterative deepening
    if IID_ENABLED && depth >= IID_MIN_DEPTH && (is_pv || IID_NON_PV) && !pv_entry.is_some() {
        let reduction = std::cmp::max(IID_REDUCTION, depth / IID_DIVISION);
        alpha_beta(
            board,
            evaluator,
            info,
            SearchData::new(
                alpha,
                beta,
                data.depth - reduction,
                false,
                data.last_move,
                extensions,
                tak_history,
                data.is_pv,
                false,
            ),
        );
        let pv_entry_foreign_2 = info.lookup_move(board);

        if let Some(entry) = pv_entry_foreign_2 {
            pv_entry = Some(entry.clone());
        }
    }

    let mut best_move = None;
    let mut best_score = None;
    let old_alpha = alpha;

    // incremental move generation:
    // step 1: check for placement wins in board.can_make_road()
    // step 2: generate all spreads in board.can_make_road()
    // step 3: check for spread wins in board.can_make_road()
    // step 4: check for TT-Move and search it immediately
    // step 5: score spread moves, generate and score placements in gen_and_score()
    // step 6: search all moves ordered by score.
    let mut stack_moves = Vec::new();
    let mut moves = SmartMoveBuffer::new();

    if board.ply() >= 6 && depth > GEN_THOROUGH_ORDER_DEPTH {
        if let Some(mv) = board.can_make_road(&mut stack_moves, None) {
            let data = &[mv];
            moves.add_move(mv);
            moves.score_wins(data);
        }
    }

    let mut has_searched_pv = false;
    if moves.len() == 0 {
        // if we don't have an immediate win, check TT move first
        if let Some(entry) = pv_entry {
            if (entry.game_move.is_place_move() && board.legal_move(entry.game_move))
                || stack_moves.contains(&entry.game_move)
            // TODO maybe a really fast legal checker is faster
            {
                let m = entry.game_move.clone();
                let rev_move = board.do_move(m);

                let score = -1
                    * alpha_beta(
                        board,
                        evaluator,
                        info,
                        SearchData::new(
                            -beta,
                            -alpha,
                            depth - 1,
                            true,
                            Some(rev_move),
                            extensions,
                            tak_history,
                            data.is_pv,
                            false,
                        ),
                    );

                board.reverse_move(rev_move);
                if info.stopped {
                    return 0;
                }
                if score > alpha {
                    if score >= beta {
                        info.stats.fail_high_first += 1;
                        info.stats.fail_high += 1;
                        info.stats.add_cut(0);
                        if m.is_place_move() {
                            info.killer_moves[board.ply() % info.max_depth].add(m);
                        } else {
                            // const WINNING: i32 = crate::eval::WIN_SCORE - 100;
                            // // Cannot be null?
                            // if score >= WINNING {
                            //     info.hist_moves.update(depth, m);
                            // }
                        }
                        info.store_move(
                            board,
                            HashEntry::new(
                                board.hash(),
                                m,
                                ScoreCutoff::Beta(beta),
                                depth,
                                board.ply(),
                            ),
                        );
                        return beta;
                    }
                    info.stats.add_alpha(0);
                    alpha = score;
                    best_move = Some(m);
                    best_score = Some(score);
                }
                has_searched_pv = true;
            }
        }
    }

    gen_and_score(depth, board, last_move, &mut stack_moves, &mut moves);

    if let Some(entry) = pv_entry {
        if has_searched_pv {
            moves.remove(entry.game_move);
        } else {
            moves.score_pv_move(entry.game_move);
        }
    }

    for c in 0..moves.len() {
        let count = if has_searched_pv { c + 1 } else { c };

        let m = moves.get_best(board.ply(), info);

        let rev_move = board.do_move(m);
        let next_extensions = extensions;

        let next_depth = depth - 1;

        let mut score;

        // search first move fully!
        if count == 0 {
            score = -1
                * alpha_beta(
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
                        data.is_pv,
                        false,
                    ),
                );
        } else {
            let mut needs_re_search_on_alpha = false;
            let mut needs_re_search_on_alpha_beta = false;
            let mut next_alpha = -beta;
            let mut reduced_depth = next_depth;

            // late move reduction
            if LMR_ENABLED
                && depth > LMR_DEPTH_LIMIT
                && count >= LMR_FULL_SEARCH_MOVES
                && (LMR_REDUCE_PV || !is_pv)
                && (LMR_REDUCE_ROOT || !is_root)
                && !tak_history.consecutive(info.ply_depth(board))
            {
                reduced_depth -= 2;
                needs_re_search_on_alpha = true;
            }
            if PV_SEARCH_ENABLED && depth > 1 && !data.is_root {
                next_alpha = -(alpha + 1);
                needs_re_search_on_alpha_beta = true;
            }

            score = -1
                * alpha_beta(
                    board,
                    evaluator,
                    info,
                    SearchData::new(
                        next_alpha,
                        -alpha,
                        reduced_depth,
                        true,
                        Some(rev_move),
                        next_extensions,
                        tak_history,
                        false,
                        false,
                    ),
                );

            // re-search full depth moves and those that don't fail low
            if needs_re_search_on_alpha && score > alpha {
                score = -1
                    * alpha_beta(
                        board,
                        evaluator,
                        info,
                        SearchData::new(
                            next_alpha,
                            -alpha,
                            next_depth,
                            true,
                            Some(rev_move),
                            next_extensions,
                            tak_history,
                            false,
                            false,
                        ),
                    );
            }

            // research with full windows if PV-Search doesn't fail
            if needs_re_search_on_alpha_beta
                && score > alpha
                && score < beta
                && (PV_RE_SEARCH_NON_PV || data.is_pv)
            {
                score = -1
                    * alpha_beta(
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
                            data.is_pv,
                            false,
                        ),
                    );
            }
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
                    info.killer_moves[board.ply() % info.max_depth].add(m);
                } else {
                    const WINNING: i32 = crate::eval::WIN_SCORE - 100;
                    // // Cannot be null?
                    if score >= WINNING && count >= 8 {
                        info.stack_win_kill[board.ply() % info.max_depth].add(m);
                    }
                }
                info.store_move(
                    board,
                    HashEntry::new(board.hash(), m, ScoreCutoff::Beta(beta), depth, board.ply()),
                );
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
                    HashEntry::new(
                        board.hash(),
                        best,
                        ScoreCutoff::Exact(best_score),
                        depth,
                        board.ply(),
                    ),
                );
            } else {
                info.store_move(
                    board,
                    HashEntry::new(
                        board.hash(),
                        best,
                        ScoreCutoff::Alpha(alpha),
                        depth,
                        board.ply(),
                    ),
                )
            }
        }
    } else {
        info.stats.bad_search += 1;
        // Waste of time?
    }
    alpha
}

fn gen_and_score<T>(
    depth: usize,
    board: &mut T,
    last_move: Option<RevGameMove>,
    stack_moves: &mut Vec<GameMove>,
    moves: &mut SmartMoveBuffer,
) where
    T: TakBoard,
{
    // Do a slower, more thorough move ordering near the root
    if depth > GEN_THOROUGH_ORDER_DEPTH && board.ply() >= 6 {
        if moves.len() > 0 {
            // we have a road in 1, no further move generation needed!
            return;
        }
        for m in stack_moves.iter().cloned() {
            moves.add_move(m);
        }
        let mut check_moves = Vec::new();
        generate_aggressive_place_moves(board, &mut check_moves);
        let tak_threats = board.get_tak_threats(&mut check_moves, None);
        moves.gen_score_place_moves(board);
        moves.score_tak_threats(&tak_threats);
        if board.ply() >= 4 {
            moves.score_stack_moves(board, last_move.filter(|x| x.game_move.is_stack_move()));
        }
    } else {
        if board.ply() >= 4 {
            generate_all_stack_moves(board, moves);
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
