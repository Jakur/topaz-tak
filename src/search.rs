use std::result;

use super::{Color, GameResult};
use crate::board::TakBoard;
use crate::eval::{Evaluator, LOSS_FOUND, WIN_FOUND};
use crate::eval::{LOSE_SCORE, WIN_SCORE};
use crate::move_gen::{GameMove, MoveBuffer, RevGameMove, ScoredMove, SmartMoveBuffer};
use crate::transposition_table::{HashEntry, ScoreCutoff};
use crate::{Bitboard, TeiCommand};

#[cfg(feature = "cli")]
pub mod book;
mod util;
pub use util::{HashHistory, SearchHyper, SearchInfo, SearchOutcome, SearchStats};

const DRAW_SCORE: i32 = 0;

const NULL_REDUCTION_ENABLED: bool = true;
const NULL_REDUCE_PV: bool = false; // probably shouldn't
const NULL_REDUCTION: usize = 6;

// late move reduction parameters
const LMR_ENABLED: bool = true;
const LMR_FULL_SEARCH_MOVES: usize = 4; // number of moves that don't get reduced
const LMR_DEPTH_LIMIT: usize = 2; // don't reduce low depth searches up to this depth
const LMR_REDUCE_PV: bool = true; // probably shouldn't
const LMR_REDUCE_ROOT: bool = true; // probably shouldn't

const PV_SEARCH_ENABLED: bool = true; // no speedup, worse playing strength
const PV_RE_SEARCH_NON_PV: bool = true; // stockfish doesn't... ONLY DISABLE WHEN SOFT CUTOFF

const REVERSE_FUTILITY_MARGIN: i32 = 115;
const FUTILITY_MARGIN: i32 = 50;
// const FUTILITY_MARGIN: i32 = 42;

// aspiration window parameters
// CAN CAUSE CUT-OFF ON ROOT WHEN USED WITH PV_SEARCH!!!
const ASPIRATION_ENABLED: bool = true; // Requires stable search, it's close
const ASPIRATION_WINDOW: i32 = 55;

// internal iterative deepening parameters TODO not tuned yet
// doesn't have much cost attached to it, so why not
const IID_ENABLED: bool = true;
const IID_NON_PV: bool = true;
const IID_MIN_DEPTH: usize = 5;

pub fn multi_search<T, E>(
    board: &mut T,
    eval: &mut E,
    info: &mut SearchInfo,
    num_threads: usize,
) -> Option<SearchOutcome<T>>
where
    T: TakBoard + Send,
    E: Evaluator<Game = T> + Default + Send,
{
    if num_threads <= 1 {
        return search(board, eval, info);
    }
    let outcome = std::thread::scope(|s| {
        let (thread_send, thread_rc) = crossbeam_channel::bounded(num_threads);
        for _ in 1..num_threads {
            let rc = thread_rc.clone();
            let mut thread_info = info.clone().input_stream(rc).quiet(true).secondary();
            let mut eval = E::default();
            let mut board_clone = board.clone();
            let _ = s.spawn(move || {
                search(&mut board_clone, &mut eval, &mut thread_info);
            });
        }
        let outcome = search(board, eval, info);
        for _ in 1..num_threads {
            thread_send.send(TeiCommand::Stop).unwrap();
        }
        outcome
    });
    outcome
}

pub fn search<T, E>(board: &mut T, eval: &mut E, info: &mut SearchInfo) -> Option<SearchOutcome<T>>
where
    T: TakBoard,
    E: Evaluator<Game = T>,
{
    let mut outcome: Option<SearchOutcome<T>> = None;
    // let mut node_counts = vec![1];
    info.start_search(board.ply());
    eval.set_tempo_offset(info.hyper.tempo_bonus);
    let mut alpha = -1_000_000;
    let mut beta = 1_000_000;
    let mut moves: [SmartMoveBuffer; 50] = core::array::from_fn(|_| SmartMoveBuffer::new(0));
    let is_multi_pv = info.multi_pv > 1 && info.main_thread;
    let mut outcomes: Vec<SearchOutcome<T>> = Vec::new();
    'outer: for depth in 1..=info.max_depth {
        // Abort if we are unlikely to finish the search at this depth
        if depth >= info.early_abort_depth {
            let elapsed = info.start_time.elapsed().as_millis();
            if elapsed > info.time_bank.goal_time as u128 / 2 {
                break;
            }
            if info.nodes() as u64 > info.soft_max_nodes {
                break;
            }
        }
        info.forbidden_root_moves.clear();
        let num_pvs = info.multi_pv as usize;
        for pv_idx in 0..num_pvs {
            let mut best_score = alpha_beta(
                board,
                eval,
                info,
                SearchData::new(alpha, beta, depth, true, None, 0, false, true, true),
                &mut moves,
            );
            if ASPIRATION_ENABLED {
                if (best_score <= alpha) || (best_score >= beta) {
                    let diff = beta - alpha;
                    best_score = alpha_beta(
                        board,
                        eval,
                        info,
                        SearchData::new(
                            alpha - 2 * diff,
                            beta + 2 * diff,
                            depth,
                            true,
                            None,
                            0,
                            false,
                            true,
                            true,
                        ),
                        &mut moves,
                    );
                    if (best_score <= alpha) || (best_score >= beta) {
                        best_score = alpha_beta(
                            board,
                            eval,
                            info,
                            SearchData::new(
                                -1_000_000, 1_000_000, depth, true, None, 0, false, true, true,
                            ),
                            &mut moves,
                        )
                    }
                }
                if pv_idx == 0 {
                    alpha = best_score - info.hyper.aspiration;
                    beta = best_score + info.hyper.aspiration;
                }
            }
            let pv_moves = info.pv_table.get_pv();
            // If we had an incomplete depth search, use the previous depth's vals
            if info.stopped {
                if let Some(data) = outcome {
                    let updated = SearchOutcome::new(data.score, data.pv, data.depth, info);
                    outcome = Some(updated);
                }
                break 'outer;
            }
            if pv_idx == 0 {
                outcome = Some(SearchOutcome::new(
                    best_score,
                    pv_moves.clone(),
                    depth,
                    info,
                ));
            }
            if is_multi_pv {
                outcomes.push(SearchOutcome::new(
                    best_score,
                    pv_moves.clone(),
                    depth,
                    info,
                ));
                info.forbidden_root_moves.try_append(pv_moves[0]);
            }
            if !info.quiet {
                if is_multi_pv && pv_idx == num_pvs - 1 {
                    for idx in 0..num_pvs {
                        if let Some(res) = outcomes.get_mut(idx) {
                            res.update_multipv_index(idx + 1);
                        }
                    }
                    if let Some((last_pv, result)) = outcomes.split_last_mut() {
                        for res in result {
                            res.update_multipv(last_pv);
                        }
                    }
                    for line in outcomes.drain(..) {
                        println!("{}", line);
                    }
                } else {
                    if let Some(ref outcome) = outcome {
                        println!("{}", outcome);
                    }
                }
            }
            // Stop wasting time
            if best_score > WIN_SCORE - 10 || best_score < LOSE_SCORE + 10 {
                if pv_idx == 0 {
                    return outcome;
                } else {
                    break;
                }
            }
        }
    }
    info.hist_moves.clear();
    info.counter_moves.clear();
    outcome
}

#[derive(Clone)]
struct SearchData {
    alpha: i32,
    beta: i32,
    depth: usize,
    null_move: bool,
    last_move: Option<RevGameMove>,
    extensions: u8,
    cut_node: bool,
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
        cut_node: bool,
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
            cut_node,
            is_pv,
            is_root,
        }
    }
}

fn alpha_beta<T, E>(
    board: &mut T,
    evaluator: &mut E,
    info: &mut SearchInfo,
    data: SearchData,
    bufs: &mut [SmartMoveBuffer],
) -> i32
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
        cut_node,
        is_pv,
        is_root,
    } = data;
    info.stats.nodes += 1;
    const FREQ: usize = (1 << 12) - 1; // Per 4k nodes
    if (info.nodes() & FREQ) == FREQ {
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
        Some(GameResult::Draw) => return DRAW_SCORE,
        None => {}
    }
    if info.hash_history.check_repetition(info.start_ply, board) {
        return DRAW_SCORE;
    }

    let ply_depth = info.ply_depth(board);
    // let mut road_move = None;
    if depth == 0 {
        let side = board.side_to_move();
        if (board.pieces_reserve(side) == 1 && board.caps_reserve(side) == 0)
            || board.bits().empty().pop_count() == 1
        {
            let flat_diff = board.flat_diff(side);
            if flat_diff >= 0 {
                return WIN_SCORE - board.ply() as i32 + info.start_ply as i32 - 1;
            }
            if flat_diff == -2 && beta < 0 {
                return 0;
            }
        }

        if board
            .can_make_road(&mut info.extra_move_buffer, None)
            .is_some()
        {
            info.extra_move_buffer.clear();
            return WIN_SCORE - board.ply() as i32 + info.start_ply as i32 - 1;
        }

        info.extra_move_buffer.clear();

        let eval = evaluator.evaluate(board, ply_depth) + info.corr_hist.correction(board);
        let opp = !board.side_to_move();
        let opp_flat_adv = board.flat_diff(opp); // Half flats
        let moves = &mut bufs[0];
        if eval > alpha
            && board.caps_reserve(opp) == 0
            && board.pieces_reserve(opp) == 1
            && opp_flat_adv > 2
        {
            moves.clear();
            moves.gen_score_stack(board, info);
            let best_fcd = 2 * moves.best_fcd_count() as i32;
            if opp_flat_adv - best_fcd > 0 {
                return LOSE_SCORE + board.ply() as i32 - info.start_ply as i32 + 1;
            }
        }
        return eval;
    }
    // Transposition Table Lookup
    // Investigate prevalence of null moves in the pv table. It seems to be very rare.
    let pv_entry: Option<HashEntry> = info.lookup_move(board);

    let raw_eval = if let Some(entry) = pv_entry {
        if !is_pv && entry.depth() as usize >= depth {
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
        entry.get_raw_eval() // Todo may need ply_depth fixing if offset is non-zero
    } else {
        let raw_v = evaluator.evaluate(board, ply_depth);
        let eval = raw_v + info.corr_hist.correction(board);
        info.store_move(
            board,
            eval,
            HashEntry::new(
                board.hash(),
                GameMove::null_move(),
                ScoreCutoff::Exact(eval),
                raw_v,
                0,
                board.ply(),
            ),
        );
        raw_v
    };
    // 1 endgame, 0 early game
    let deep_endgame =
        board.pieces_reserve(Color::White) == 1 || board.pieces_reserve(Color::Black) == 1;
    // (R)FP
    let eval = if depth <= 3 && !is_pv {
        // Reverse futility pruning
        let rfp_margin = info.hyper.rfp_margin;
        // let rfp_margin = [info.hyper.rfp_margin, info.hyper.rfp_margin + 15][phase];
        let eval = raw_eval + info.corr_hist.correction(board);
        info.eval_hist.set_eval(ply_depth, eval);
        if info.eval_hist.is_improving(ply_depth) {
            if eval - (depth as i32 * rfp_margin - info.hyper.improving_rfp_offset) >= beta {
                return beta;
            }
        } else {
            if eval - (depth as i32 * rfp_margin) >= beta {
                return beta;
            }
        }
        eval
    } else {
        let eval = raw_eval + info.corr_hist.correction(board);
        info.eval_hist.set_eval(ply_depth, eval);
        eval
    };

    let (moves, rest) = bufs.split_first_mut().unwrap();

    moves.clear();
    let mut has_win = false;
    if board.ply() >= 6 {
        if let Some(mv) = board.can_make_road(&mut info.extra_move_buffer, None) {
            has_win = true;
            moves.add_scored(mv, 10000);
        }
    }
    info.extra_move_buffer.clear();
    let next_cut = if is_pv { false } else { !cut_node };
    let side = board.side_to_move();
    if NULL_REDUCTION_ENABLED
        && (!data.is_pv || NULL_REDUCE_PV)
        && null_move
        && depth > NULL_REDUCTION
        && (board.pieces_reserve(side) > 1 || board.flat_diff(side) >= 1)
    {
        // Require that we can place a new flat without losing
        board.null_move();
        info.hash_history.push(0, board.ply());
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
                    next_cut,
                    false,
                    false,
                ),
                rest,
            );
        board.rev_null_move();
        info.hash_history.pop();
        // If we beta cutoff from the null move, then we can stop searching
        if score >= beta {
            return beta;
        }
    }

    // internal iterative reduction
    let mut reduction = 0;
    if IID_ENABLED && depth >= IID_MIN_DEPTH && (is_pv || IID_NON_PV) && pv_entry.is_none() {
        // Don't bother with the old IID stuff, if you have no pv this node sucks, so just reduce
        reduction += 1;
    }
    // Adjust history with static eval
    if let Some(mv) = last_move {
        let mv = mv.game_move;
        if mv.is_place_move() {
            if let Some(diff) = info.eval_hist.eval_diff_from_last_move(ply_depth) {
                if diff < -20 {
                    info.hist_moves.update(1, mv);
                } else if diff > 20 {
                    info.hist_moves.update(-1, mv);
                }
            }
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

    let mut has_searched_pv = false;
    if !has_win {
        // if we don't have an immediate win, check TT move first
        if let Some(entry) = pv_entry {
            if entry.game_move.is_valid()
                && board.legal_move(entry.game_move)
                && (!is_root
                    || info.multi_pv < 2
                    || !info.forbidden_root_moves.contains(&entry.game_move))
            // TODO maybe a really fast legal checker is faster
            {
                let m = entry.game_move;
                let rev_move = board.do_move(m);
                info.hash_history.push(board.hash(), board.ply());
                let score = -1
                    * alpha_beta(
                        board,
                        evaluator,
                        info,
                        SearchData::new(
                            -beta,
                            -alpha,
                            depth - 1 - reduction,
                            true,
                            Some(rev_move),
                            extensions,
                            next_cut,
                            data.is_pv,
                            false,
                        ),
                        rest,
                    );

                board.reverse_move(rev_move);
                info.hash_history.pop();
                if info.stopped {
                    return 0;
                }
                if score > alpha {
                    if is_pv && info.main_thread {
                        info.pv_table.update(ply_depth, m);
                    }
                    if score >= beta {
                        info.stats.add_cut(0);
                        if m.is_stack_move() {
                            if let Some(prev) = last_move {
                                if prev.game_move.is_place_move() {
                                    // Update countermove
                                    info.counter_moves.update(prev.game_move, m);
                                }
                            }
                        } else {
                            let bonus = 5 * depth as i32 - 4;
                            info.hist_moves.update(bonus, m);
                            // Should be no bad moves, so move on
                        }
                        info.store_move(
                            board,
                            info.eval_hist
                                .get_eval(ply_depth)
                                .unwrap_or_else(|| raw_eval + info.corr_hist.correction(board)),
                            HashEntry::new(
                                board.hash(),
                                m,
                                ScoreCutoff::Beta(beta),
                                raw_eval,
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
    moves.gen_and_score(depth, board, info);
    if is_root && info.multi_pv > 1 {
        moves.drop_arbitrary(&info.forbidden_root_moves);
    }
    if !has_win {
        // moves.gen_score_place(depth, board, info);
        // Futility pruning
        let fp_margin = info.hyper.fp_margin;
        if depth <= 6 && !is_pv && eval + (depth.ilog2() as i32 * depth as i32 * fp_margin) < alpha
        {
            let _num_pruned = moves.drop_below_score(100 - info.hyper.quiet_score);
        }
        // if depth <= 6 && !is_pv && eval + (depth as i32 * fp_margin) < alpha {
        //     let _num_pruned = moves.drop_below_score(0);
        // }
    }

    if let Some(entry) = pv_entry {
        if entry.game_move.is_valid() {
            if has_searched_pv {
                moves.remove(entry.game_move);
            } else {
                moves.score_pv_move(entry.game_move);
            }
        }
    }

    let mut beta_cut = false;

    for c in 0..moves.len() {
        let count = if has_searched_pv { c + 1 } else { c };
        let ScoredMove {
            mv,
            score: mv_order_score,
            is_tak,
            fcd,
        } = moves.get_best_scored(info, last_move);
        // if is_tak {
        //     tak_history = tak_history.add(ply_depth + 1);
        // }
        // let mv = moves.get_best(info, last_move);
        // if is_root && count <= 16 {
        //     // println!(
        //     //     "Depth: {} MC: {} Move {} Move Score {}",
        //     //     depth,
        //     //     count,
        //     //     mv.to_ptn::<T>(),
        //     //     mv_order_score
        //     // );
        //     if count == 16 {
        //         // dbg!(&info.corr_hist.white.iter().max());
        //         dbg!(&info.corr_hist.black[0..10]);
        //         // let mean = info.hist_moves.mean_flat_score(board.side_to_move());
        //         // println!("Mean Flat Score: {mean}");
        //         // info.hist_moves.debug();
        //     }
        // }

        let rev_move = board.do_move(mv);
        info.hash_history.push(board.hash(), board.ply());
        let next_extensions = extensions;

        let next_depth = depth - 1 - reduction;

        let mut score;

        // search first move fully!
        if count == 0 {
            score = -alpha_beta(
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
                    next_cut,
                    data.is_pv,
                    false,
                ),
                rest,
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
                && mv_order_score < 500 // Not a flat placement tak move
                && (LMR_REDUCE_PV || !is_pv)
                && (LMR_REDUCE_ROOT || !is_root)
            {
                reduced_depth = moves.get_lmr_reduced_depth(depth, cut_node, is_pv);
                needs_re_search_on_alpha = true;
            }
            let next_beta = -alpha;
            if PV_SEARCH_ENABLED {
                // && depth > 1 && !data.is_root {
                next_alpha = -(alpha + 1);
                needs_re_search_on_alpha_beta = true;
            }

            score = -alpha_beta(
                board,
                evaluator,
                info,
                SearchData::new(
                    next_alpha,
                    next_beta,
                    reduced_depth,
                    true,
                    Some(rev_move),
                    next_extensions,
                    !cut_node,
                    false,
                    false,
                ),
                rest,
            );

            // re-search full depth moves and those that don't fail low
            if needs_re_search_on_alpha && next_depth > reduced_depth && score > alpha {
                score = -alpha_beta(
                    board,
                    evaluator,
                    info,
                    SearchData::new(
                        next_alpha,
                        next_beta,
                        next_depth,
                        true,
                        Some(rev_move),
                        next_extensions,
                        !cut_node,
                        false,
                        false,
                    ),
                    rest,
                );
            }

            // research with full windows if PV-Search doesn't fail
            if needs_re_search_on_alpha_beta
                && score > alpha
                && score < beta
                && (PV_RE_SEARCH_NON_PV || is_pv)
            {
                score = -alpha_beta(
                    board,
                    evaluator,
                    info,
                    SearchData::new(
                        -beta,
                        next_beta,
                        next_depth,
                        true,
                        Some(rev_move),
                        next_extensions,
                        next_cut,
                        is_pv,
                        false,
                    ),
                    rest,
                );
            }
        }

        board.reverse_move(rev_move);
        info.hash_history.pop();

        if info.stopped {
            return 0;
        }
        if score > alpha {
            if is_pv && info.main_thread {
                info.pv_table.update(ply_depth, mv);
            }

            if score >= beta {
                info.stats.add_cut(count);
                if mv.is_stack_move() {
                    if let Some(prev) = last_move {
                        if prev.game_move.is_place_move() {
                            // Update countermove
                            info.counter_moves.update(prev.game_move, mv);
                        }
                    }
                    let bonus = 5 * depth as i32 - 4;
                    info.capture_hist.update(board.side_to_move(), bonus, mv);
                    // if mv_order_score < 0 {
                    //     info.killers[depth % 64].add(mv);
                    // }
                    // info.killers[depth % 64].add(mv);
                    // Penalize already searched moves
                    moves.apply_stack_penalty(
                        board.side_to_move(),
                        -bonus,
                        mv,
                        &mut info.capture_hist,
                    );
                    moves.apply_history_penalty(-bonus, mv, &mut info.hist_moves);
                } else {
                    // Placement
                    let bonus = 5 * depth as i32 - 4;
                    info.hist_moves.update(bonus, mv);
                    // Penalize already searched moves
                    moves.apply_history_penalty(-bonus, mv, &mut info.hist_moves);
                    moves.apply_stack_penalty(
                        board.side_to_move(),
                        -bonus,
                        mv,
                        &mut info.capture_hist,
                    );
                }
                beta_cut = true;
                best_move = Some(mv);
                best_score = Some(beta);
                break;
            }
            info.stats.add_alpha(count);
            alpha = score;
            best_move = Some(mv);
            best_score = Some(score);
        }
    }

    if let Some(best) = best_move {
        if let Some(best_score) = best_score {
            let cutoff = if beta_cut {
                ScoreCutoff::Beta(beta)
            } else if alpha != old_alpha {
                ScoreCutoff::Exact(best_score)
            } else {
                ScoreCutoff::Alpha(alpha)
            };
            info.store_move(
                board,
                info.eval_hist
                    .get_eval(ply_depth)
                    .unwrap_or_else(|| raw_eval + info.corr_hist.correction(board)),
                HashEntry::new(board.hash(), best, cutoff, raw_eval, depth, board.ply()),
            );
        }
    } else {
        info.stats.bad_search += 1;
        // Waste of time?
    }
    if beta_cut {
        return beta;
    }
    alpha
}

/// A naive minimax function without pruning used for debugging and benchmarking
#[cfg(test)]
pub fn root_minimax<T, E>(board: &mut T, eval: &mut E, depth: u16) -> (Option<GameMove>, i32)
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
#[cfg(test)]
fn naive_minimax<T: TakBoard, E: Evaluator<Game = T>>(
    board: &mut T,
    eval: &mut E,
    depth: u16,
) -> i32 {
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
    use crate::board::{Board5, Board6};
    use crate::eval::{Weights5, Weights6, LOSE_SCORE};
    use crate::transposition_table::*;
    use crate::{Position, TakBoard};
    #[test]
    fn rep_detection() {
        let tps = "2,1,2,2,1,2/2S,1,1,1221C,1,1/112111112S,x2,112S,2,2/1,121S,112C,12,2,2/x2,2,221,2,2/1,1,2,1,1,1 1 38";
        let mvs = "4d5> 3d4> 5e5> 3e4> 6f5< 3f4< b1> c2- 6e5> 3e4> 6f5< 3f4< 6e5> 3e4> 6f5< 3f4< 6e5> 3e4> 6f5< 3f4< 6e5> 3e4> 6f5< 3f4< 6e5> 3e4> 6f5< 3f4< 6e5> 3e4> 6f5< 3f4<";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let search_start_ply = board.ply();
        let mut hash_history = HashHistory::new(Vec::new(), board.ply());
        hash_history.reset_search_start(search_start_ply);
        let mut reps = Vec::new();
        for ptn_mv in mvs.split_whitespace() {
            let mv = GameMove::try_from_ptn(ptn_mv, &board).unwrap();
            board.do_move(mv);
            hash_history.push(board.hash(), board.ply());
            reps.push(hash_history.check_repetition(search_start_ply, &board));
        }
        assert!(reps.iter().any(|x| *x));

        let tps2 = "2,x,21C,x3/2S,1,2,x,1,x/112,x,1,2,2,x/1,1112S,1,12C,1,x/1,x,2,2,x2/1,1,x4 1 18";
        let mvs = "d5 f5 e6 f6 f4 d1 2c6- f3 e3+ b4 e3 d4> e3+ Sd4 b6";
        let mut board = Board6::try_from_tps(tps2).unwrap();
        let search_start_ply = board.ply();
        let mut hash_history = HashHistory::new(Vec::new(), board.ply());
        hash_history.reset_search_start(search_start_ply);
        let mut reps = Vec::new();
        for ptn_mv in mvs.split_whitespace() {
            let mv = GameMove::try_from_ptn(ptn_mv, &board).unwrap();
            board.do_move(mv);
            hash_history.push(board.hash(), board.ply());
            let check = hash_history.check_repetition(search_start_ply, &board);
            reps.push(check);
        }
        assert!(!reps.iter().any(|x| *x));
    }
    #[test]
    fn small_minimax() {
        let tps = "2,1,1,1,1,2S/1,12,1,x,1C,11112/x,2,2,212,2C,11121/2,21122,x2,1,x/x3,1,1,x/x2,2,21,x,112S 1 34";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let mut eval = Weights6::default();
        let (mv, score) = root_minimax(&mut board, &mut eval, 2);
        assert!(score != LOSE_SCORE);
        let only_move = GameMove::try_from_ptn("c5-", &board);
        assert_eq!(mv, only_move);
    }

    #[test]
    fn small_alpha_beta() {
        let tps = "2,1,1,1,1,2S/1,12,1,x,1C,11112/x,2,2,212,2C,11121/2,21122,x2,1,x/x3,1,1,x/x2,2,21,x,112S 1 34";
        let table = HashTable::new(50000);
        let mut board = Board6::try_from_tps(tps).unwrap();
        let mut info = SearchInfo::new(4, &table);
        table.clear();
        let mut eval = Weights6::default();
        search(&mut board, &mut eval, &mut info);
        let tps5 = "x4,1/x3,1,x/x,1,2C,1,2/x,2,1C,2,x/2,x4 1 6";
        let mut board = Board5::try_from_tps(tps5).unwrap();
        let mut info = SearchInfo::new(3, &table);
        let mut eval = Weights5::default();
        search(&mut board, &mut eval, &mut info);
    }
    #[test]
    fn unk_puzzle() {
        let tps = "x2,1,21,2,2/1,2,21,1,21,2/1S,2,2,2C,2,2/21S,1,121C,x,1,12/2,2,121,1,1,1/2,2,x3,22S 1 27";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let table = HashTable::new(100_000);
        let mut eval = Weights6::default();
        let mut info = SearchInfo::new(5, &table);
        search(&mut board, &mut eval, &mut info);
    }
    #[test]
    fn no_capstone_zero_flats() {
        let tps = "2,1,2,1,1,1/2,1,2,2,2,2/1,2,1,1,1,2/1,2,2,1,1,2/1,11112,1112,12,11112,1/1,1,2121,x,2,x 2 33";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let mut eval = Weights6::default();
        eval.evaluate(&board, 2);
        let table = HashTable::new(1 << 14);
        let mut info = SearchInfo::new(5, &table);
        search(&mut board, &mut eval, &mut info);
    }
}
