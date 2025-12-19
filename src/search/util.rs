use crate::board::{Board6, TakBoard};
use crate::eval::WIN_SCORE;
use crate::move_gen::{
    CaptureHistory, CorrHist, CounterMoves, EvalHist, GameMove, PlaceHistory, SimpleMoveList,
};
use crate::transposition_table::{HashEntry, HashTable, ScoreCutoff};
use crate::{TeiCommand, TimeBank};
use crossbeam_channel::Receiver;
use instant::Instant;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct SearchInfo<'a> {
    pub max_depth: usize,
    pub(crate) trans_table: &'a HashTable,
    pub(crate) pv_table: PVTable<32>,
    pub hash_history: HashHistory,
    pub counter_moves: CounterMoves<49>, // Todo make this better generic
    pub hist_moves: PlaceHistory<49>,    // Todo make this better generic
    pub eval_hist: EvalHist<50>,         // Todo make this not crash if max depth is set higher
    pub capture_hist: CaptureHistory,    // Todo make this better generic
    pub(crate) corr_hist: CorrHist,
    pub(crate) stopped: bool,
    pub(crate) input: Option<Receiver<TeiCommand>>,
    pub(crate) time_bank: TimeBank,
    pub(crate) start_time: Instant,
    pub(crate) start_ply: usize,
    pub stats: SearchStats,
    pub(crate) early_abort_depth: usize,
    pub(crate) soft_max_nodes: u64,
    pub(crate) hard_max_nodes: u64,
    pub(crate) quiet: bool,
    pub(crate) hyper: SearchHyper,
    pub(crate) extra_move_buffer: Vec<GameMove>,
    pub(crate) main_thread: bool,
    pub(crate) multi_pv: usize,
    pub(crate) forbidden_root_moves: SimpleMoveList<GameMove>,
}

impl<'a> SearchInfo<'a> {
    pub fn new(max_depth: usize, trans_table: &'a HashTable) -> Self {
        let depth_size = (max_depth + 1).next_power_of_two();
        Self {
            max_depth,
            trans_table,
            pv_table: PVTable::new(),
            hash_history: HashHistory::new(Vec::new(), 0),
            counter_moves: CounterMoves::new(),
            hist_moves: PlaceHistory::new(), // Todo init in search
            eval_hist: EvalHist::new(),
            capture_hist: CaptureHistory::new(6),
            corr_hist: CorrHist::new(),
            stopped: false,
            input: None,
            time_bank: TimeBank::flat(120_000), // Some large but not enormous default
            start_time: Instant::now(),
            start_ply: 0,
            stats: SearchStats::new(16),
            early_abort_depth: 6,
            quiet: false,
            soft_max_nodes: u64::MAX,
            hard_max_nodes: u64::MAX,
            hyper: SearchHyper::default(),
            extra_move_buffer: Vec::with_capacity(128),
            main_thread: true,
            multi_pv: 1,
            forbidden_root_moves: SimpleMoveList::new(),
        }
    }
    pub fn set_multi_pv(mut self, num_pvs: usize) -> Self {
        self.multi_pv = std::cmp::max(1, num_pvs);
        self
    }
    pub fn secondary(mut self) -> Self {
        self.main_thread = false;
        self.multi_pv = 1;
        self.forbidden_root_moves = SimpleMoveList::new();
        self
    }
    pub fn buffer(&mut self) -> &mut Vec<GameMove> {
        &mut self.extra_move_buffer
    }
    pub fn nodes(&self) -> usize {
        self.stats.nodes
    }
    pub fn set_max_nodes(mut self, soft: u64, hard: u64) -> Self {
        self.soft_max_nodes = soft;
        self.hard_max_nodes = hard;
        self
    }
    pub fn input_stream(mut self, r: Receiver<TeiCommand>) -> Self {
        self.input = Some(r);
        self
    }
    pub fn time_bank(mut self, time_bank: TimeBank) -> Self {
        self.time_bank = time_bank;
        self
    }
    pub fn with_hyper(mut self, hyper: SearchHyper) -> Self {
        self.hyper = hyper;
        self
    }
    pub fn abort_depth(mut self, depth: usize) -> Self {
        self.early_abort_depth = depth;
        self
    }
    pub fn start_search(&mut self, start_ply: usize) {
        self.start_ply = start_ply;
        self.stopped = false;
        self.hash_history.reset_search_start(start_ply);
        self.stats = SearchStats::new(16);
        self.start_time = Instant::now();
        self.counter_moves = CounterMoves::new();
        self.hist_moves = PlaceHistory::new();
        self.eval_hist = EvalHist::new();
        self.corr_hist = CorrHist::new();
        self.capture_hist.clear();
        self.pv_table.clear();
    }
    pub fn take_input_stream(&mut self) -> Option<Receiver<TeiCommand>> {
        self.input.take()
    }
    pub fn check_stop(&mut self) {
        if self.nodes() as u64 > self.hard_max_nodes {
            self.stopped = true;
            return;
        }
        let secs = self.start_time.elapsed().as_millis();
        if secs >= self.time_bank.goal_time as u128 {
            self.stopped = true;
        }
        if let Some(ref r) = self.input {
            if r.try_recv().is_ok() {
                self.stopped = true;
            }
        }
    }
    pub fn set_history(&mut self, hashes: Vec<u64>) {
        self.hash_history.data = hashes;
    }
    pub fn store_move<E: TakBoard>(&mut self, position: &E, static_eval: i32, entry: HashEntry) {
        self.trans_table.put(position.hash(), entry);
        if !entry.is_forced() {
            match entry.score() {
                ScoreCutoff::Alpha(alpha) => {
                    if alpha >= static_eval {
                        return;
                    }
                }
                ScoreCutoff::Beta(beta) => {
                    if beta <= static_eval {
                        return;
                    }
                }
                ScoreCutoff::Exact(_) => {}
            }
            let eval_diff = entry.get_diff(static_eval);
            self.corr_hist
                .update(position, entry.depth() as usize, eval_diff);
        }
    }
    pub fn lookup_move<E: TakBoard>(&mut self, position: &E) -> Option<HashEntry> {
        self.trans_table.get(&position.hash()).map(|mut e| {
            e.correct_score(self.corr_hist.correction(position));
            e
        })
    }
    pub fn pv_move<E: TakBoard>(&mut self, position: &E) -> Option<GameMove> {
        self.trans_table.get(&position.hash()).map(|e| e.game_move)
    }
    pub fn ply_depth<E: TakBoard>(&self, position: &E) -> usize {
        assert!(position.ply() >= self.start_ply);
        position.ply() - self.start_ply
    }
    pub fn clear_tt(&mut self) {
        self.trans_table.clear();
    }
    pub fn quiet(mut self, val: bool) -> Self {
        self.quiet = val;
        self
    }
}

#[derive(Debug, Clone)]
pub struct HashHistory {
    data: Vec<u64>,
    start_ply: usize,
}

impl HashHistory {
    pub fn new(data: Vec<u64>, start_ply: usize) -> Self {
        Self { data, start_ply }
    }
    pub fn push(&mut self, hash: u64, ply: usize) {
        self.data.push(hash);
        debug_assert_eq!(Some(hash), self.get(ply));
    }
    pub fn pop(&mut self) {
        self.data.pop();
    }
    pub fn get(&self, ply: usize) -> Option<u64> {
        self.data.get(ply.wrapping_sub(self.start_ply + 1)).copied()
    }
    pub fn reset_search_start(&mut self, search_start_ply: usize) {
        self.start_ply = search_start_ply - self.data.len();
        assert!(self.start_ply < 10_000);
    }
    pub fn check_repetition<T: TakBoard>(&self, search_start_ply: usize, board: &T) -> bool {
        let board_ply = board.ply();
        let board_hash = board.hash();
        let fifty_count = board.fifty_move_rule();
        let mut rep_count = 0;
        for i in (2..=fifty_count).step_by(2) {
            let ply = board_ply - i;
            if let Some(hash) = self.get(ply) {
                if hash == board_hash {
                    // Two-fold within the search tree
                    if ply > search_start_ply {
                        return true;
                    }
                    rep_count += 1;
                }
                // 3-fold partially occurring before root
                if rep_count >= 2 {
                    return true;
                }
            } else {
                return false;
            }
        }
        false
    }
}

#[derive(Clone, Debug)]
pub struct SearchHyper {
    pub rfp_margin: i32,
    pub improving_rfp_offset: i32,
    pub fp_margin: i32,
    pub tempo_bonus: i32,
    pub aspiration: i32,
    pub quiet_score: i16,
}

impl SearchHyper {
    pub fn new(
        rfp_margin: i32,
        improving_rfp_offset: i32,
        fp_margin: i32,
        tempo_bonus: i32,
        aspiration: i32,
        quiet_score: i16,
    ) -> Self {
        Self {
            rfp_margin,
            improving_rfp_offset,
            fp_margin,
            tempo_bonus,
            aspiration,
            quiet_score,
        }
    }
}

impl std::default::Default for SearchHyper {
    // fn default() -> Self {
    //     Self::new(111, 35, 40, 0, 50, 21)
    // }
    // fn default() -> Self {
    //     Self::new(139, 33, 43, 23, 59)
    // }
    // fn default() -> Self {
    //     Self {
    //         rfp_margin: super::REVERSE_FUTILITY_MARGIN,
    //         improving_rfp_offset: 35,
    //         fp_margin: super::FUTILITY_MARGIN,
    //         tempo_bonus: 0, // 134 ? or 100
    //         aspiration: super::ASPIRATION_WINDOW,
    //         quiet_score: 40,
    //     }
    // }
    fn default() -> Self {
        Self::new(118, 47, 34, 0, 51, 34)
    }
}

// impl std::default::Default for SearchHyper {
//     fn default() -> Self {
//         Self {
//             rfp_margin: 115,
//             improving_rfp_offset: 30,
//             fp_margin: 75,
//             tempo_bonus: 134,
//             aspiration: 73,
//         }
//     }
// }

#[derive(Debug, Clone)]
pub struct SearchStats {
    pub nodes: usize,
    pub transposition_cutoffs: u64,
    pub ordering_cut: Vec<usize>,
    pub ordering_alpha: Vec<usize>,
    pub bad_search: u64,
}

impl SearchStats {
    pub fn new(ordering_size: usize) -> Self {
        Self {
            nodes: 0,
            transposition_cutoffs: 0,
            ordering_cut: vec![0; ordering_size],
            ordering_alpha: vec![0; ordering_size],
            bad_search: 0,
        }
    }
    pub fn add_cut(&mut self, order: usize) {
        if order >= self.ordering_cut.len() {
            *self.ordering_cut.last_mut().unwrap() += 1;
        } else {
            self.ordering_cut[order] += 1;
        }
    }
    pub fn add_alpha(&mut self, order: usize) {
        if order >= self.ordering_alpha.len() {
            *self.ordering_alpha.last_mut().unwrap() += 1;
        } else {
            self.ordering_alpha[order] += 1;
        }
    }
}
#[derive(Debug, Clone)]
pub struct SearchOutcome<T> {
    pub score: i32,
    pub time: u128,
    pub pv: Vec<GameMove>,
    pub nodes: usize,
    pub depth: usize,
    pub t_cuts: u64,
    pub hashfull: usize,
    pub phantom: PhantomData<T>,
    multi_pv_idx: usize,
}

impl<T> SearchOutcome<T>
where
    T: TakBoard,
{
    pub fn new(score: i32, pv: Vec<GameMove>, depth: usize, search_info: &SearchInfo) -> Self {
        let nodes = search_info.nodes();
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
            hashfull: search_info.trans_table.occupancy(),
            phantom: PhantomData,
            multi_pv_idx: 0,
        }
    }
    pub fn update_multipv_index(&mut self, idx: usize) {
        self.multi_pv_idx = idx;
    }
    pub fn update_multipv(&mut self, last_pv: &Self) {
        self.time = last_pv.time;
        self.nodes = last_pv.nodes;
        self.t_cuts = last_pv.t_cuts;
        self.hashfull = last_pv.hashfull;
    }
    pub fn next(&self) -> Option<GameMove> {
        self.pv.get(0).copied()
    }
    pub fn best_move(&self) -> Option<String> {
        self.pv.get(0).map(|m| m.to_ptn::<T>())
    }
    pub fn score(&self) -> i32 {
        self.score
    }
    pub fn search_depth(&self) -> usize {
        self.depth
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
        if self.multi_pv_idx > 0 {
            write!(
                f,
                "info depth {} multipv {} score cp {} time {} nodes {} nps {} hashfull {} pv {}",
                self.depth,
                self.multi_pv_idx,
                readable_eval(self.score),
                self.time,
                self.nodes,
                nps,
                self.hashfull,
                pv_string,
            )?;
        } else {
            write!(
                f,
                "info depth {} score cp {} time {} nodes {} nps {} hashfull {} pv {}",
                self.depth,
                readable_eval(self.score),
                self.time,
                self.nodes,
                nps,
                self.hashfull,
                pv_string,
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PVTable<const SIZE: usize> {
    pub(crate) table: [[GameMove; SIZE]; SIZE],
    pub(crate) table_length: [usize; SIZE],
}

impl<const SIZE: usize> PVTable<SIZE> {
    fn new() -> Self {
        Self {
            table: [[GameMove::null_move(); SIZE]; SIZE],
            table_length: [0; SIZE],
        }
    }
    pub fn clear(&mut self) {
        for i in 0..SIZE {
            self.table_length[i] = 0;
        }
    }
    pub fn update(&mut self, ply: usize, mv: GameMove) {
        self.table[ply][ply] = mv;
        let next = 1 + ply;
        for idx in next..=next + self.table_length[ply + 1] {
            self.table[ply][idx] = self.table[ply + 1][idx];
        }
        self.table_length[ply] = 1 + self.table_length[ply + 1];
    }
    pub fn get_best(&self) -> Option<GameMove> {
        let mv = self.table[0][0];
        if mv.is_valid() {
            Some(mv)
        } else {
            None
        }
    }
    pub fn get_pv(&self) -> Vec<GameMove> {
        let mut out = Vec::new();
        for i in 0..self.table_length[0] {
            let mv = self.table[0][i];
            if mv.is_valid() {
                out.push(mv);
            } else {
                return out;
            }
        }
        out
    }
}

fn readable_eval(eval: i32) -> i32 {
    if eval.abs() >= (WIN_SCORE - 100) {
        eval
    } else {
        eval / 4
    }
}
