use crate::search::SearchInfo;

const GEN_THOROUGH_ORDER_DEPTH: usize = 1; // where to stop bothering with accurate move ordering

use super::*;

const WIN_SORT: i16 = 10_000;
const PV_SORT: i16 = 5_000;
const TAK_SORT: i16 = 500;

#[derive(Clone)]
pub struct SmartMoveBuffer {
    moves: Vec<ScoredMove>,
    placement_start: usize,
    top_placements: SimpleMoveList<GameMove>,
    top_stack_moves: SimpleMoveList<ScoredMove>,
    // stack_hist: Vec<i16>,
    queries: usize,
    best_fcd: i8,
    // flat_attempts: i16,
}

impl SmartMoveBuffer {
    const THOROUGH_MOVES: usize = 16;
    pub fn new(buffer_size: usize) -> Self {
        Self {
            moves: Vec::with_capacity(buffer_size),
            placement_start: 0,
            // stack_hist: vec![0; buffer_size],
            top_placements: SimpleMoveList::new(),
            top_stack_moves: SimpleMoveList::new(),
            queries: 0,
            best_fcd: 1,
            // flat_attempts: 0,
        }
    }
    pub fn clear(&mut self) {
        self.moves.clear();
        self.placement_start = 0;
        self.top_placements.clear();
        self.top_stack_moves.clear();
        self.queries = 0;
        self.best_fcd = 1;
    }
    pub fn gen_score_stack<T: TakBoard>(&mut self, board: &mut T, info: &mut SearchInfo) {
        if board.ply() >= 4 {
            self.gen_score_stack_moves(board, &info.capture_hist);
        }
    }
    pub fn gen_score_place<T: TakBoard>(
        &mut self,
        depth: usize,
        board: &mut T,
        info: &mut SearchInfo,
    ) {
        self.gen_score_place_moves(
            board,
            info,
            board.ply() >= 4
                && depth > GEN_THOROUGH_ORDER_DEPTH
                && board.pieces_reserve(board.side_to_move()) > 1,
        );
    }
    pub fn gen_and_score<T: TakBoard>(
        &mut self,
        depth: usize,
        board: &mut T,
        info: &mut SearchInfo,
    ) {
        if self.moves.len() > 0 {
            // we have a road in 1, no further move generation needed!
            return;
        }
        if board.ply() >= 4 {
            self.gen_score_stack_moves(board, &info.capture_hist);
        }
        self.gen_score_place_moves(
            board,
            info,
            board.ply() >= 4
                && depth > GEN_THOROUGH_ORDER_DEPTH
                && board.pieces_reserve(board.side_to_move()) > 1,
        );
    }
    pub fn apply_history_penalty<const T: usize>(
        &self,
        bonus: i32,
        cut_move: GameMove,
        hist: &mut PlaceHistory<T>,
    ) {
        // Todo is this too slow?
        for mv in self.top_placements.iter() {
            if cut_move.is_stack_move()
                || !(mv.place_piece() == cut_move.place_piece()
                    && mv.src_index() == cut_move.src_index())
            {
                hist.update(bonus, *mv);
            }
        }
    }
    pub fn apply_stack_penalty(
        &self,
        side_to_move: Color,
        bonus: i32,
        cut_move: GameMove,
        hist: &mut CaptureHistory,
    ) {
        // Todo is this too slow?
        for smv in self.top_stack_moves.iter() {
            let mv = smv.mv;
            if cut_move.is_place_move()
                || !(cut_move.src_index() == mv.src_index()
                    && cut_move.unique_slide_idx() == mv.unique_slide_idx())
            {
                hist.update(side_to_move, bonus, mv);
            }
            // if mv.is_stack_move() && smv.changed_bits != changed_bits {
            //     hist.update(side_to_move, bonus, smv.changed_bits);
            // }
            // let diff = smv.changed_bits & (smv.changed_bits ^ changed_bits);
        }
    }
    pub fn drop_below_score(&mut self, threshold: i16) -> usize {
        let original = self.moves.len();
        self.moves.retain(|x| x.score >= threshold);
        original - self.moves.len()
    }
    pub fn best_fcd_count(&self) -> i8 {
        self.best_fcd
    }
    pub fn gen_score_stack_moves<T: TakBoard>(&mut self, board: &T, captures: &CaptureHistory) {
        generate_all_stack_moves(board, &mut self.moves);
        self.placement_start = self.moves.len();
        let active_side = board.side_to_move();
        let has_reserves = board.pieces_reserve(active_side) > 1;
        let mut stack_idx = usize::MAX;
        // Moves should be grouped by src index due to generator impl
        let mut stack_data = [Piece::WhiteFlat; 8]; // Todo T::SIZE + 1 when rust figures out this valid
        for x in self.moves.iter_mut().filter(|x| x.mv.is_stack_move()) {
            // let mut score = info.stack_moves.score(x.mv);
            let mut fcd = -1; // Assume we lost control of the original stack
            let src_idx = x.mv.src_index();
            // TODO figure out if this is right
            if src_idx != stack_idx {
                stack_idx = src_idx;
                // Update stack data
                // let number = x.mv.number() as usize;
                let stack = board.index(src_idx);
                let limit = std::cmp::min(stack.len(), T::SIZE);
                for i in 0..limit {
                    stack_data[limit - i] = stack.from_top(i).unwrap();
                }
                if let Some(piece) = stack.from_top(limit) {
                    stack_data[0] = piece;
                }
            }
            // [2, 3, 2, 6, 2, 3, 2, 0, 10, 1, 0, 2, 0, -1, 3, 0, 0, 2]
            let mut offset = 0;
            let mut dest = board.index(stack_idx);
            let mut covered_an_enemy = false;
            let mut contiguous = true;
            for step in x.mv.quantity_iter(T::SIZE) {
                debug_assert!(step.quantity > 0);
                offset += step.quantity as usize;
                dest = board.index(step.index);
                let covered = dest.top();
                let covering = stack_data[offset];
                let bit_idx = 1 << step.index;
                if let Some(piece) = covered {
                    if piece.owner() == active_side {
                        if piece.is_flat() {
                            fcd -= 1;
                        }
                    } else {
                        covered_an_enemy = true;
                        if piece.is_flat() {
                            fcd += 1;
                        }
                    }
                }
                if covering.owner() == active_side {
                    fcd += 1;
                } else {
                    fcd -= 1;
                    contiguous = false;
                }
            }
            let src_stack = board.index(src_idx);
            let top_piece = src_stack.top().unwrap();
            if let Some(piece) = src_stack.from_top(x.mv.number() as usize) {
                if piece.owner() == active_side && top_piece.is_blocker() {
                    fcd += 1;
                } else if piece.owner() != active_side {
                    fcd -= 1;
                }
            }
            // Here score == fcd
            if fcd <= 0 {
                let q = x.mv.number();
                if top_piece.is_flat() && !covered_an_enemy {
                    if q <= 2 {
                        x.score -= 400;
                    } else if q <= 3 {
                        x.score -= 50;
                    }
                }
                if q == 1 && top_piece.is_wall() && has_reserves {
                    x.score -= 200;
                }
            } else if fcd <= 1 {
                let q = x.mv.number();
                if top_piece.is_flat() && !covered_an_enemy {
                    if q <= 2 {
                        x.score -= 50;
                    } else if q <= 3 {
                        x.score -= 25;
                    }
                }
            }
            if top_piece.is_flat() && !contiguous {
                x.score -= 50;
            }
            if x.mv.crush() {
                x.score += 85;
            }
            if top_piece.is_cap() {
                x.score += 10 * ((dest.len() as i16) - (src_stack.len() as i16));
            }
            if top_piece.is_wall() {
                x.score += 10 * ((dest.len() as i16) - (src_stack.len() as i16));
            }
            x.score += 65 * fcd;
            let fcd = fcd as i8;
            x.fcd = fcd;
            if fcd > self.best_fcd {
                self.best_fcd = fcd;
            }
            x.score += captures.score_stack_move(active_side, x.mv);
        }
    }
    pub fn gen_score_place_moves<T: TakBoard>(
        &mut self,
        board: &mut T,
        info: &mut SearchInfo,
        check_tak: bool,
    ) {
        let side = board.side_to_move();
        for idx in board.empty_tiles() {
            let mut flat_score = 100 + info.hist_moves.flat_score(idx, side);
            let mut wall_score = -50 + info.hist_moves.wall_score(idx, side);
            let mut cap_score = info.hist_moves.cap_score(idx, side);
            let neighbors = T::Bits::index_to_bit(idx).adjacent();
            let enemies = neighbors & board.bits().all_pieces(!side);
            let mut num_enemies = 0;
            for n_idx in BitIndexIterator::new(enemies) {
                let n_stack = board.index(n_idx);
                if n_stack.len() > 1 {
                    wall_score += 50;
                    cap_score += 50;
                }
                if !n_stack.top().unwrap().is_wall() {
                    num_enemies += 1;
                }
            }
            if num_enemies >= 3 {
                flat_score -= 20;
                cap_score += 20;
                wall_score += 20;
            }
            if num_enemies != 0 {
                wall_score += 75;
            }
            let friends = neighbors & board.bits().all_pieces(side);
            if friends.is_zero() {
                cap_score -= 25;
                flat_score -= 50;
            }
            let mut tak_score = 0;
            if check_tak {
                let mv = GameMove::from_placement(Piece::flat(side), idx);
                let rev = board.do_move(mv);
                board.null_move();
                let buffer = info.buffer();
                if let Some(_winning) = board.can_make_road(buffer, None) {
                    tak_score += TAK_SORT;
                }
                board.rev_null_move();
                board.reverse_move(rev);
                buffer.clear();
            }
            if board.caps_reserve(board.side_to_move()) > 0 && board.ply() >= 4 {
                self.moves.push(ScoredMove::new(
                    GameMove::from_placement(Piece::cap(side), idx),
                    cap_score + tak_score,
                    TAK_SORT != 0,
                    0,
                ));
            }
            if board.pieces_reserve(board.side_to_move()) > 0 {
                if board.ply() >= 4 {
                    self.moves.push(ScoredMove::new(
                        GameMove::from_placement(Piece::wall(side), idx),
                        wall_score,
                        false,
                        0,
                    ));
                }
                self.moves.push(ScoredMove::new(
                    GameMove::from_placement(Piece::flat(side), idx),
                    flat_score + tak_score,
                    TAK_SORT != 0,
                    1,
                ));
            }
        }
    }
    pub fn remove(&mut self, mv: GameMove) {
        if let Some(pos) = self.moves.iter().position(|m| m.mv == mv) {
            self.moves.remove(pos);
        }
    }
    pub fn score_pv_move(&mut self, pv_move: GameMove) {
        if let Some(found) = self.moves.iter_mut().find(|m| m.mv == pv_move) {
            found.score += PV_SORT;
        }
    }
    pub fn score_tak_threats(&mut self, tak_threats: &[GameMove]) {
        for m in self.moves.iter_mut() {
            if tak_threats.contains(&m.mv) {
                m.score += TAK_SORT;
            }
        }
    }
    pub fn score_wins(&mut self, winning_moves: &[GameMove]) {
        for m in self.moves.iter_mut() {
            if winning_moves.contains(&m.mv) {
                m.score += WIN_SORT
            }
        }
    }
    pub fn get_lmr_reduced_depth(&self, depth: usize, improving: bool) -> usize {
        let reduction = (depth as f32).log2()
            + (self.queries.clamp(1, Self::THOROUGH_MOVES) as f32).log2()
            - 2.0;
        depth - (reduction.floor() as usize).clamp(2, depth - 1) // Reduce at minimum 2, at max to depth 1
    }
    pub(crate) fn get_best_scored(
        &mut self,
        info: &SearchInfo,
        prev: Option<RevGameMove>,
    ) -> ScoredMove {
        if self.queries <= Self::THOROUGH_MOVES {
            self.queries += 1;
            let (idx, m) = self
                .moves
                .iter()
                .enumerate()
                .max_by_key(|(_i, &m)| {
                    m.score
                        + info.counter_moves.score_stack_move(
                            prev.map(|x| x.game_move).unwrap_or(GameMove::null_move()),
                            m.mv,
                        ) as i16
                })
                .unwrap();
            let m = *m;
            if m.mv.is_place_move() {
                self.top_placements.try_append(m.mv);
            } else {
                self.top_stack_moves.try_append(m);
            }
            self.moves.swap_remove(idx);
            m
        } else {
            // Probably an all node, so search order doesn't really matter
            let x = self.moves.pop().unwrap();
            x
        }
    }
    pub fn get_best(&mut self, info: &SearchInfo, prev: Option<RevGameMove>) -> GameMove {
        self.get_best_scored(info, prev).mv
    }
    pub fn len(&self) -> usize {
        self.moves.len()
    }
}

#[derive(Debug, Clone)]
struct SimpleMoveList<T> {
    data: [T; 10],
    idx: usize,
}

impl<T> SimpleMoveList<T>
where
    T: Default + Clone + Copy,
{
    fn new() -> Self {
        Self {
            data: [T::default(); 10],
            idx: 0,
        }
    }
    fn clear(&mut self) {
        self.idx = 0;
    }
    /// Adds a new move to the list if there is sufficient capacity, else noop
    fn try_append(&mut self, mv: T) {
        if self.idx < self.data.len() {
            self.data[self.idx] = mv;
            self.idx += 1;
        }
    }
    fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().take(self.idx)
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct ScoredMove {
    pub(crate) mv: GameMove,
    pub(crate) score: i16,
    pub(crate) is_tak: bool,
    pub(crate) fcd: i8,
}

impl ScoredMove {
    fn new(mv: GameMove, score: i16, is_tak: bool, fcd: i8) -> Self {
        Self {
            mv,
            score,
            is_tak,
            fcd,
        }
    }
}

impl MoveBuffer for SmartMoveBuffer {
    fn add_move(&mut self, mv: GameMove) {
        // (bits + self.number() + 10 * self.is_stack_move() as u64)
        if mv.is_place_move() {
            if mv.place_piece().is_wall() {
                self.moves.push(ScoredMove::new(mv, 2, false, 0));
            } else {
                self.moves.push(ScoredMove::new(mv, 3, false, 1));
            }
        } else {
            self.moves.push(ScoredMove::new(mv, 0, false, 0));
        }
    }

    fn add_scored(&mut self, mv: GameMove, score: i16) {
        self.moves.push(ScoredMove::new(mv, score, false, 0))
    }
}

#[derive(Clone)]
pub struct CounterMoves<const SIZE: usize> {
    white_flat: [GameMove; SIZE],
    black_flat: [GameMove; SIZE],
    white_wall: [GameMove; SIZE],
    black_wall: [GameMove; SIZE],
    all_cap: [GameMove; SIZE],
}

impl<const SIZE: usize> CounterMoves<SIZE> {
    pub fn new() -> Self {
        Self {
            black_flat: [GameMove::null_move(); SIZE],
            white_flat: [GameMove::null_move(); SIZE],
            white_wall: [GameMove::null_move(); SIZE],
            black_wall: [GameMove::null_move(); SIZE],
            all_cap: [GameMove::null_move(); SIZE],
        }
    }
    pub fn clear(&mut self) {
        let mut new = Self::new();
        std::mem::swap(self, &mut new);
    }
    pub fn update(&mut self, prev: GameMove, response: GameMove) {
        if prev.is_place_move() && response.is_stack_move() {
            let idx = prev.src_index();
            let table = match prev.place_piece() {
                Piece::WhiteFlat => &mut self.white_flat,
                Piece::BlackFlat => &mut self.black_flat,
                Piece::WhiteCap | Piece::BlackCap => &mut self.all_cap,
                Piece::WhiteWall => &mut self.white_wall,
                Piece::BlackWall => &mut self.black_wall,
            };
            table[idx] = response;
        }
    }
    pub fn score_stack_move(&self, prev: GameMove, candidate: GameMove) -> i32 {
        if prev.is_place_move() && candidate.is_stack_move() {
            let idx = prev.src_index();
            let table = match prev.place_piece() {
                Piece::WhiteFlat => &self.white_flat,
                Piece::BlackFlat => &self.black_flat,
                Piece::WhiteCap | Piece::BlackCap => &self.all_cap,
                Piece::WhiteWall => &self.white_wall,
                Piece::BlackWall => &self.black_wall,
            };
            let response = table[idx];
            if response == candidate {
                return 100;
            }
        }
        0
    }
}

#[derive(Clone, Copy)]
pub struct KillerMoves {
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
            10
        } else {
            0
        }
    }
}

#[derive(Clone, Debug)]
pub struct CorrHist {
    pub(crate) white: [i32; 512],
    pub(crate) black: [i32; 512],
}

impl CorrHist {
    pub fn new() -> Self {
        Self {
            white: [0; 512],
            black: [0; 512],
        }
    }
    pub fn update<T: TakBoard>(&mut self, board: &T, depth: usize, eval_diff: i32) {
        const MAX_HISTORY: i32 = 1 << 14;
        let table = match board.side_to_move() {
            Color::White => &mut self.white,
            Color::Black => &mut self.black,
        };
        let weight = depth;
        let scaled_diff = eval_diff / 8;
        let clamped = (scaled_diff * weight as i32).clamp(-MAX_HISTORY, MAX_HISTORY);
        let idx = (board.bits().blocker_hash() % 512) as usize;
        table[idx] += clamped - table[idx] * clamped.abs() / MAX_HISTORY;
    }
    pub fn correction<T: TakBoard>(&self, board: &T) -> i32 {
        const CORRECTION_FACTOR: i32 = 1 << 9;
        let table = match board.side_to_move() {
            Color::White => &self.white,
            Color::Black => &self.black,
        };
        let idx = (board.bits().blocker_hash() % 512) as usize;
        let out = table[idx] / CORRECTION_FACTOR;
        assert!(out.abs() <= 32);
        out
    }
}

#[derive(Clone)]
pub struct CaptureHistory {
    white: Vec<i32>,
    black: Vec<i32>,
}

impl CaptureHistory {
    const MAX_HISTORY: i32 = 256;
    pub fn new(size: usize) -> Self {
        assert_eq!(size, 6);
        Self {
            white: vec![0; 64 * 4 * 36],
            black: vec![0; 64 * 4 * 36],
        }
    }
    pub fn clear(&mut self) {
        for i in 0..self.white.len() {
            self.white[i] = 0;
            self.black[i] = 0;
        }
        // let mut new = Self::new();
        // std::mem::swap(self, &mut new);
    }
    pub fn update(&mut self, side_to_move: Color, bonus: i32, mv: GameMove) {
        let clamped = bonus.clamp(-Self::MAX_HISTORY, Self::MAX_HISTORY);
        let table = match side_to_move {
            Color::White => &mut self.white,
            Color::Black => &mut self.black,
        };
        let idx = mv.src_index() * 256 + mv.unique_slide_idx();
        table[idx] += clamped - table[idx] * clamped.abs() / Self::MAX_HISTORY;
    }
    fn score_stack_move(&self, side_to_move: Color, mv: GameMove) -> i16 {
        let idx = mv.src_index() * 256 + mv.unique_slide_idx();
        let table = match side_to_move {
            Color::White => &self.white,
            Color::Black => &self.black,
        };
        let sign = table[idx].signum() as i16;
        let raw_score = 5 * (1 + table[idx].abs()).isqrt();
        // let raw_score = table[idx].abs();
        let total_score = sign * raw_score as i16;
        return total_score.clamp(-Self::MAX_HISTORY as i16, Self::MAX_HISTORY as i16);
    }
}

#[derive(Clone)]
pub struct PlaceHistory<const SIZE: usize> {
    white_flat: [i32; SIZE],
    black_flat: [i32; SIZE],
    white_wall: [i32; SIZE],
    black_wall: [i32; SIZE],
    white_cap: [i32; SIZE],
    black_cap: [i32; SIZE],
}

impl<const SIZE: usize> PlaceHistory<SIZE> {
    pub fn new() -> Self {
        Self {
            black_flat: [0; SIZE],
            white_flat: [0; SIZE],
            white_wall: [0; SIZE],
            black_wall: [0; SIZE],
            white_cap: [0; SIZE],
            black_cap: [0; SIZE],
        }
    }
    pub fn mean_flat_score(&self, side: Color) -> i16 {
        let mut total = 0;
        let mut count = 0;
        for idx in 0..SIZE {
            let score = self.flat_score(idx, side);
            if score.abs() >= 10 {
                total += self.flat_score(idx, side);
                count += 1;
            }
        }
        100 + (total as f32 / count as f32) as i16
    }
    pub fn clear(&mut self) {
        let mut new = Self::new();
        std::mem::swap(self, &mut new);
    }
    pub fn update(&mut self, bonus: i32, mv: GameMove) {
        const MAX_HISTORY: i32 = 256;
        let clamped = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
        let idx = mv.src_index();
        let table = match mv.place_piece() {
            Piece::WhiteFlat => &mut self.white_flat,
            Piece::BlackFlat => &mut self.black_flat,
            Piece::WhiteCap => &mut self.white_cap,
            Piece::BlackCap => &mut self.black_cap,
            Piece::WhiteWall => &mut self.white_wall,
            Piece::BlackWall => &mut self.black_wall,
        };
        table[idx] += clamped - table[idx] * clamped.abs() / MAX_HISTORY;
    }
    pub fn flat_score(&self, idx: usize, color: Color) -> i16 {
        let table = match color {
            Color::White => &self.white_flat,
            Color::Black => &self.black_flat,
        };
        let sign = table[idx].signum() as i16;
        let raw_score = 10 * (1 + table[idx].abs()).isqrt();
        // let raw_score = table[idx].abs();
        sign * raw_score as i16
    }
    pub fn cap_score(&self, idx: usize, color: Color) -> i16 {
        let table = match color {
            Color::White => &self.white_cap,
            Color::Black => &self.black_cap,
        };
        let sign = table[idx].signum() as i16;
        let raw_score = 10 * (1 + table[idx].abs()).isqrt();
        // let raw_score = table[idx].abs();
        sign * raw_score as i16
    }
    pub fn wall_score(&self, idx: usize, color: Color) -> i16 {
        let table = match color {
            Color::White => &self.white_wall,
            Color::Black => &self.black_wall,
        };
        let sign = table[idx].signum() as i16;
        let raw_score = 10 * (1 + table[idx].abs()).isqrt();
        // let raw_score = table[idx].abs();
        sign * raw_score as i16
    }
    pub fn debug(&self) {
        dbg!(&self.white_flat);
        dbg!(&self.black_flat);
        dbg!(&self.white_wall);
        dbg!(&self.black_wall);
        dbg!(&self.white_cap);
        dbg!(&self.black_cap);
    }
}

impl<const SIZE: usize> std::fmt::Debug for PlaceHistory<SIZE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn construct_table(arr: &[i32], length: usize) -> Result<String, fmt::Error> {
            use std::fmt::Write;
            let mut out = String::new();
            writeln!(&mut out, "[")?;
            for i in 0..length {
                for j in 0..length {
                    write!(&mut out, "{},", arr[j + i * length])?;
                }
                writeln!(&mut out, "")?;
            }
            writeln!(&mut out, "]")?;
            Ok(out)
        }
        f.debug_struct("PlaceHistory")
            .field("white_flat", &construct_table(&self.white_flat, 7)?)
            .field("white_wall", &construct_table(&self.white_wall, 7)?)
            .field("black_flat", &construct_table(&self.black_flat, 7)?)
            .field("black_wall", &construct_table(&self.black_wall, 7)?)
            .field("white_cap", &construct_table(&self.white_cap, 7)?)
            .field("black_cap", &construct_table(&self.black_cap, 7)?)
            .finish()
    }
}

#[derive(Clone)]
pub struct HistoryMoves<const SIZE: usize> {
    to_from: [u32; SIZE],
    board_length: usize,
}

impl<const SIZE: usize> HistoryMoves<SIZE> {
    pub fn new(board_size: usize) -> Self {
        Self {
            to_from: [1; SIZE],
            board_length: board_size,
        }
    }
    pub fn update(&mut self, depth: usize, mv: GameMove) {
        let value = depth as u32;
        let f = mv.src_index();
        let t = mv.dest_sq(self.board_length);
        self.to_from[f * self.board_length * self.board_length + t] += value;
    }
    pub fn score(&self, mv: GameMove) -> i16 {
        let f = mv.src_index();
        let t = mv.dest_sq(self.board_length);
        self.to_from[f * self.board_length * self.board_length + t].ilog2() as i16
    }
}

#[derive(Clone)]
pub struct EvalHist<const SIZE: usize> {
    evals: [i32; SIZE],
}

impl<const SIZE: usize> EvalHist<SIZE> {
    const EMPTY: i32 = i32::MIN;
    pub fn new() -> Self {
        Self {
            evals: [Self::EMPTY; SIZE],
        }
    }
    /// Negative if the current depth is worse than previous ply, else positive
    pub fn eval_diff_from_last_move(&self, ply_depth: usize) -> Option<i32> {
        Some(self.get_eval(ply_depth)? + self.get_eval(ply_depth - 1)?)
    }
    pub fn get_previous_eval(&self, ply: usize) -> Option<i32> {
        ply.checked_sub(2).map(|idx| self.evals[idx])
    }
    pub fn is_improving(&self, ply: usize) -> bool {
        let eval = self.evals[ply];
        if eval == Self::EMPTY {
            return false;
        }
        // assert!(eval != Self::EMPTY);
        if let Some(prev) = self.get_previous_eval(ply) {
            eval > prev
        } else {
            false
        }
    }
    pub fn set_eval(&mut self, ply: usize, score: i32) {
        self.evals[ply] = score;
    }
    pub fn get_eval(&self, ply: usize) -> Option<i32> {
        let eval = self.evals[ply];
        if eval == Self::EMPTY {
            None
        } else {
            Some(eval)
        }
    }
}
