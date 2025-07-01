use crate::search::SearchInfo;

use super::*;

pub struct SmartMoveBuffer {
    moves: Vec<ScoredMove>,
    top_placements: [GameMove; 10],
    top_placement_idx: usize,
    // stack_hist: Vec<i16>,
    queries: usize,
    // flat_attempts: i16,
}

impl SmartMoveBuffer {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            moves: Vec::new(),
            // stack_hist: vec![0; buffer_size],
            top_placements: [GameMove::null_move(); 10],
            queries: 0,
            top_placement_idx: 0,
            // flat_attempts: 0,
        }
    }
    pub fn apply_history_penalty<const T: usize>(
        &self,
        bonus: i32,
        cut_move: GameMove,
        hist: &mut PlaceHistory<T>,
    ) {
        // Todo is this too slow?
        for mv in self.top_placements {
            if mv.is_place_move() && mv.place_piece() == cut_move.place_piece() {
                hist.update(bonus, mv);
            }
        }
    }
    pub fn drop_below_score(&mut self, threshold: i16) {
        self.moves.retain(|x| x.score >= threshold);
    }
    pub fn score_stack_moves<T: TakBoard>(&mut self, board: &T) {
        let active_side = board.side_to_move();
        let has_reserves = board.pieces_reserve(active_side) > 1;
        let mut stack_idx = usize::MAX;
        // Moves should be grouped by src index due to generator impl
        let mut stack_data = [Piece::WhiteFlat; 8]; // Todo T::SIZE + 1 when rust figures out this valid
        for x in self.moves.iter_mut().filter(|x| x.mv.is_stack_move()) {
            // let mut score = info.stack_moves.score(x.mv);
            let mut score = 0;
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
            for step in x.mv.quantity_iter(T::SIZE) {
                debug_assert!(step.quantity > 0);
                offset += step.quantity as usize;
                dest = board.index(step.index);
                let covered = dest.top();
                let covering = stack_data[offset];
                if let Some(piece) = covered {
                    if piece.owner() == active_side {
                        score -= 1;
                    } else {
                        covered_an_enemy = true;
                        score += 1;
                    }
                }
                if covering.owner() == active_side {
                    score += 1;
                } else {
                    score -= 1;
                }
            }
            let src_stack = board.index(src_idx);
            let top_piece = src_stack.top().unwrap();
            if let Some(piece) = src_stack.from_top(x.mv.number() as usize) {
                if piece.owner() == active_side {
                    score += 1;
                } else {
                    score -= 1;
                }
            }
            // Here score == fcd
            if score <= 0 {
                let q = x.mv.number();
                if top_piece.is_flat() && !covered_an_enemy {
                    if q <= 2 {
                        x.score -= 30;
                    } else if q <= 3 {
                        x.score -= 5;
                    }
                }
                if q == 1 && top_piece.is_wall() && has_reserves {
                    x.score -= 20;
                }
            } else if score <= 1 {
                let q = x.mv.number();
                if top_piece.is_flat() && !covered_an_enemy {
                    if q <= 2 {
                        x.score -= 5;
                    }
                }
            }
            if top_piece.is_cap() {
                x.score += 2 * (dest.len() as i16) - (src_stack.len() as i16);
            }
            if top_piece.is_wall() {
                x.score += 2 * (dest.len() as i16) - (src_stack.len() as i16) - 2;
            }
            x.score += 2 * score;
        }
    }
    pub fn gen_score_place_moves<T: TakBoard>(&mut self, board: &T, place_hist: &PlaceHistory<49>) {
        let side = board.side_to_move();
        // [2, 3, 2, 6, 2, 3, 2, 0, 10, 1, 0, 2, 0, -1, 3, 0, 0, 2]
        for idx in board.empty_tiles() {
            let mut flat_score = 4 + place_hist.flat_score(idx, side);
            let mut wall_score = 2 + place_hist.wall_score(idx, side);
            let mut cap_score = 5 + place_hist.cap_score(idx);
            let neighbors = T::Bits::index_to_bit(idx).adjacent();
            let enemies = neighbors & board.bits().all_pieces(!side);
            for n_idx in BitIndexIterator::new(enemies) {
                let n_stack = board.index(n_idx);
                // if n_stack.len() > 3
                //     && n_stack.last() != Some(&Piece::WhiteCap)
                //     && n_stack.last() != Some(&Piece::BlackCap)
                // {
                //     wall_score += 3;
                //     cap_score += 5;
                if n_stack.len() > 1 {
                    wall_score += 1;
                    cap_score += 2;
                }
            }
            if enemies.pop_count() >= 3 {
                flat_score -= 1;
                cap_score += 1;
                wall_score += 1;
            }
            let friends = neighbors & board.bits().all_pieces(side);
            if friends.is_zero() {
                cap_score -= 2;
                flat_score -= 1;
            }
            // match friends.pop_count() {
            //     0 => {
            //         flat_score -= 1;
            //         cap_score -= 2;
            //     }
            //     1 | 2 => {
            //         flat_score += 1;
            //         cap_score += 2;
            //     }
            //     3 => {
            //         flat_score += 1;
            //     }
            //     _ => {
            //         wall_score -= 1;
            //         cap_score -= 1;
            //     }
            // }
            if board.caps_reserve(board.side_to_move()) > 0 && board.ply() >= 4 {
                self.moves.push(ScoredMove::new(
                    GameMove::from_placement(Piece::cap(side), idx),
                    cap_score,
                ));
            }
            if board.pieces_reserve(board.side_to_move()) > 0 {
                if board.ply() >= 4 {
                    self.moves.push(ScoredMove::new(
                        GameMove::from_placement(Piece::wall(side), idx),
                        wall_score,
                    ));
                }
                self.moves.push(ScoredMove::new(
                    GameMove::from_placement(Piece::flat(side), idx),
                    flat_score,
                ));
            }
        }
        // let neighbors = T::Bits::index_to_bit(idx).adjacent();
        // todo!()
    }
    pub fn remove(&mut self, mv: GameMove) {
        if let Some(pos) = self.moves.iter().position(|m| m.mv == mv) {
            self.moves.remove(pos);
        }
    }
    pub fn score_pv_move(&mut self, pv_move: GameMove) {
        if let Some(found) = self.moves.iter_mut().find(|m| m.mv == pv_move) {
            found.score += 250;
        }
    }
    pub fn score_tak_threats(&mut self, tak_threats: &[GameMove]) {
        for m in self.moves.iter_mut() {
            if tak_threats.contains(&m.mv) {
                m.score += 50;
            }
        }
    }
    pub fn score_wins(&mut self, winning_moves: &[GameMove]) {
        for m in self.moves.iter_mut() {
            if winning_moves.contains(&m.mv) {
                m.score += 1000
            }
        }
    }
    pub fn get_best(&mut self, info: &SearchInfo, prev: Option<RevGameMove>) -> GameMove {
        if self.queries <= 16 {
            self.queries += 1;
            let (idx, m) = self
                .moves
                .iter()
                .enumerate()
                .max_by_key(|(_i, &m)| {
                    // if m.score < 3 {
                    //     m.score + info.killer_moves[ply % info.max_depth].score(m.mv) as i16
                    // } else {
                    //     m.score
                    // }
                    m.score
                        + info.counter_moves.score_stack_move(
                            prev.map(|x| x.game_move).unwrap_or(GameMove::null_move()),
                            m.mv,
                        ) as i16
                    // m.score // + info.killer_moves[ply % info.max_depth].score(m.mv) as i16
                    // - self.penalty_hist_score(m.mv)
                })
                .unwrap();
            let m = *m;
            if m.mv.is_place_move() && self.top_placement_idx < self.top_placements.len() {
                self.top_placements[self.top_placement_idx] = m.mv;
                self.top_placement_idx += 1;
            }
            // if m.mv.is_stack_move() {
            //     let hist_score = &mut self.stack_hist[m.mv.src_index()];
            //     if *hist_score < 10 {
            //         *hist_score += 1;
            //     }
            // } else if m.mv.place_piece().is_flat() {
            //     self.flat_attempts += 1;
            // }
            self.moves.swap_remove(idx);
            m.mv
        } else {
            // Probably an all node, so search order doesn't really matter
            let x = self.moves.pop().unwrap();
            x.mv
        }
    }
    // fn penalty_hist_score(&self, mv: GameMove) -> i16 {
    //     if mv.is_stack_move() {
    //         self.stack_hist[mv.src_index()]
    //     } else if mv.place_piece().is_flat() {
    //         self.flat_attempts * 2
    //     } else {
    //         0
    //     }
    // }
    pub fn len(&self) -> usize {
        self.moves.len()
    }
}

#[derive(Clone, Copy)]
struct ScoredMove {
    mv: GameMove,
    score: i16,
}

impl ScoredMove {
    fn new(mv: GameMove, score: i16) -> Self {
        Self { mv, score }
    }
}

impl MoveBuffer for SmartMoveBuffer {
    fn add_move(&mut self, mv: GameMove) {
        // (bits + self.number() + 10 * self.is_stack_move() as u64)
        if mv.is_place_move() {
            if mv.place_piece().is_wall() {
                self.moves.push(ScoredMove::new(mv, 2));
            } else {
                self.moves.push(ScoredMove::new(mv, 3));
            }
        } else {
            self.moves.push(ScoredMove::new(mv, 0));
        }
    }

    fn add_scored(&mut self, mv: GameMove, score: i16) {
        self.moves.push(ScoredMove::new(mv, score))
    }
}

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
                return 5;
            }
        }
        0
    }
}

#[derive(Clone)]
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
            80
        } else {
            0
        }
    }
}

pub struct PlaceHistory<const SIZE: usize> {
    white_flat: [i32; SIZE],
    black_flat: [i32; SIZE],
    white_wall: [i32; SIZE],
    black_wall: [i32; SIZE],
    all_cap: [i32; SIZE],
}

impl<const SIZE: usize> PlaceHistory<SIZE> {
    pub fn new() -> Self {
        Self {
            black_flat: [0; SIZE],
            white_flat: [0; SIZE],
            white_wall: [0; SIZE],
            black_wall: [0; SIZE],
            all_cap: [0; SIZE],
        }
    }
    pub fn clear(&mut self) {
        let mut new = Self::new();
        std::mem::swap(self, &mut new);
    }
    pub fn update(&mut self, bonus: i32, mv: GameMove) {
        const MAX_HISTORY: i32 = 125;
        let clamped = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
        let idx = mv.src_index();
        let table = match mv.place_piece() {
            Piece::WhiteFlat => &mut self.white_flat,
            Piece::BlackFlat => &mut self.black_flat,
            Piece::WhiteCap | Piece::BlackCap => &mut self.all_cap,
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
        let raw_score = (1 + table[idx].abs()).ilog2();
        sign * raw_score as i16
    }
    pub fn cap_score(&self, idx: usize) -> i16 {
        let sign = self.all_cap[idx].signum() as i16;
        let raw_score = (1 + self.all_cap[idx].abs()).ilog2();
        sign * raw_score as i16
    }
    pub fn wall_score(&self, idx: usize, color: Color) -> i16 {
        let table = match color {
            Color::White => &self.white_wall,
            Color::Black => &self.black_wall,
        };
        let sign = table[idx].signum() as i16;
        let raw_score = (1 + table[idx].abs()).ilog2();
        sign * raw_score as i16
    }
    pub fn debug(&self) {
        dbg!(&self.white_flat);
        dbg!(&self.black_flat);
        dbg!(&self.white_wall);
        dbg!(&self.black_wall);
        dbg!(&self.all_cap);
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
            .field("all_cap", &construct_table(&self.all_cap, 7)?)
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
pub struct EvalHistory<const SIZE: usize> {
    evals: [i32; SIZE],
}

impl<const SIZE: usize> EvalHistory<SIZE> {
    const EMPTY: i32 = i32::MIN;
    pub fn new() -> Self {
        Self {
            evals: [Self::EMPTY; SIZE],
        }
    }
    pub fn get_previous(&self, ply: usize) -> Option<i32> {
        ply.checked_sub(2).map(|idx| self.evals[idx])
    }
    pub fn set_eval(&mut self, ply: usize, score: i32) {
        self.evals[ply] = score;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{transposition_table::HashTable, Board6};
    #[test]
    fn big_stack_order() {
        let tps = "21C,222222,2,x3/2,2,2S,12121S,x,2/2,2,1,1,1,1/x,1S,111112C,1,1,x/1,12112S,x4/x,2,x3,1 2 31";
        let board = Board6::try_from_tps(tps).unwrap();
        let mut moves = SmartMoveBuffer::new(36);
        generate_all_moves(&board, &mut moves);
        moves.score_stack_moves(&board);
        moves.moves.sort_by_key(|x| -x.score);
        assert!(moves.moves[0].score >= moves.moves.last().unwrap().score);
        let table = HashTable::new(1 << 6);
        let info = SearchInfo::new(6, &table);
        let prev = None;
        let order = (0..moves.moves.len()).map(|_| moves.get_best(&info, prev));
        // let order: Vec<_> = moves.moves.into_iter().map(|x| *x.mv).collect();
        for m in order {
            println!("{}", m.to_ptn::<Board6>());
        }
    }
}
