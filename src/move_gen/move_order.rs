use super::*;
use crate::search::SearchInfo;

pub struct SmartMoveBuffer {
    moves: Vec<ScoredMove>,
    stack_hist: Vec<i16>,
    queries: usize,
    flat_attempts: i16,
}

impl SmartMoveBuffer {
    pub fn new() -> Self {
        Self {
            moves: Vec::new(),
            stack_hist: vec![0; 36],
            queries: 0,
            flat_attempts: 0,
        }
    }
    pub fn score_stack_moves<T: TakBoard>(&mut self, board: &T) {
        let active_side = board.side_to_move();
        let mut stack_idx = usize::MAX;
        // Moves should be grouped by src index due to generator impl
        let mut stack_data = [Piece::WhiteFlat; 8]; // Todo T::SIZE + 1 when rust figures out this valid
        for x in self.moves.iter_mut().filter(|x| x.mv.is_stack_move()) {
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
            // if &x.mv.to_ptn::<T>() == "6b6>1113" {
            //     panic!()
            //     // println!("{}", "HELLO \n \n \n \n ");
            // }
            let mut offset = 0;
            for step in x.mv.quantity_iter(T::SIZE) {
                debug_assert!(step.quantity > 0);
                offset += step.quantity as usize;
                let covered = board.index(step.index).top();
                let covering = stack_data[offset];
                if let Some(piece) = covered {
                    if piece.owner() == active_side {
                        score -= 1;
                    } else {
                        score += 1;
                    }
                }
                if covering.owner() == active_side {
                    score += 2;
                } else {
                    score -= 2;
                }
            }
            let src_stack = board.index(src_idx);
            if let Some(piece) = src_stack.from_top(x.mv.number() as usize) {
                if piece.owner() == active_side {
                    score += 2;
                } else {
                    score -= 2;
                }
            }
            if let Some(piece) = src_stack.top() {
                if piece.is_cap() {
                    score += 1;
                    if x.mv.crush() {
                        score += 1;
                    }
                }
            }
            x.score += score;
            // if &x.mv.to_ptn::<T>() == DEBUG {
            //     dbg!(x.score);
            // }
        }
    }
    pub fn gen_score_place_moves<T: TakBoard>(&mut self, board: &T, place_hist: &PlaceHistory<36>) {
        let side = board.side_to_move();
        for idx in board.empty_tiles() {
            let mut flat_score = 3 + place_hist.flat_score(idx);
            let mut wall_score = 1 + place_hist.wall_score(idx, side);
            let mut cap_score = 3 + place_hist.cap_score(idx);
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
            if friends.pop_count() == 0 {
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
    pub fn get_best(&mut self, ply: usize, info: &SearchInfo) -> GameMove {
        if self.queries <= 16 {
            self.queries += 1;
            let (idx, m) = self
                .moves
                .iter()
                .enumerate()
                .max_by_key(|(_i, &m)| {
                    m.score //+ info.killer_moves[ply % info.max_depth].score(m.mv) as i16
                        - self.penalty_hist_score(m.mv)
                })
                .unwrap();
            let m = *m;
            if m.mv.is_stack_move() {
                let hist_score = &mut self.stack_hist[m.mv.src_index()];
                if *hist_score < 10 {
                    *hist_score += 1;
                }
            } else if m.mv.place_piece().is_flat() {
                self.flat_attempts += 1;
            }
            self.moves.swap_remove(idx);
            m.mv
        } else {
            // Probably an all node, so search order doesn't really matter
            let x = self.moves.pop().unwrap();
            x.mv
        }
    }
    fn penalty_hist_score(&self, mv: GameMove) -> i16 {
        if mv.is_stack_move() {
            self.stack_hist[mv.src_index()]
        } else if mv.place_piece().is_flat() {
            self.flat_attempts * 2
        } else {
            0
        }
    }
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

    fn add_limit(&mut self, _limit: MoveLimits) {
        // self.limits.push(limit);
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
    all_flat: [u32; SIZE],
    flat_total: u32,
    white_wall: [u32; SIZE],
    black_wall: [u32; SIZE],
    all_cap: [u32; SIZE],
    cap_total: u32,
    white_wall_total: u32,
    black_wall_total: u32,
}

impl<const SIZE: usize> PlaceHistory<SIZE> {
    pub fn new() -> Self {
        Self {
            all_flat: [0; SIZE],
            white_wall: [0; SIZE],
            black_wall: [0; SIZE],
            all_cap: [0; SIZE],
            flat_total: 0,
            cap_total: 0,
            white_wall_total: 0,
            black_wall_total: 0,
        }
    }
    pub fn update(&mut self, depth: usize, mv: GameMove) {
        let value = (depth * depth) as u32;
        let idx = mv.src_index();
        match mv.place_piece() {
            Piece::BlackFlat | Piece::WhiteFlat => {
                self.all_flat[idx] += value;
                self.flat_total += value;
            }
            Piece::WhiteCap | Piece::BlackCap => {
                self.all_cap[idx] += value;
                self.cap_total += value;
            }
            Piece::WhiteWall => {
                self.white_wall[idx] += value;
                self.white_wall_total += value;
            }
            Piece::BlackWall => {
                self.black_wall[idx] += value;
                self.black_wall_total += value;
            }
        }
    }
    pub fn flat_score(&self, idx: usize) -> i16 {
        let raw_score = self.all_flat[idx];
        base_2_log(raw_score).saturating_sub(base_2_log(self.flat_total) / 2)
    }
    pub fn cap_score(&self, idx: usize) -> i16 {
        let raw_score = self.all_cap[idx];
        base_2_log(raw_score).saturating_sub(base_2_log(self.cap_total) / 2)
    }
    pub fn wall_score(&self, idx: usize, color: Color) -> i16 {
        let (raw_score, total) = match color {
            Color::White => (self.white_wall[idx], self.white_wall_total),
            Color::Black => (self.black_wall[idx], self.black_wall_total),
        };
        base_2_log(raw_score).saturating_sub(base_2_log(total) / 2)
    }
    pub fn score_place_move(&mut self, mv: GameMove) -> i16 {
        let idx = mv.src_index();
        let raw_score = {
            match mv.place_piece() {
                Piece::BlackFlat | Piece::WhiteFlat => self.all_flat[idx],
                Piece::WhiteCap | Piece::BlackCap => self.all_cap[idx],
                Piece::WhiteWall => self.white_wall[idx],
                Piece::BlackWall => self.black_wall[idx],
            }
        };
        // Hacky base 2 log
        base_2_log(raw_score)
    }
    pub fn debug(&self) {
        dbg!(&self.all_flat);
        dbg!(&self.white_wall);
        dbg!(&self.black_wall);
        dbg!(&self.all_cap);
    }
}

fn base_2_log(val: u32) -> i16 {
    (32 - val.leading_zeros()) as i16
}

#[derive(Clone)]
pub struct HistoryMoves {
    vec: Vec<u32>,
}

impl HistoryMoves {
    pub fn new(board_size: usize) -> Self {
        Self {
            vec: vec![1; board_size * board_size * 4],
        }
    }
    pub fn update(&mut self, depth: usize, mv: GameMove) {
        let value = depth as u32;
        self.vec[mv.direction() as usize + mv.src_index() * 4] += value * value;
    }
    pub fn square_data(&self, square: usize) -> &[u32] {
        &self.vec[square * 4..square * 4 + 4]
    }
    pub fn score(&self, mv: GameMove) -> i16 {
        // Hacky base 2 log
        (32 - self.vec[mv.direction() as usize + mv.src_index() * 4].leading_zeros()) as i16
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Board6;
    #[test]
    fn big_stack_order() {
        let tps = "21C,222222,2,x3/2,2,2S,12121S,x,2/2,2,1,1,1,1/x,1S,111112C,1,1,x/1,12112S,x4/x,2,x3,1 2 31";
        let board = Board6::try_from_tps(tps).unwrap();
        let mut moves = SmartMoveBuffer::new();
        generate_all_moves(&board, &mut moves);
        moves.score_stack_moves(&board);
        moves.moves.sort_by_key(|x| -x.score);
        assert!(moves.moves[0].score >= moves.moves.last().unwrap().score);
        let info = SearchInfo::new(1, 0);
        let order = (0..moves.moves.len()).map(|_| moves.get_best(0, &info));
        // let order: Vec<_> = moves.moves.into_iter().map(|x| *x.mv).collect();
        for m in order {
            println!("{}", m.to_ptn::<Board6>());
        }
    }
}
