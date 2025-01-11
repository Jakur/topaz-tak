use std::marker::PhantomData;

use super::*;
use crate::search::SearchInfo;

// [6, 5, 2, 7, 3, 3, 2, -2, 10, 1, -1, 0, -2, -1, -2, 4, -2, 1]
#[derive(Debug, Clone)]
pub struct MoveBufferHyper {
    covered_self: i16,
    covered_other: i16,
    top_cover_self: i16,
    top_cover_other: i16,
    retain_control: i16,
    lose_control: i16,
    use_cap: i16,
    crush: i16,
    flat_score: i16,
    wall_score: i16,
    cap_score: i16,
    wall_near_stack: i16,
    cap_near_stack: i16,
    flat_enemy: i16,
    wall_enemy: i16,
    cap_enemy: i16,
    lonely_cap: i16,
    lonely_flat: i16,
}

impl MoveBufferHyper {
    pub fn new_from_slice(arr: &[i16]) -> Self {
        assert_eq!(arr.len(), 18);
        Self::new(
            arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
            arr[10], arr[11], arr[12], arr[13], arr[14], arr[15], arr[16], arr[17],
        )
    }
    fn new(
        covered_self: i16,
        covered_other: i16,
        top_cover_self: i16,
        top_cover_other: i16,
        retain_control: i16,
        lose_control: i16,
        use_cap: i16,
        crush: i16,
        flat_score: i16,
        wall_score: i16,
        cap_score: i16,
        wall_near_stack: i16,
        cap_near_stack: i16,
        flat_enemy: i16,
        wall_enemy: i16,
        cap_enemy: i16,
        lonely_cap: i16,
        lonely_flat: i16,
    ) -> Self {
        Self {
            covered_self,
            covered_other,
            top_cover_self,
            top_cover_other,
            retain_control,
            lose_control,
            use_cap,
            crush,
            flat_score,
            wall_score,
            cap_score,
            wall_near_stack,
            cap_near_stack,
            flat_enemy,
            wall_enemy,
            cap_enemy,
            lonely_cap,
            lonely_flat,
        }
    }
}

pub struct TestMoveBuffer {
    moves: Vec<ScoredMove>,
    // stack_hist: Vec<i16>,
    queries: usize,
    // flat_attempts: i16,
    hyper: MoveBufferHyper,
}

impl TestMoveBuffer {
    pub fn new(buffer_size: usize, hyper: MoveBufferHyper) -> Self {
        Self {
            moves: Vec::new(),
            // stack_hist: vec![0; buffer_size],
            queries: 0,
            // flat_attempts: 0,
            hyper,
        }
    }
    pub fn drop_below_score(&mut self, threshold: i16) {
        self.moves.retain(|x| x.score >= threshold);
    }
    pub fn score_stack_moves<T: TakBoard>(&mut self, board: &T) {
        let active_side = board.side_to_move();
        let mut stack_idx = usize::MAX;
        // Moves should be grouped by src index due to generator impl
        let mut stack_data = [Piece::WhiteFlat; 8]; // Todo T::SIZE + 1 when rust figures out this valid
        for x in self.moves.iter_mut().filter(|x| x.mv.is_stack_move()) {
            let mut score = 0;
            let src_idx = x.mv.src_index();
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
            let mut offset = 0;
            for step in x.mv.quantity_iter(T::SIZE) {
                debug_assert!(step.quantity > 0);
                offset += step.quantity as usize;
                let covered = board.index(step.index).top();
                let covering = stack_data[offset];
                if let Some(piece) = covered {
                    if piece.owner() == active_side {
                        score -= self.hyper.covered_self;
                    } else {
                        score += self.hyper.covered_other;
                    }
                }
                if covering.owner() == active_side {
                    score += self.hyper.top_cover_self;
                } else {
                    score -= self.hyper.top_cover_other;
                }
            }
            let src_stack = board.index(src_idx);
            if let Some(piece) = src_stack.from_top(x.mv.number() as usize) {
                if piece.owner() == active_side {
                    score += self.hyper.retain_control;
                } else {
                    score -= self.hyper.lose_control;
                }
            }
            if let Some(piece) = src_stack.top() {
                if piece.is_cap() {
                    score += self.hyper.use_cap;
                    if x.mv.crush() {
                        score += self.hyper.crush;
                    }
                }
            }
            x.score += score;
            // if &x.mv.to_ptn::<T>() == DEBUG {
            //     dbg!(x.score);
            // }
        }
    }
    pub fn gen_score_place_moves<T: TakBoard>(&mut self, board: &T, place_hist: &PlaceHistory<49>) {
        let side = board.side_to_move();
        for idx in board.empty_tiles() {
            let mut flat_score = self.hyper.flat_score + place_hist.flat_score(idx);
            let mut wall_score = self.hyper.wall_score + place_hist.wall_score(idx, side);
            let mut cap_score = self.hyper.cap_score + place_hist.cap_score(idx);
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
                    wall_score += self.hyper.wall_near_stack;
                    cap_score += self.hyper.cap_near_stack;
                }
            }
            if enemies.pop_count() >= 3 {
                flat_score -= self.hyper.flat_enemy;
                cap_score += self.hyper.cap_enemy;
                wall_score += self.hyper.wall_enemy;
            }
            let friends = neighbors & board.bits().all_pieces(side);
            if friends.is_zero() {
                cap_score -= self.hyper.lonely_cap;
                flat_score -= self.hyper.lonely_flat;
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
    pub fn get_best<T: TakBoard>(&mut self, last_move: Option<RevGameMove>, board: &T) -> GameMove {
        if self.queries <= 16 {
            self.queries += 1;
            let (idx, m) = self
                .moves
                .iter()
                .enumerate()
                .max_by_key(|(_i, &m)| {
                    m.score // + info.killer_moves[ply % info.max_depth].score(m.mv) as i16
                            // - self.penalty_hist_score(m.mv)
                })
                .unwrap();
            let m = *m;
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

pub struct SmartMoveBuffer {
    moves: Vec<ScoredMove>,
    // stack_hist: Vec<i16>,
    queries: usize,
    // flat_attempts: i16,
}

impl SmartMoveBuffer {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            moves: Vec::new(),
            // stack_hist: vec![0; buffer_size],
            queries: 0,
            // flat_attempts: 0,
        }
    }
    pub fn drop_below_score(&mut self, threshold: i16) {
        self.moves.retain(|x| x.score >= threshold);
    }
    pub fn score_stack_moves<T: TakBoard>(&mut self, info: &SearchInfo, board: &T) {
        let active_side = board.side_to_move();
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
            let top_piece = src_stack.top().unwrap();
            if x.mv.number() == 1 && score == 2 && top_piece.is_flat() {
                x.score = -100;
            } else {
                if let Some(piece) = src_stack.from_top(x.mv.number() as usize) {
                    if piece.owner() == active_side {
                        score += 2;
                    } else {
                        score -= 2;
                    }
                }
                if top_piece.is_cap() {
                    score += 1;
                    if x.mv.crush() {
                        score += 1;
                    }
                }
                x.score += score;
            }
            // if &x.mv.to_ptn::<T>() == DEBUG {
            //     dbg!(x.score);
            // }
        }
    }
    pub fn gen_score_place_moves<T: TakBoard>(&mut self, board: &T, place_hist: &PlaceHistory<49>) {
        let side = board.side_to_move();
        // [2, 3, 2, 6, 2, 3, 2, 0, 10, 1, 0, 2, 0, -1, 3, 0, 0, 2]
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
    pub fn get_best(&mut self, ply: usize, info: &SearchInfo) -> GameMove {
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
                    // m.score // + info.killer_moves[ply % info.max_depth].score(m.mv) as i16
                    // - self.penalty_hist_score(m.mv)
                })
                .unwrap();
            let m = *m;
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

impl MoveBuffer for TestMoveBuffer {
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
        let sz = SIZE as u32;
        Self {
            all_flat: [1; SIZE],
            white_wall: [1; SIZE],
            black_wall: [1; SIZE],
            all_cap: [1; SIZE],
            flat_total: sz,
            cap_total: sz,
            white_wall_total: sz,
            black_wall_total: sz,
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
        (raw_score.ilog2() as i16).saturating_sub((self.flat_total.ilog2() as i16) / 2)
    }
    pub fn cap_score(&self, idx: usize) -> i16 {
        let raw_score = self.all_cap[idx];
        (raw_score.ilog2() as i16).saturating_sub((self.cap_total.ilog2() as i16) / 2)
    }
    pub fn wall_score(&self, idx: usize, color: Color) -> i16 {
        let (raw_score, total) = match color {
            Color::White => (self.white_wall[idx], self.white_wall_total),
            Color::Black => (self.black_wall[idx], self.black_wall_total),
        };
        (raw_score.ilog2() as i16).saturating_sub((total.ilog2() as i16) / 2)
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
        raw_score.ilog2() as i16
    }
    pub fn debug(&self) {
        dbg!(&self.all_flat);
        dbg!(&self.white_wall);
        dbg!(&self.black_wall);
        dbg!(&self.all_cap);
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::Board6;
    #[test]
    fn big_stack_order() {
        let tps = "21C,222222,2,x3/2,2,2S,12121S,x,2/2,2,1,1,1,1/x,1S,111112C,1,1,x/1,12112S,x4/x,2,x3,1 2 31";
        let board = Board6::try_from_tps(tps).unwrap();
        let mut moves = SmartMoveBuffer::new(36);
        generate_all_moves(&board, &mut moves);
        moves.score_stack_moves(&board);
        moves.moves.sort_by_key(|x| -x.score);
        assert!(moves.moves[0].score >= moves.moves.last().unwrap().score);
        let order = (0..moves.moves.len()).map(|_| moves.get_best(None, &board));
        // let order: Vec<_> = moves.moves.into_iter().map(|x| *x.mv).collect();
        for m in order {
            println!("{}", m.to_ptn::<Board6>());
        }
    }
}
