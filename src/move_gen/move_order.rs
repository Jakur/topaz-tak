use super::*;
use crate::search::SearchInfo;

pub struct SmartMoveBuffer {
    moves: Vec<ScoredMove>,
    stack_hist: Vec<i16>,
    queries: usize,
}

impl SmartMoveBuffer {
    pub fn new() -> Self {
        Self {
            moves: Vec::new(),
            stack_hist: vec![0; 36],
            queries: 0,
        }
    }
    pub fn score_stack_moves<T: TakBoard>(&mut self, board: &T, last_capture: Option<RevGameMove>) {
        // let DEBUG: &'static str = "5b2>221";
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
                let covered = board.index(step.index).last();
                let covering = stack_data[offset];
                // println!("{}", &x.mv.to_ptn::<T>());
                // if &x.mv.to_ptn::<T>() == DEBUG {
                //     println!("Covered: {:?}", covered);
                //     println!("Covering {:?}", covering);
                // }
                if let Some(piece) = covered {
                    if piece.owner() == active_side {
                        score -= 1;
                    } else {
                        score += 1;
                    }
                }
                if covering.owner() == active_side {
                    // if let Some(capture) = last_capture {
                    //     if step.index == capture.dest_sq {
                    //         score += 3;
                    //     }
                    // }
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
            if let Some(piece) = src_stack.last() {
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
    pub fn score_pv_move(&mut self, pv_move: GameMove) {
        if let Some(found) = self.moves.iter_mut().find(|m| m.mv == pv_move) {
            found.score += 100;
        }
    }
    pub fn score_tak_threats(&mut self, tak_threats: &[GameMove]) {
        for m in self.moves.iter_mut() {
            if tak_threats.contains(&m.mv) {
                m.score += 50;
            }
        }
    }
    pub fn get_best(&mut self, depth: usize, info: &SearchInfo) -> GameMove {
        if self.queries <= 10 {
            let (idx, m) = self
                .moves
                .iter()
                .enumerate()
                .max_by_key(|(_i, &m)| {
                    m.score + info.killer_moves[depth].score(m.mv) as i16
                        - self.stack_hist_score(m.mv)
                })
                .unwrap();
            let m = *m;
            if m.mv.is_stack_move() {
                let hist_score = &mut self.stack_hist[m.mv.src_index()];
                if *hist_score < 10 {
                    *hist_score += 1;
                }
            }
            self.moves.swap_remove(idx);
            m.mv
        } else {
            // Probably an all node, so search order doesn't really matter
            let x = self.moves.pop().unwrap();
            x.mv
        }
    }
    fn stack_hist_score(&self, mv: GameMove) -> i16 {
        if mv.is_stack_move() {
            self.stack_hist[mv.src_index()]
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
            self.moves.push(ScoredMove::new(mv, 3));
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
        moves.score_stack_moves(&board, None);
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
