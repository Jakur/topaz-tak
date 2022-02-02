use super::*;
use crate::search::SearchInfo;

pub struct SmartMoveBuffer {
    moves: Vec<ScoredMove>,
    limits: Vec<MoveLimits>,
    queries: usize,
}

impl SmartMoveBuffer {
    pub fn new() -> Self {
        Self {
            moves: Vec::new(),
            limits: Vec::new(),
            queries: 0,
        }
    }
    pub fn score_stack_moves<T: TakBoard>(&mut self, board: &T) {
        let active_side = board.side_to_move();
        let mut stack_idx = usize::MAX;
        // Moves should be grouped by src index due to generator impl
        let mut stack_data = [Piece::WhiteFlat; 8];
        for x in self.moves.iter_mut().filter(|x| x.mv.is_stack_move()) {
            let mut score = 0;
            let src_idx = x.mv.src_index();
            // TODO figure out if this is right
            if src_idx != stack_idx {
                stack_idx = src_idx;
                // Update stack data
                let number = x.mv.number() as usize;
                for i in 0..number {
                    stack_data[i] = board.index(src_idx).from_top(i).unwrap();
                }
            }
            let mut offset = 0;
            for step in x.mv.quantity_iter(T::SIZE) {
                offset += step.quantity as usize;
                let covered = board.index(step.index).last();
                let covering = stack_data[offset];
                if let Some(piece) = covered {
                    if piece.owner() == active_side {
                        score -= 1;
                    } else {
                        score += 1;
                    }
                }
                if covering.owner() == active_side {
                    score += 1;
                }
            }
            x.score += score;
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
                .max_by_key(|(_i, &m)| m.score + info.killer_moves[depth].score(m.mv) as i16)
                .unwrap();
            let m = *m;
            self.moves.swap_remove(idx);
            m.mv
        } else {
            // Probably an all node, so search order doesn't really matter
            let x = self.moves.pop().unwrap();
            x.mv
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
            self.moves.push(ScoredMove::new(mv, mv.number() as i16));
        }
    }

    fn add_limit(&mut self, limit: MoveLimits) {
        self.limits.push(limit);
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
