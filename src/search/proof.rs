use super::*;
use crate::eval::Evaluate;
use crate::generate_all_moves;
use crate::{Board6, Position, RevGameMove};
use std::cmp::{max, min};
use std::collections::HashMap;

const INFINITY: u32 = 1_000_000;

#[derive(Clone)]
pub struct Child {
    bounds: Bounds,
    game_move: GameMove,
    zobrist: u64,
    best_child: usize,
}

impl Child {
    pub fn new(bounds: Bounds, game_move: GameMove, zobrist: u64) -> Self {
        Self {
            bounds,
            game_move,
            zobrist,
            best_child: usize::MAX,
        }
    }
    pub fn update_bounds(&mut self, bounds: Bounds, table: &mut HashMap<u64, Bounds>) {
        self.bounds = bounds;
        table.insert(self.zobrist, bounds);
    }
    pub fn phi(&self) -> u32 {
        self.bounds.phi
    }
    pub fn delta(&self) -> u32 {
        self.bounds.delta
    }
}

pub fn compute_bounds(children: &[Child]) -> Bounds {
    let mut out = Bounds {
        phi: INFINITY,
        delta: 0,
    };
    for ch in children.iter() {
        out.phi = min(out.phi, ch.bounds.delta);
        out.delta = out.delta.saturating_add(ch.bounds.phi);
    }
    out.delta = min(out.delta, INFINITY);
    return out;
}

#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    phi: u32,
    delta: u32,
}

impl Bounds {
    pub fn winning() -> Self {
        Bounds {
            phi: 0,
            delta: INFINITY,
        }
    }

    pub fn losing() -> Self {
        Bounds {
            phi: INFINITY,
            delta: 0,
        }
    }

    pub fn unity() -> Self {
        Bounds { phi: 1, delta: 1 }
    }

    pub fn infinity() -> Self {
        Bounds {
            phi: INFINITY,
            delta: INFINITY,
        }
    }

    pub fn root() -> Self {
        Bounds {
            phi: INFINITY / 2,
            delta: INFINITY / 2,
        }
    }
}

impl Default for Bounds {
    fn default() -> Self {
        Self::unity()
    }
}

struct TinueSearch {
    board: Board6,
    bounds_table: HashMap<u64, Bounds>,
    rev_moves: Vec<RevGameMove>,
    attacker: Color,
    nodes: usize,
}

impl TinueSearch {
    fn is_tinue(board: Board6) -> bool {
        let attacker = board.side_to_move();
        let mut search = Self {
            board,
            bounds_table: HashMap::new(),
            rev_moves: Vec::new(),
            attacker,
            nodes: 0,
        };
        let mut root = Child::new(Bounds::root(), GameMove::null_move(), search.board.hash());
        search.MID(&mut root, 0);
        dbg!(search.nodes);
        if root.delta() == INFINITY {
            true
        } else {
            false
        }
    }
    fn MID(&mut self, child: &mut Child, depth: usize) {
        self.nodes += 1;
        if child.game_move != GameMove::null_move() {
            let rev = self.board.do_move(child.game_move);
            self.rev_moves.push(rev);
        }
        assert_eq!(child.zobrist, self.board.hash());
        let side_to_move = self.board.side_to_move();
        let attacker = side_to_move == self.attacker;
        let moves = if attacker {
            match tinue_evaluate(&mut self.board) {
                AttackerOutcome::HasRoad(_m) => {
                    let eval = Bounds::winning();
                    child.update_bounds(eval, &mut self.bounds_table);
                    self.undo_move();
                    return;
                }
                AttackerOutcome::TakThreats(vec) => vec,
                AttackerOutcome::NoTakThreats => {
                    let eval = Bounds::losing();
                    child.update_bounds(eval, &mut self.bounds_table);
                    self.undo_move();
                    return;
                }
            }
        } else {
            let mut moves = Vec::new();
            generate_all_moves(&mut self.board, &mut moves);
            moves
                .into_iter()
                .filter(|m| !m.is_place_move() || m.place_piece().is_blocker())
                .collect()
        };
        assert!(!moves.is_empty());

        if child.game_move == GameMove::null_move() {
            let debug_vec: Vec<_> = moves.iter().map(|m| m.to_ptn()).collect();
            dbg!(&debug_vec); // Root moves?
        }
        // generate_all_moves(&mut self.board, &mut moves);
        let mut child_pns: Vec<_> = moves.into_iter().map(|m| self.init_pns(m)).collect();
        loop {
            let limit = compute_bounds(&child_pns);
            if child.phi() <= limit.phi || child.delta() <= limit.delta {
                child.update_bounds(limit, &mut self.bounds_table);
                self.undo_move();
                return;
            }
            let (best_idx, second_best_bounds) = Self::select_child(&child_pns);
            let best_child = &mut child_pns[best_idx];
            // println!("Depth: {} Move: {}", depth, best_child.game_move.to_ptn());
            let updated_bounds = Bounds {
                phi: child.delta() + best_child.phi() - limit.delta,
                delta: min(child.phi(), second_best_bounds.delta + 1),
            };
            best_child.update_bounds(updated_bounds, &mut self.bounds_table);
            self.MID(best_child, depth + 1);
        }
    }
    fn select_child(children: &[Child]) -> (usize, Bounds) {
        let mut c_best_idx = 0;
        let mut best = children[c_best_idx].bounds;
        let mut second_best = Bounds::infinity();
        for (idx, child) in children.iter().enumerate().skip(1) {
            if child.bounds.delta < best.delta {
                c_best_idx = idx;
                second_best = best;
                best = child.bounds;
            } else if child.bounds.delta < second_best.delta {
                second_best = child.bounds;
            }
            if child.bounds.phi == INFINITY {
                return (c_best_idx, second_best);
            }
        }
        (c_best_idx, second_best)
    }
    fn init_pns(&mut self, game_move: GameMove) -> Child {
        let rev = self.board.do_move(game_move);
        let hash = self.board.hash();
        let bounds = self.bounds_table.entry(hash).or_default();
        let child = Child::new(bounds.clone(), game_move, hash);
        self.board.reverse_move(rev);
        child
    }
    fn undo_move(&mut self) -> Option<()> {
        let m = self.rev_moves.pop()?;
        self.board.reverse_move(m);
        Some(())
    }
}

enum AttackerOutcome {
    HasRoad(GameMove),
    TakThreats(Vec<GameMove>),
    NoTakThreats,
}

fn tinue_evaluate(pos: &mut Board6) -> AttackerOutcome {
    let mut moves = Vec::new();
    if let Some(m) = pos.can_make_road(&mut moves) {
        return AttackerOutcome::HasRoad(m);
    }
    // Moves contains all stack moves due to the can_make_road call
    crate::move_gen::generate_aggressive_place_moves(pos, &mut moves);
    let tak_threats = pos.get_tak_threats(&moves);
    if tak_threats.is_empty() {
        AttackerOutcome::NoTakThreats
    } else {
        AttackerOutcome::TakThreats(tak_threats)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn simple() {
        let s = "x2,2,x2,1/x5,1/x,2,x,1,1,1/x,2,x2,1,x/x,2C,x4/x,2,x4 2 6";
        let board = Board6::try_from_tps(s).unwrap();
        dbg!(&board);
        assert!(TinueSearch::is_tinue(board));
    }
    #[test]
    fn simple2() {
        let s = "1,1,1,1,1112C,1/x,121C,x,1,2,1/1,2,x,12,1S,x/x,2,2,1221S,x,2/x3,121,x2/2,2,2,1,2,x 1 25";
        let s2 =
            "1,1,1,1,1112C,1/x,x,x,1,2,1/1,2,x,12,1S,x/x,2,2,1221S,x,2/x3,121,x2/2,2,2,1,2,x 1 25";
        let board = Board6::try_from_tps(s).unwrap();
        assert!(TinueSearch::is_tinue(board));

        assert!(!TinueSearch::is_tinue(Board6::try_from_tps(s2).unwrap()));
    }
}
