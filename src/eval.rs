use super::{Bitboard, Board6, Piece, Stack};
use crate::move_gen::generate_all_stack_moves;
use crate::{Bitboard6, GameMove, RevGameMove};
use board_game_traits::{Color, GameResult, Position};

pub trait Evaluate: Position<Move = GameMove, ReverseMove = RevGameMove> {
    fn evaluate(&self) -> i32;
    fn hash(&self) -> u64;
    fn legal_move(&self, game_move: GameMove) -> bool;
    fn ply(&self) -> usize;
    fn null_move(&mut self);
    fn rev_null_move(&mut self);
    fn get_tak_threats(
        &mut self,
        legal_moves: &Vec<GameMove>,
        hint: Option<&[GameMove]>,
    ) -> Vec<GameMove>;
    fn can_make_road(
        &mut self,
        storage: &mut Vec<GameMove>,
        hint: Option<&[GameMove]>,
    ) -> Option<GameMove>;
}

fn win_color(res: GameResult) -> Option<Color> {
    match res {
        GameResult::WhiteWin => Some(Color::White),
        GameResult::BlackWin => Some(Color::Black),
        GameResult::Draw => None,
    }
}

pub const WIN_SCORE: i32 = 10_000;
pub const LOSE_SCORE: i32 = -1 * WIN_SCORE;

#[rustfmt::skip]
const LOCATION_WEIGHT: [i32; 36] = [
    00, 05, 05, 05, 05, 00,
    05, 10, 15, 15, 10, 05,
    05, 15, 20, 20, 15, 05,
    05, 15, 20, 20, 15, 05,
    05, 10, 15, 15, 10, 05,
    00, 05, 05, 05, 05, 00, 
];

fn piece_weight(p: Piece) -> i32 {
    match p {
        Piece::WhiteFlat | Piece::BlackFlat => 100,
        Piece::WhiteWall | Piece::BlackWall => 40,
        Piece::WhiteCap | Piece::BlackCap => 80,
    }
}

fn stack_top_multiplier(p: Piece) -> (i32, i32) {
    match p {
        Piece::WhiteFlat | Piece::BlackFlat => (-50, 60),
        Piece::WhiteWall | Piece::BlackWall => (-30, 70),
        Piece::WhiteCap | Piece::BlackCap => (-20, 90),
    }
}

impl Evaluate for Board6 {
    fn evaluate(&self) -> i32 {
        let mut score = 0;
        for (idx, stack) in self.board.iter().enumerate() {
            if stack.len() == 1 {
                let top = *stack.last().unwrap();
                let pw = piece_weight(top) + LOCATION_WEIGHT[idx];
                if let Color::White = top.owner() {
                    score += pw;
                } else {
                    score -= pw;
                }
            } else if stack.len() > 1 {
                let top = *stack.last().unwrap();
                let pw = piece_weight(top) + LOCATION_WEIGHT[idx];
                let (captive, friendly) = captive_friendly(&stack, top);
                let (c_mul, f_mul) = stack_top_multiplier(top);
                let stack_score = captive * c_mul + friendly * f_mul + pw;
                if let Color::White = top.owner() {
                    score += stack_score;
                } else {
                    score -= stack_score;
                }
            }
        }
        let white_connectivity = (self.bits.white.adjacent() & self.bits.white).pop_count();
        let black_connectivity = (self.bits.black.adjacent() & self.bits.black).pop_count();
        score += white_connectivity as i32 * 20;
        score -= black_connectivity as i32 * 20;
        if let Color::White = self.side_to_move() {
            score - 100
        } else {
            -1 * score
        }
    }
    fn hash(&self) -> u64 {
        self.zobrist()
    }
    fn legal_move(&self, game_move: GameMove) -> bool {
        let mut vec = Vec::new();
        self.generate_moves(&mut vec);
        vec.into_iter().find(|&m| m == game_move).is_some()
    }
    fn ply(&self) -> usize {
        match self.side_to_move() {
            Color::White => self.move_num() * 2,
            Color::Black => self.move_num() * 2 + 1,
        }
    }
    fn null_move(&mut self) {
        self.swap_active_player();
    }
    fn rev_null_move(&mut self) {
        self.swap_active_player();
    }
    fn get_tak_threats(
        &mut self,
        legal_moves: &Vec<GameMove>,
        hint: Option<&[GameMove]>,
    ) -> Vec<GameMove> {
        let mut tak_threats = Vec::new();
        let mut stack_moves = Vec::new();
        for m in legal_moves.iter().copied() {
            let rev = self.do_move(m);
            self.null_move();
            if self.can_make_road(&mut stack_moves, hint).is_some() {
                tak_threats.push(m);
            }
            self.rev_null_move();
            self.reverse_move(rev);
            stack_moves.clear();
        }
        tak_threats
    }
    fn can_make_road(
        &mut self,
        storage: &mut Vec<GameMove>,
        hint: Option<&[GameMove]>,
    ) -> Option<GameMove> {
        let player = self.active_player();
        let road_pieces = self.bits.road_pieces(player);
        let mut attempt = road_pieces.adjacent() & self.bits.empty();
        // Check flat placements
        while attempt != Bitboard6::ZERO {
            let lowest = attempt.pop_lowest();
            let check = lowest | road_pieces;
            if check.check_road() {
                return Some(GameMove::from_placement(
                    Piece::flat(player),
                    lowest.lowest_index(),
                ));
            }
        }
        // Check stack movements
        generate_all_stack_moves(self, storage);
        if let Some(suggestions) = hint {
            for m in suggestions.iter().copied() {
                if storage.contains(&m) {
                    if self.road_stack_throw(road_pieces, m) {
                        return Some(m);
                    }
                    // let rev = self.do_move(m);
                    // let road = self.road(player);
                    // self.reverse_move(rev);
                    // if road {
                    //     return Some(m);
                    // }
                }
            }
        }
        // Todo optimize this
        for m in storage.iter().copied() {
            if self.road_stack_throw(road_pieces, m) {
                return Some(m);
            }
            // let rev = self.do_move(m);
            // let road = self.road(player);
            // self.reverse_move(rev);
            // if road {
            //     return Some(m);
            // }
        }
        // for m in stack_moves {
        //     let mut bits = Bitboard6::ZERO;
        //     let mut offset = m.number() as usize;
        //     let source = board.index(m.src_index());
        //     source.from_top(offset)
        // }
        None
    }
}

fn captive_friendly(stack: &Stack, top: Piece) -> (i32, i32) {
    let mut captive = 0;
    let mut friendly = 0;
    let controller = top.owner();
    for piece in stack.iter().rev().skip(1) {
        if piece.owner() == controller {
            friendly += 1;
        } else {
            captive += 1;
        }
    }
    (captive, friendly)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::generate_all_moves;
    #[test]
    fn check_tak_threats() {
        let s = "x2,2,x2,1/x5,1/x,2,x,1,1,1/x,2,x2,1,x/x,2C,x4/x,2,x4 2 6";
        let mut board = crate::Board6::try_from_tps(s).unwrap();
        let mut moves = Vec::new();
        generate_all_moves(&mut board, &mut moves);
        let tak_threats = board.get_tak_threats(&moves, None);
        for m in tak_threats.iter() {
            dbg!(m.to_ptn());
        }
        assert_eq!(tak_threats.len(), 5);
    }
}
