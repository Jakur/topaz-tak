use super::{Bitboard, Board6, Piece, Stack};
use crate::move_gen::generate_all_stack_moves;
use crate::{Bitboard6, GameMove, RevGameMove};
use board_game_traits::{Color, GameResult, Position};

pub trait Evaluator {
    type Game: TakBoard;
    fn evaluate(&self, game: &Self::Game, depth: usize) -> i32;
}

pub struct Evaluator6 {}

impl Evaluator6 {
    const TEMPO_OFFSET: i32 = 50;
    const CONNECTIVITY: i32 = 20;
    const fn piece_weight(p: Piece) -> i32 {
        match p {
            Piece::WhiteFlat | Piece::BlackFlat => 100,
            Piece::WhiteWall | Piece::BlackWall => 40,
            Piece::WhiteCap | Piece::BlackCap => 80,
        }
    }
    const fn stack_top_multiplier(p: Piece) -> (i32, i32) {
        match p {
            Piece::WhiteFlat | Piece::BlackFlat => (-50, 60),
            Piece::WhiteWall | Piece::BlackWall => (-30, 70),
            Piece::WhiteCap | Piece::BlackCap => (-20, 90),
        }
    }
}

impl Evaluator for Evaluator6 {
    type Game = Board6;
    fn evaluate(&self, game: &Self::Game, depth: usize) -> i32 {
        let mut score = 0;
        for (idx, stack) in game.board.iter().enumerate() {
            if stack.len() == 1 {
                let top = *stack.last().unwrap();
                let pw = Self::piece_weight(top) + LOCATION_WEIGHT[idx];
                if let Color::White = top.owner() {
                    score += pw;
                } else {
                    score -= pw;
                }
            } else if stack.len() > 1 {
                let top = *stack.last().unwrap();
                let pw = Self::piece_weight(top) + LOCATION_WEIGHT[idx];
                let (captive, friendly) = captive_friendly(&stack, top);
                let (c_mul, f_mul) = Self::stack_top_multiplier(top);
                let stack_score = captive * c_mul + friendly * f_mul + pw;
                if let Color::White = top.owner() {
                    score += stack_score;
                } else {
                    score -= stack_score;
                }
            }
        }
        let white_connectivity = (game.bits.white.adjacent() & game.bits.white).pop_count();
        let black_connectivity = (game.bits.black.adjacent() & game.bits.black).pop_count();
        score += white_connectivity as i32 * Self::CONNECTIVITY;
        score -= black_connectivity as i32 * Self::CONNECTIVITY;
        if let Color::White = game.side_to_move() {
            if depth % 2 == 0 {
                score
            } else {
                score - Self::TEMPO_OFFSET
            }
        } else {
            if depth % 2 == 0 {
                -1 * score
            } else {
                -1 * score + Self::TEMPO_OFFSET
            }
        }
    }
}

pub trait TakBoard: Position<Move = GameMove, ReverseMove = RevGameMove> {
    type Bits: Bitboard;
    const SIZE: usize;
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

pub struct Weights6 {
    location: [i32; 36],
    connectivity: i32,
    tempo_offset: i32,
    piece: [i32; 3],
    stack_top: [i32; 6],
}

impl Evaluator for Weights6 {
    type Game = Board6;
    fn evaluate(&self, game: &Self::Game, depth: usize) -> i32 {
        let mut score = 0;
        for (idx, stack) in game.board.iter().enumerate() {
            if stack.len() == 1 {
                let top = *stack.last().unwrap();
                let pw = self.piece_weight(top) + self.location[idx];
                if let Color::White = top.owner() {
                    score += pw;
                } else {
                    score -= pw;
                }
            } else if stack.len() > 1 {
                let top = *stack.last().unwrap();
                let pw = self.piece_weight(top) + self.location[idx];
                let (captive, friendly) = captive_friendly(&stack, top);
                let (c_mul, f_mul) = self.stack_top_multiplier(top);
                let stack_score = captive * c_mul + friendly * f_mul + pw;
                if let Color::White = top.owner() {
                    score += stack_score;
                } else {
                    score -= stack_score;
                }
            }
        }
        let white_connectivity = (game.bits.white.adjacent() & game.bits.white).pop_count();
        let black_connectivity = (game.bits.black.adjacent() & game.bits.black).pop_count();
        score += white_connectivity as i32 * self.connectivity;
        score -= black_connectivity as i32 * self.connectivity;
        if let Color::White = game.side_to_move() {
            if depth % 2 == 0 {
                score
            } else {
                score - self.tempo_offset
            }
        } else {
            if depth % 2 == 0 {
                -1 * score
            } else {
                -1 * score + self.tempo_offset
            }
        }
    }
}

impl Weights6 {
    pub fn new(
        location: [i32; 36],
        connectivity: i32,
        tempo_offset: i32,
        piece: [i32; 3],
        stack_top: [i32; 6],
    ) -> Self {
        Self {
            location,
            connectivity,
            tempo_offset,
            piece,
            stack_top,
        }
    }
    fn piece_weight(&self, p: Piece) -> i32 {
        match p {
            Piece::WhiteFlat | Piece::BlackFlat => self.piece[0],
            Piece::WhiteWall | Piece::BlackWall => self.piece[1],
            Piece::WhiteCap | Piece::BlackCap => self.piece[2],
        }
    }
    fn stack_top_multiplier(&self, p: Piece) -> (i32, i32) {
        match p {
            Piece::WhiteFlat | Piece::BlackFlat => (self.stack_top[0], self.stack_top[1]),
            Piece::WhiteWall | Piece::BlackWall => (self.stack_top[2], self.stack_top[3]),
            Piece::WhiteCap | Piece::BlackCap => (self.stack_top[4], self.stack_top[5]),
        }
    }
}

impl Default for Weights6 {
    fn default() -> Self {
        let piece_arr = [
            Evaluator6::piece_weight(Piece::WhiteFlat),
            Evaluator6::piece_weight(Piece::WhiteWall),
            Evaluator6::piece_weight(Piece::WhiteCap),
        ];
        let st1 = Evaluator6::stack_top_multiplier(Piece::WhiteFlat);
        let st2 = Evaluator6::stack_top_multiplier(Piece::WhiteWall);
        let st3 = Evaluator6::stack_top_multiplier(Piece::WhiteCap);
        Self::new(
            LOCATION_WEIGHT,
            Evaluator6::CONNECTIVITY,
            Evaluator6::TEMPO_OFFSET,
            piece_arr,
            [st1.0, st1.1, st2.0, st2.1, st3.0, st3.1],
        )
    }
}

// corner, offcorner, edge, edge, offcorner, corner,
// offcorner, kosumi, ctouch, ctouch, kosumi, corner,
// edge, ctouch, center, center, ctouch, edge,
// edge, ctouch, center, center, ctouch, edge,
// offcorner, kosumi, ctouch, ctouch, kosumi, corner,
// corner, offcorner, edge, edge, offcorner, corner,

#[rustfmt::skip]
const LOCATION_WEIGHT: [i32; 36] = [
    00, 05, 05, 05, 05, 00,
    05, 10, 15, 15, 10, 05,
    05, 15, 20, 20, 15, 05,
    05, 15, 20, 20, 15, 05,
    05, 10, 15, 15, 10, 05,
    00, 05, 05, 05, 05, 00, 
];

impl TakBoard for Board6 {
    type Bits = Bitboard6;
    const SIZE: usize = 6;
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
        // Check placement
        let place_road = find_placement_road(player, road_pieces, self.bits.empty());
        if place_road.is_some() {
            return place_road;
        }
        // Check stack movements
        generate_all_stack_moves(self, storage);
        if let Some(suggestions) = hint {
            for m in suggestions.iter().copied() {
                if storage.contains(&m) {
                    if self.road_stack_throw(road_pieces, m) {
                        return Some(m);
                    }
                }
            }
        }
        for m in storage.iter().copied() {
            if self.road_stack_throw(road_pieces, m) {
                return Some(m);
            }
        }
        None
    }
}

pub fn find_placement_road(
    player: Color,
    road_pieces: Bitboard6,
    empty: Bitboard6,
) -> Option<GameMove> {
    let valid = road_pieces.critical_squares() & empty;
    if valid != Bitboard6::ZERO {
        let sq_index = valid.lowest_index();
        Some(GameMove::from_placement(Piece::flat(player), sq_index))
    } else {
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
