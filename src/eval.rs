use super::{Board6, Piece};
use board_game_traits::{Color, GameResult, Position};

trait Evaluate {
    fn evaluate(&self) -> i32;
}

fn win_color(res: GameResult) -> Option<Color> {
    match res {
        GameResult::WhiteWin => Some(Color::White),
        GameResult::BlackWin => Some(Color::Black),
        GameResult::Draw => None,
    }
}

const WIN_SCORE: i32 = 10_000;
const LOSE_SCORE: i32 = -1 * WIN_SCORE;

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
        if let Some(res) = self.game_result() {
            if let Some(color) = win_color(res) {
                if color == self.side_to_move() {
                    return WIN_SCORE;
                } else {
                    return LOSE_SCORE;
                }
            } else {
                return 0;
            }
        }
        let mut score = 0;
        for (idx, stack) in self.board.iter().enumerate() {
            if stack.len() == 1 {
                let pw = piece_weight(stack[0]) + LOCATION_WEIGHT[idx];
                if let Color::White = stack[0].owner() {
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
        if let Color::White = self.side_to_move() {
            score
        } else {
            -1 * score
        }
    }
}

fn captive_friendly(stack: &Vec<Piece>, top: Piece) -> (i32, i32) {
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
