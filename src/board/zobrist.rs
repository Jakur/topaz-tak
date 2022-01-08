use crate::board::Board6;
use crate::Piece;
use crate::TakBoard;
use board_game_traits::Color;
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub const TABLE: ZobristTable = ZobristTable::new();

const TOPS: usize = (Board6::SIZE + 1) * (Board6::SIZE * Board6::SIZE + 1);
const STACKS: usize = (Board6::FLATS * 2 + 2) * (Board6::SIZE * Board6::SIZE);
/// A Zobrist Table holding enough random values to support a size 6 board
pub struct ZobristTable {
    white_to_move: u64,
    black_to_move: u64,
    table: [u64; TOPS + STACKS * 2],
}

impl ZobristTable {
    const fn new() -> Self {
        const TABLE_DATA: [u64; TOPS + STACKS * 2] = include!("zobrist.table");
        Self {
            table: TABLE_DATA,
            white_to_move: 0x1C1E9C11991A743F,
            black_to_move: 0xD272922F20FEAB1E,
        }
    }
    pub fn color_hash(&self, color: Color) -> u64 {
        match color {
            Color::White => self.white_to_move,
            Color::Black => self.black_to_move,
        }
    }
    pub fn top_hash(&self, piece: Piece, index: usize) -> u64 {
        self.table[piece as usize + index * 7]
    }
    pub fn stack_hash(&self, piece: Piece, sq_index: usize, stack_index: usize) -> u64 {
        let color_offset = match piece {
            Piece::WhiteCap | Piece::WhiteWall | Piece::WhiteFlat => 0,
            Piece::BlackCap | Piece::BlackWall | Piece::BlackFlat => STACKS,
        };
        let idx = color_offset + sq_index * 62 + stack_index;
        self.table[TOPS + idx]
    }
    pub fn manual_build_hash<T: TakBoard>(&self, board: &T) -> u64 {
        let mut hash = 0;
        for (sq, stack) in board.board().iter().enumerate() {
            for (stack_pos, piece) in stack.iter().enumerate() {
                hash ^= self.stack_hash(*piece, sq, stack_pos);
            }
            if let Some(piece) = stack.last() {
                hash ^= self.top_hash(*piece, sq);
            }
        }
        hash ^= self.color_hash(board.active_player());
        hash
    }
    #[allow(dead_code)]
    fn build_table() {
        use std::fs::File;
        use std::io::{BufWriter, Write};
        const SEED: [u8; 32] = [
            9, 60, 225, 206, 252, 134, 26, 117, 109, 127, 202, 188, 75, 28, 41, 96, 219, 156, 76,
            19, 163, 106, 124, 70, 39, 78, 146, 125, 40, 120, 244, 216,
        ];
        let mut table = [0; TOPS + STACKS * 2];
        let mut rng = Xoshiro256PlusPlus::from_seed(SEED);
        for idx in 0..table.len() {
            table[idx] = rng.next_u64();
        }
        let f_name = "src/board/zobrist.table";
        let f = File::create(f_name).unwrap();
        let mut f = BufWriter::new(f);
        write!(f, "[").unwrap();
        for num in table.iter() {
            write!(f, "{},", num).unwrap();
        }
        write!(f, "]").unwrap();
        f.flush().unwrap();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::generate_all_moves;
    use board_game_traits::Position;
    use std::collections::HashSet;
    #[test]
    pub fn test_zobrist_table() {
        let mut seen = HashSet::new();
        for piece_idx in 1..=6 {
            let piece = Piece::from_index(piece_idx);
            for sq_index in 0..36 {
                let hash = TABLE.top_hash(piece, sq_index);
                assert!(seen.insert(hash));
            }
        }
        for piece in [Piece::WhiteFlat, Piece::BlackFlat].iter() {
            for sq_index in 0..36 {
                for stack_index in 0..62 {
                    let hash = TABLE.stack_hash(*piece, sq_index, stack_index);
                    assert!(seen.insert(hash));
                }
            }
        }
    }
    #[test]
    pub fn test_update_zobrist() {
        let tps = "1,112S,2,2,x2/212121S,2S,2,12,2,x/1S,2,11,2122C,x2/1,1,221C,1S,1,1/x3,21,1,x/1,x,1,2,2,x 2 32";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let mut moves = Vec::new();
        let init_zobrist = TABLE.manual_build_hash(&board);
        generate_all_moves(&board, &mut moves);
        assert!(moves.iter().find(|&m| m.crush()).is_some());
        for m in moves {
            let rev = board.do_move(m);
            assert_eq!(board.bits.zobrist(), TABLE.manual_build_hash(&board));
            board.reverse_move(rev);
            assert_eq!(board.bits.zobrist(), init_zobrist);
        }
    }

    // pub fn dummy() {
    //     ZobristTable::build_table();
    // }
}
