use crate::board::Board7;
use crate::{Color, Piece, TakBoard};
// use rand_core::{RngCore, SeedableRng};
// use rand_xoshiro::Xoshiro256PlusPlus;

pub const TABLE: ZobristTable = ZobristTable::new();

const LENGTH: usize = Board7::SIZE;
const MAX_PIECES: usize = Board7::FLATS * 2 + 2;
const TOPS: usize = (LENGTH + 1) * (LENGTH * LENGTH + 1);
const STACKS: usize = MAX_PIECES * (LENGTH * LENGTH);
const ARR_SIZE: usize = TOPS + STACKS * 2 + 2;
/// A Zobrist Table holding enough random values to support a size 7 board
pub struct ZobristTable {
    white_to_move: u64,
    black_to_move: u64,
    table: [u64; ARR_SIZE],
}

impl ZobristTable {
    const fn new() -> Self {
        // const TABLE_DATA: [u64; ARR_SIZE] = [0; ARR_SIZE];
        const TABLE_DATA: [u64; ARR_SIZE] = include!("zobrist.table");
        let len = TABLE_DATA.len();
        Self {
            table: TABLE_DATA,
            white_to_move: TABLE_DATA[len - 1],
            black_to_move: TABLE_DATA[len - 2],
        }
    }
    pub fn color_hash(&self, color: Color) -> u64 {
        match color {
            Color::White => self.white_to_move,
            Color::Black => self.black_to_move,
        }
    }
    pub fn top_hash(&self, piece: Piece, index: usize) -> u64 {
        self.table[piece as usize + index * (LENGTH + 1)]
    }
    pub fn stack_hash(&self, piece: Piece, sq_index: usize, stack_index: usize) -> u64 {
        let color_offset = match piece {
            Piece::WhiteCap | Piece::WhiteWall | Piece::WhiteFlat => 0,
            Piece::BlackCap | Piece::BlackWall | Piece::BlackFlat => STACKS,
        };
        let idx = color_offset + sq_index * MAX_PIECES + stack_index;
        self.table[TOPS + idx]
    }
    pub fn manual_hash_from_stacks(
        &self,
        stacks: &[crate::Stack],
        side_to_move: Color,
    ) -> (u64, u64) {
        let mut hash = 0;
        let mut hash_blocker = 0;
        for (sq, stack) in stacks.iter().enumerate() {
            // if stack.is_empty() {
            //     continue;
            // }
            for (stack_pos, piece) in stack.iter().enumerate() {
                hash ^= self.stack_hash(piece, sq, stack_pos);
            }
            if let Some(piece) = stack.top() {
                if piece.is_blocker() {
                    hash_blocker ^= self.top_hash(piece, sq);
                } else {
                    hash ^= self.top_hash(piece, sq);
                }
            }
        }
        hash ^= self.color_hash(side_to_move);
        (hash_blocker, hash)
    }
    pub fn manual_build_hash<T: TakBoard>(&self, board: &T) -> (u64, u64) {
        self.manual_hash_from_stacks(board.board(), board.side_to_move())
    }
    #[allow(dead_code)]
    #[cfg(feature = "random")]
    fn build_table() {
        use rand_core::{RngCore, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;
        use std::fs::File;
        use std::io::{BufWriter, Write};
        const SEED: [u8; 32] = [
            9, 60, 225, 206, 252, 134, 26, 117, 109, 127, 202, 188, 75, 28, 41, 96, 219, 156, 76,
            19, 163, 106, 124, 70, 39, 78, 146, 125, 40, 120, 244, 216,
        ];
        let mut table = [0; ARR_SIZE];
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
    use crate::board::Board6;
    use crate::generate_all_moves;
    use crate::GameMove;
    use crate::Position;
    use std::collections::HashSet;
    #[test]
    pub fn test_zobrist_table() {
        let mut seen = HashSet::new();
        for piece_idx in 1..=6 {
            let piece = Piece::from_index(piece_idx);
            for sq_index in 0..(LENGTH * LENGTH) {
                let hash = TABLE.top_hash(piece, sq_index);
                assert!(seen.insert(hash));
            }
        }
        for piece in [Piece::WhiteFlat, Piece::BlackFlat].iter() {
            for sq_index in 0..(LENGTH * LENGTH) {
                for stack_index in 0..MAX_PIECES {
                    let hash = TABLE.stack_hash(*piece, sq_index, stack_index);
                    assert!(seen.insert(hash));
                }
            }
        }
    }
    #[test]
    pub fn test_consistent_zobrist_load() {
        let tps2 = "x3,12,x2/x,1,11111221C,x,1,1111112S/1,21,x2,112S,12/2,221S,22,12,1,1/x2,212,2,21,2/2,2,1212C,1,2,12 1 47";
        let data = "1. a1 f6
2. e4 c3
3. f4 e3
4. d4 d3
5. f3 f2
6. Ce2 f5
7. e5 c4
8. d5 Cc5
9. e6 Sd6
10. e1 d6>
11. f3< 2e6-11
12. f3 f5-
13. d2 c2
14. f5 d3+
15. Sd3 c6
16. d3< c5>
17. d3 b4
18. b3 a3
19. d1 2e4+
20. e4 4e5>
21. e5 Se6
22. a2 b2
23. a4 c5
24. e2+ b1
25. a2+ Sa2
26. f1 a2+
27. 3e3<21* e2
28. d2< 2d4-
29. c1 f2-
30. Sa2 d2
31. a2> 2d5-11
32. 3c3+12 5f5+
33. 3c5- f2
34. f5 e6-
35. e3 2e5-
36. d5 2a3>
37. e1+ c3
38. d6 4d3+
39. e5 6f6-
40. c5 c6>
41. 5c4+ 5d4+
42. 5c5-41 6d5<
43. c3<* c5-
44. 4b3+13 e1
45. 2b5> b3
46. 2b2+ 5c4-113";
        let board2 = Board6::try_from_tps(tps2).unwrap();
        let hash_tps = board2.hash();
        // let mut board = Board6::try_from_tps("x6/x6/x6/x6/x6/x6 1 1").unwrap();
        let mut board = Board6::new();
        for mv in data.split_whitespace() {
            if let Some(mv) = GameMove::try_from_ptn(mv, &board) {
                board.do_move(mv);
            }
        }
        assert_eq!(format!("{:?}", &board), format!("{:?}", board2));
        let manual = TABLE.manual_build_hash(&board2);
        assert_eq!(manual.0 ^ manual.1, board2.zobrist());
        assert_eq!(board.hash(), hash_tps);
    }
    #[test]
    pub fn test_update_zobrist() {
        let tps = "1,112S,2,2,x2/212121S,2S,2,12,2,x/1S,2,11,2122C,x2/1,1,221C,1S,1,1/x3,21,1,x/1,x,1,2,2,x 2 32";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let mut moves: Vec<GameMove> = Vec::new();
        let init_zobrist = TABLE.manual_build_hash(&board);
        generate_all_moves(&board, &mut moves);
        assert!(moves.iter().any(|m| m.crush()));
        for m in moves {
            let rev = board.do_move(m);
            let hash = TABLE.manual_build_hash(&board);
            assert_eq!(board.bits.zobrist(), hash.0 ^ hash.1);
            board.reverse_move(rev);
            assert_eq!(board.bits.zobrist(), init_zobrist.0 ^ init_zobrist.1);
        }
    }
    #[test]
    pub fn consistent_hash() {
        // Used for checking that zobrist hashes are as expected for move gen and do_move purposes
        // This function must be changed if the zobrist keys themselves are updated
        let tps = "1,1,1,1,1,2/x,1,1,2,2,2/1,1,112S,2S,2S,1/1,2,2,2,21221,21/21,2,2,21,x,12/12,12C,x,221,2222221C,x 1 39";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let mut buffer = Vec::new();
        generate_all_moves(&board, &mut buffer);
        assert_eq!(buffer.len(), 174);
        let mut hash = 0;
        for mv in buffer {
            let rev = board.do_move(mv);
            hash ^= board.hash();
            board.reverse_move(rev);
        }
        assert_eq!(hash, 9736573932680568458);
    }
    // #[test]
    // pub fn dummy() {
    //     ZobristTable::build_table();
    // }
}
