use super::{Board6, GameMove, Piece};
use crate::eval::TakBoard;

impl GameMove {
    pub fn to_ptn(self) -> String {
        let square = tile_ptn(self.src_index());
        if self.is_place_move() {
            match self.place_piece() {
                Piece::WhiteCap | Piece::BlackCap => return format!("C{}", square),
                Piece::WhiteWall | Piece::BlackWall => return format!("S{}", square),
                _ => return square,
            }
        }
        let dir = match self.direction() {
            0 => "+",
            1 => ">",
            2 => "-",
            3 => "<",
            _ => unimplemented!(),
        };
        let spread_bits = (self.0 & 0xFFFFFFF000) >> 12;
        let spread = format!("{:X}", spread_bits);
        // Our bitwise order is reversed, so we need to reverse this String
        let spread: String = spread.chars().rev().collect();
        let num_string = if self.number() == 1 {
            "".to_string()
        } else {
            format!("{}", self.number())
        };
        let crush_str = if self.crush() { "*" } else { "" };
        if format!("{}", self.number()) == spread {
            format!("{}{}{}{}", num_string, square, dir, crush_str)
        } else {
            format!("{}{}{}{}{}", num_string, square, dir, spread, crush_str)
        }
    }
    // fn try_from_ptn(s: &str, size: usize) -> Result<Self>
    pub fn try_from_ptn(s: &str, board: &Board6) -> Option<Self> {
        let mut iter = s.chars().take_while(|&c| c != '*');
        let first = iter.next()?;
        let pieces = first.to_digit(10);
        let (row, col) = if pieces.is_some() || first == 'C' || first == 'S' {
            // The next two tiles are the square
            let col = iter.next()?;
            let row = iter.next()?;
            (row, col)
        } else {
            // The first tile was the square, so we only need the row
            let row = iter.next()?;
            (row, first)
        };
        let col = match col {
            'a' => 0,
            'b' => 1,
            'c' => 2,
            'd' => 3,
            'e' => 4,
            'f' => 5,
            'g' => 6,
            'h' => 7,
            _ => return None,
        };
        // ensure!(col < size, "Board column out of range");
        let row = board.board_size() - row.to_digit(10)? as usize;
        // ensure!(row < size, "Board row out of range");
        let square = row * board.board_size() + col;

        if let Some(dir) = iter.next() {
            // Stack Move
            // let stack_move = Self(square);
            let pieces = pieces.unwrap_or(1) as u64;
            let dir = match dir {
                '+' => 0,
                '>' => 1,
                '-' => 2,
                '<' => 3,
                _ => return None,
            };
            let crush = s.ends_with("*");
            let mut slide_bits = 0u64;

            let mut counter = 0;
            while let Some(ch) = iter.next() {
                if let Some(num) = ch.to_digit(16) {
                    let value = (num as u64) << (4 * counter);
                    slide_bits |= value;
                    counter += 1;
                } else {
                    break;
                }
            }
            if slide_bits == 0 {
                slide_bits = pieces;
            }
            Some(
                Self(slide_bits << 12)
                    .set_number(pieces)
                    .set_direction(dir)
                    .chain_crush(crush)
                    .set_index(square as u64),
            )
        } else {
            // Placement
            let color = board.active_player();
            let piece = if first == 'S' {
                Piece::wall(color)
            } else if first == 'C' {
                Piece::cap(color)
            } else {
                Piece::flat(color)
            };
            Some(Self::from_placement(piece, square))
        }
    }
}

fn tile_ptn(index: usize) -> String {
    let size = Board6::SIZE;
    let (row, col) = Board6::row_col_static(index);
    let col = match col {
        0 => "a",
        1 => "b",
        2 => "c",
        3 => "d",
        4 => "e",
        5 => "f",
        _ => unimplemented!(),
    };
    let row = size - row;
    format!("{}{}", col, row)
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    pub fn ptn_equivalence() {
        let s = "2,2,2,21,12,x/x4,2,x/x4,2C,x/1,2,12,122211C,x2/x2,1S,1,12,1/x3,2S,1,1 1 19";
        let board = Board6::try_from_tps(s).unwrap();
        // The last move is not legal, but we're not checking for that here
        let ptn = &[
            "a2", "c2+", "d2>", "f2<", "a3-", "5d3<41", "5d3-41*", "6d3+", "Sf6", "Ca1",
        ];
        for p in ptn {
            let m = GameMove::try_from_ptn(p, &board).unwrap();
            assert_eq!(p, &m.to_ptn())
        }
    }
}
