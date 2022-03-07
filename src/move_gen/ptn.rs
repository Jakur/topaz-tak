use board_game_traits::Color;

use super::{GameMove, Piece};
use crate::board::TakBoard;

impl GameMove {
    pub fn to_ptn<T: TakBoard>(self) -> String {
        let square = tile_ptn::<T>(self.src_index());
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
        let spread_bits = self.large_slide_bits();
        // let spread_bits = 0;
        // todo!();
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
    pub fn to_playtak<T: TakBoard>(self) -> String {
        let origin = tile_ptn::<T>(self.src_index()).to_ascii_uppercase();
        if self.is_stack_move() {
            let quantities: Vec<_> = self.quantity_iter(T::SIZE).collect();
            let dest = tile_ptn::<T>(quantities.last().unwrap().index).to_ascii_uppercase();
            let mut move_q_s = String::new();
            for q in quantities {
                move_q_s.push_str(&format!("{}", q.quantity));
                move_q_s.push(' ');
            }
            move_q_s.pop();
            format!("M {} {} {}", origin, dest, move_q_s)
        } else {
            match self.place_piece() {
                Piece::WhiteFlat | Piece::BlackFlat => {
                    format!("P {}", origin)
                }
                Piece::WhiteWall | Piece::BlackWall => {
                    format!("P {} W", origin)
                }
                Piece::WhiteCap | Piece::BlackCap => {
                    format!("P {} C", origin)
                }
            }
        }
    }
    pub fn try_from_playtak<T: TakBoard>(s: &str, board: &T) -> Option<Self> {
        let size = T::SIZE;
        let side_to_move = board.active_player();
        let mut split = s.split_whitespace();
        let stack_move = match split.next()? {
            "M" => true,
            "P" => false,
            _ => return None,
        };
        let origin = parse_playtak_sq(split.next()?, size)?;
        if stack_move {
            let mut game_move = GameMove(0).set_index(origin as u32);
            let dest = parse_playtak_sq(split.next()?, size)?;
            let dir = if origin > dest {
                // North or West
                if origin - dest >= size {
                    0 // North
                } else {
                    3 // West
                }
            } else {
                if dest - origin >= size {
                    2 // South
                } else {
                    1 // East
                }
            };
            game_move = game_move.set_direction(dir);
            let mut tile_counter = 1;
            let mut total_pieces = 0;
            while let Some(num) = split.next() {
                game_move = match tile_counter {
                    1..=7 => {
                        let pieces = num.parse().ok()?;
                        total_pieces += pieces;
                        game_move.set_next_tile(pieces)
                    }
                    _ => return None,
                };
                tile_counter += 1;
            }
            if let Some(p) = board.index(dest).last() {
                if p.is_wall() {
                    game_move = game_move.set_crush();
                }
            }
            Some(game_move.set_number(total_pieces))
        } else {
            let piece = match split.next() {
                Some("W") => Piece::wall(side_to_move),
                Some("C") => Piece::cap(side_to_move),
                None => Piece::flat(side_to_move),
                _ => return None,
            };
            Some(GameMove::from_placement(piece, origin))
        }
    }
    pub fn try_from_ptn<T: TakBoard>(s: &str, board: &T) -> Option<Self> {
        Self::try_from_ptn_m(s, T::SIZE, board.side_to_move())
    }
    pub fn try_from_ptn_m(s: &str, size: usize, active_player: Color) -> Option<Self> {
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
        let row = size - row.to_digit(10)? as usize;
        let square = row * size + col;

        if let Some(dir) = iter.next() {
            // Stack Move
            let pieces = pieces.unwrap_or(1) as u32;
            let dir = match dir {
                '+' => 0,
                '>' => 1,
                '-' => 2,
                '<' => 3,
                _ => return None,
            };
            let crush = s.ends_with("*");
            let mut mv = Self(0)
                .set_number(pieces)
                .set_direction(dir)
                .chain_crush(crush)
                .set_index(square as u32);
            let mut counter = 0;
            while let Some(ch) = iter.next() {
                if let Some(num) = ch.to_digit(16) {
                    mv = mv.set_next_tile(num);
                    // let value = (num as u64) << (4 * counter);
                    // slide_bits |= value;
                    counter += 1;
                } else {
                    break;
                }
            }
            if counter == 0 {
                mv = mv.set_next_tile(mv.number());
            }
            Some(mv)
        } else {
            // Placement
            let color = active_player;
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

fn parse_playtak_sq(s: &str, size: usize) -> Option<usize> {
    let mut sq = s.chars();
    let col = match sq.next()? {
        'A' => 0,
        'B' => 1,
        'C' => 2,
        'D' => 3,
        'E' => 4,
        'F' => 5,
        'G' => 6,
        'H' => 7,
        _ => return None,
    };
    let row = size - sq.next().and_then(|x| x.to_digit(10))? as usize;
    Some(row * size + col)
}

fn tile_ptn<T: TakBoard>(index: usize) -> String {
    let size = T::SIZE;
    let (row, col) = T::row_col_static(index);
    let col = match col {
        0 => "a",
        1 => "b",
        2 => "c",
        3 => "d",
        4 => "e",
        5 => "f",
        6 => "g",
        _ => unimplemented!(),
    };
    let row = size - row;
    format!("{}{}", col, row)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{generate_all_moves, Board6};
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
            assert_eq!(p, &m.to_ptn::<Board6>())
        }
    }
    #[test]
    pub fn playtak_move() {
        let s = "2,2,2,21,12,x/x4,2,x/x4,2C,x/1,2,12,122211C,x2/x2,1S,1,12,1/x3,2S,1,1 1 19";
        let board = Board6::try_from_tps(s).unwrap();
        let ptn = "5b4>32";
        let m = GameMove::try_from_ptn(ptn, &board).unwrap();
        let pt = m.to_playtak::<Board6>();
        assert_eq!(pt.as_str(), "M B4 D4 3 2");
        let m2 = GameMove::try_from_playtak(&pt, &board).unwrap();
        assert_eq!(m, m2);
        let mut buffer = Vec::new();
        generate_all_moves(&board, &mut buffer);
        for m in buffer {
            let pt = m.to_playtak::<Board6>();
            let same = GameMove::try_from_playtak(&pt, &board).unwrap();
            assert_eq!(same.direction(), m.direction());
            assert_eq!(same, m);
        }
    }
}
