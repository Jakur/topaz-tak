use anyhow::{anyhow, bail, ensure, Result};

mod move_gen;

fn main() {
    let x = Bitboard6::new(0x20103c407e00);
    println!("Road: {}", x.check_road());
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Piece {
    WhiteFlat,
    WhiteWall,
    WhiteCap,
    BlackFlat,
    BlackWall,
    BlackCap,
}

impl Piece {
    fn owner(self) -> Player {
        match self {
            Piece::WhiteFlat | Piece::WhiteWall | Piece::WhiteCap => Player::White,
            Piece::BlackFlat | Piece::BlackWall | Piece::BlackCap => Player::Black,
        }
    }
    fn is_wall(self) -> bool {
        match self {
            Piece::WhiteWall | Piece::BlackWall => true,
            _ => false,
        }
    }
    fn is_cap(self) -> bool {
        match self {
            Piece::WhiteCap | Piece::BlackCap => true,
            _ => false,
        }
    }
    fn is_blocker(self) -> bool {
        match self {
            Piece::WhiteFlat | Piece::BlackFlat => false,
            _ => true,
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Player {
    White,
    Black,
}

pub struct Board6 {
    board: [Vec<Piece>; 36],
    active_player: Player,
    move_num: usize,
}

impl Board6 {
    fn new() -> Self {
        const SIZE: usize = 36;
        const INIT: Vec<Piece> = Vec::new();
        Self {
            board: [INIT; SIZE],
            active_player: Player::White,
            move_num: 1,
        }
    }
    const fn size(&self) -> usize {
        6
    }
    fn try_from_tps(tps: &str) -> Result<Self> {
        let data: Vec<_> = tps.split_whitespace().collect();
        ensure!(data.len() == 3, "Malformed tps string!");
        let rows: Vec<_> = data[0].split("/").collect();
        ensure!(rows.len() == 6, "Wrong board size for tps");
        let mut board = Self::new();
        for (r_idx, row) in rows.into_iter().enumerate() {
            let mut col = 0;
            let tiles = row.split(",");
            for tile in tiles {
                if tile.starts_with("x") {
                    if let Some(c) = tile.chars().nth(1) {
                        col += c
                            .to_digit(10)
                            .ok_or_else(|| anyhow!("Failed to parse digit"))?;
                    } else {
                        col += 1;
                    }
                } else {
                    let stack = parse_tps_stack(tile)?;
                    ensure!(col < 6, "Too many columns for this board size");
                    board.board[r_idx * 6 + col as usize].extend(stack.into_iter());
                    col += 1;
                }
            }
        }
        let active_player = match data[1] {
            "1" => Player::White,
            "2" => Player::Black,
            _ => bail!("Unknown active player id"),
        };
        board.active_player = active_player;
        board.move_num = data[2].parse()?;
        Ok(board)
    }
    fn row_col(&self, index: usize) -> (usize, usize) {
        (index / 6, index % 6)
    }
    fn try_tile(&self, row: usize, col: usize) -> Option<&Vec<Piece>> {
        if row >= 6 || col >= 6 {
            None
        } else {
            Some(&self.board[row * 6 + col])
        }
    }
    fn tile(&self, row: usize, col: usize) -> &Vec<Piece> {
        &self.board[row * 6 + col]
    }
    fn tile_mut(&mut self, row: usize, col: usize) -> &mut Vec<Piece> {
        &mut self.board[row * 6 + col]
    }
    fn scan_active_stacks(&self, player: Player) -> Vec<usize> {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, vec)| match vec.last() {
                Some(piece) if piece.owner() == player => Some(i),
                _ => None,
            })
            .collect()
    }
}

fn parse_tps_stack(tile: &str) -> Result<Vec<Piece>> {
    let mut vec = Vec::new();
    for c in tile.chars() {
        match c {
            '1' => vec.push(Piece::WhiteFlat),
            '2' => vec.push(Piece::BlackFlat),
            'S' => match vec.pop() {
                Some(Piece::WhiteFlat) => vec.push(Piece::WhiteWall),
                Some(Piece::BlackFlat) => vec.push(Piece::BlackWall),
                _ => bail!("Bad wall notation"),
            },
            'C' => match vec.pop() {
                Some(Piece::WhiteFlat) => vec.push(Piece::WhiteCap),
                Some(Piece::BlackFlat) => vec.push(Piece::BlackCap),
                _ => bail!("Bad capstone notation"),
            },
            _ => bail!("Unknown character encountered in tile"),
        }
    }
    Ok(vec)
}

// 09 10 11 12 13 14
// 16 17 18 19 20 21

pub trait Bitboard {
    fn adjacent(self) -> Self;
    fn check_road(self) -> bool;
    fn pop_lowest(&mut self) -> Self;
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Bitboard6(u64);

impl std::ops::BitAnd for Bitboard6 {
    type Output = Self;

    fn bitand(self, Self(rhs): Self) -> Self::Output {
        Bitboard6::new(self.0 & rhs)
    }
}

impl std::ops::BitOr for Bitboard6 {
    type Output = Self;

    fn bitor(self, Self(rhs): Self) -> Self::Output {
        Bitboard6::new(self.0 | rhs)
    }
}

impl std::ops::Not for Bitboard6 {
    type Output = Self;

    fn not(self) -> Self::Output {
        Bitboard6::new(!self.0)
    }
}

impl std::ops::BitOrAssign for Bitboard6 {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

impl std::ops::BitAndAssign for Bitboard6 {
    fn bitand_assign(&mut self, other: Self) {
        self.0 &= other.0;
    }
}

impl Bitboard6 {
    const fn new(data: u64) -> Self {
        const INNER: u64 = 0x7e7e7e7e7e7e00; // 6x6 Board
        Self(data & INNER)
    }
    fn nonzero(&self) -> bool {
        self.0 != 0
    }
}

impl Bitboard for Bitboard6 {
    fn adjacent(self) -> Self {
        let Bitboard6(data) = self;
        let up = data >> 8;
        let down = data << 8;
        let left = data >> 1;
        let right = data << 1;
        Bitboard6::new(up | down | left | right)
    }
    fn check_road(self) -> bool {
        const LEFT_TOP: Bitboard6 = Bitboard6::new(0x2020202027e00);
        const TOP: Bitboard6 = Bitboard6::new(0x7e00);
        const BOTTOM: Bitboard6 = Bitboard6::new(0x7e000000000000);
        const LEFT: Bitboard6 = Bitboard6::new(0x2020202020200);
        const RIGHT: Bitboard6 = Bitboard6::new(0x40404040404000);
        let mut unchecked = self & LEFT_TOP; // Only need to start from two edges
        while unchecked.nonzero() {
            // Find all connected components
            let mut component = unchecked.pop_lowest();
            let mut prev = Bitboard6::new(0);
            // While component is still being updated
            while component != prev {
                prev = component;
                component |= component.adjacent() & self
            }
            // If component has squares on two opposite edges there is a road
            if (component & TOP).nonzero() && (component & BOTTOM).nonzero() {
                return true;
            }
            if (component & LEFT).nonzero() && (component & RIGHT).nonzero() {
                return true;
            }
            unchecked = unchecked & !component;
        }
        false
    }
    fn pop_lowest(&mut self) -> Self {
        let highest_index = self.0.trailing_zeros();
        if highest_index == 64 {
            Self::new(0)
        } else {
            let value = 1 << highest_index;
            self.0 ^= value;
            Self::new(value)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    pub fn pop_lowest() {
        let mut x = Bitboard6::new(0x20103c007e00);
        let y = x.pop_lowest();
        assert_eq!(x, Bitboard6::new(0x20103c007c00));
        assert_eq!(y, Bitboard6::new(0x200));
    }
    #[test]
    pub fn bitboard_road() {
        // Positive
        assert!(Bitboard6::new(0x20103c407e00).check_road());
        assert!(Bitboard6::new(0x2020303c446e00).check_road());
        assert!(Bitboard6::new(0xffffffffffffffff).check_road());
        // Negative
        assert!(!Bitboard6::new(0x20103c406e00).check_road());
        assert!(!Bitboard6::new(0x42243c34446200).check_road());
        assert!(!Bitboard6::new(0).check_road());
    }
    #[test]
    pub fn test_read_tps() {
        let example_tps = "x6/x2,2,x3/x3,2C,x2/x2,211S,x2,2/x6/x,1,1,2,2,1 2 7";
        let board = Board6::try_from_tps(example_tps);
        assert!(board.is_ok());
        let board = board.unwrap();
        assert_eq!(board.active_player, Player::Black);
        assert_eq!(board.move_num, 7);
        let mut b = Board6::new();
        b.tile_mut(1, 2).push(Piece::BlackFlat);
        b.tile_mut(2, 3).push(Piece::BlackCap);
        let mut stack = vec![Piece::BlackFlat, Piece::WhiteFlat, Piece::WhiteWall];
        // b.tile_mut(3, 2) = &mut stack;
        std::mem::swap(b.tile_mut(3, 2), &mut stack);
        b.tile_mut(3, 5).push(Piece::BlackFlat);

        b.tile_mut(5, 1).push(Piece::WhiteFlat);
        b.tile_mut(5, 2).push(Piece::WhiteFlat);
        b.tile_mut(5, 3).push(Piece::BlackFlat);
        b.tile_mut(5, 4).push(Piece::BlackFlat);
        b.tile_mut(5, 5).push(Piece::WhiteFlat);
        assert_eq!(board.board, b.board);

        assert_eq!(board.scan_active_stacks(Player::White).len(), 4);
        assert_eq!(board.scan_active_stacks(Player::Black).len(), 5);
    }
}
