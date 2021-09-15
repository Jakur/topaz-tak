fn main() {
    let x = Bitboard6::new(0x20103c407e00);
    println!("Road: {}", x.check_road());
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
}
