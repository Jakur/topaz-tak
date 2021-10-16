use board_game_traits::Color;

#[derive(PartialEq, Clone, Copy)]
pub enum Piece {
    WhiteFlat = 1,
    WhiteWall = 2,
    WhiteCap = 3,
    BlackFlat = 4,
    BlackWall = 5,
    BlackCap = 6,
}

impl Piece {
    pub fn from_index(index: u64) -> Self {
        match index {
            1 => Piece::WhiteFlat,
            2 => Piece::WhiteWall,
            3 => Piece::WhiteCap,
            4 => Piece::BlackFlat,
            5 => Piece::BlackWall,
            6 => Piece::BlackCap,
            _ => unimplemented!(),
        }
    }
    pub fn owner(self) -> Color {
        match self {
            Piece::WhiteFlat | Piece::WhiteWall | Piece::WhiteCap => Color::White,
            Piece::BlackFlat | Piece::BlackWall | Piece::BlackCap => Color::Black,
        }
    }
    pub fn is_wall(self) -> bool {
        match self {
            Piece::WhiteWall | Piece::BlackWall => true,
            _ => false,
        }
    }
    pub fn is_cap(self) -> bool {
        match self {
            Piece::WhiteCap | Piece::BlackCap => true,
            _ => false,
        }
    }
    pub fn is_blocker(self) -> bool {
        match self {
            Piece::WhiteFlat | Piece::BlackFlat => false,
            _ => true,
        }
    }
    pub fn wall(color: Color) -> Self {
        match color {
            Color::White => Piece::WhiteWall,
            Color::Black => Piece::BlackWall,
        }
    }
    pub fn flat(color: Color) -> Self {
        match color {
            Color::White => Piece::WhiteFlat,
            Color::Black => Piece::BlackFlat,
        }
    }
    pub fn cap(color: Color) -> Self {
        match color {
            Color::White => Piece::WhiteCap,
            Color::Black => Piece::BlackCap,
        }
    }
    pub fn crush(self) -> Option<Piece> {
        match self {
            Piece::WhiteWall => Some(Piece::WhiteFlat),
            Piece::BlackWall => Some(Piece::BlackFlat),
            _ => None,
        }
    }
    pub fn uncrush(self) -> Option<Piece> {
        match self {
            Piece::WhiteFlat => Some(Piece::WhiteWall),
            Piece::BlackFlat => Some(Piece::BlackWall),
            _ => None,
        }
    }
    pub fn swap_color(self) -> Self {
        match self {
            Piece::WhiteFlat => Piece::BlackFlat,
            Piece::BlackFlat => Piece::WhiteFlat,
            Piece::WhiteCap => Piece::BlackCap,
            Piece::BlackCap => Piece::WhiteCap,
            Piece::WhiteWall => Piece::BlackWall,
            Piece::BlackWall => Piece::WhiteWall,
        }
    }
    pub fn road_piece(self, color: Color) -> bool {
        if let Color::White = color {
            match self {
                Piece::WhiteFlat | Piece::WhiteCap => true,
                _ => false,
            }
        } else {
            match self {
                Piece::BlackFlat | Piece::BlackCap => true,
                _ => false,
            }
        }
    }
}

impl std::fmt::Debug for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let s = match self {
            Piece::WhiteFlat => "w",
            Piece::BlackFlat => "b",
            Piece::WhiteCap => "C",
            Piece::BlackCap => "D",
            Piece::WhiteWall => "S",
            Piece::BlackWall => "T",
        };
        write!(f, "{}", s)
    }
}
