use super::*;
use crate::board::TakBoard;
use crate::move_gen::generate_all_moves;
use crate::Piece;
use crate::RevGameMove;
use anyhow::{anyhow, Result};
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use termtree::Tree;

const INFINITY: u32 = 100_000_000;

#[derive(Clone)]
pub struct Child {
    bounds: Bounds,
    game_move: GameMove,
    zobrist: u64,
    best_child: usize,
}

impl Child {
    fn new(bounds: Bounds, game_move: GameMove, zobrist: u64) -> Self {
        Self {
            bounds,
            game_move,
            zobrist,
            best_child: usize::MAX,
        }
    }
    fn update_best_child(
        &mut self,
        best_child: usize,
        game_move: GameMove,
        table: &mut HashMap<u64, GameMove>,
    ) {
        self.best_child = best_child;
        table.insert(self.zobrist, game_move);
    }
    fn update_bounds(&mut self, bounds: Bounds, table: &mut HashMap<u64, Bounds>) {
        self.bounds = bounds;
        table.insert(self.zobrist, bounds);
    }
    fn phi(&self) -> u32 {
        self.bounds.phi
    }
    fn delta(&self) -> u32 {
        self.bounds.delta
    }
}

fn compute_bounds(children: &[Child]) -> Bounds {
    let mut out = Bounds {
        phi: INFINITY,
        delta: 0,
    };
    for ch in children.iter() {
        out.phi = min(out.phi, ch.bounds.delta);
        out.delta = out.delta.saturating_add(ch.bounds.phi);
    }
    out.delta = min(out.delta, INFINITY);
    return out;
}

#[derive(Debug, Clone, Copy)]
struct Bounds {
    phi: u32,
    delta: u32,
}

impl Bounds {
    pub fn winning() -> Self {
        Bounds {
            phi: 0,
            delta: INFINITY,
        }
    }

    pub fn losing() -> Self {
        Bounds {
            phi: INFINITY,
            delta: 0,
        }
    }
    pub fn infinity() -> Self {
        Bounds {
            phi: INFINITY,
            delta: INFINITY,
        }
    }

    pub fn root() -> Self {
        Bounds {
            phi: INFINITY / 2,
            delta: INFINITY / 2,
        }
    }
}

#[derive(Clone)]
struct TopMoves {
    moves: [GameMove; Self::MAX_SIZE],
    size: usize,
}

impl TopMoves {
    const MAX_SIZE: usize = 3;
    fn new() -> Self {
        Self {
            moves: [GameMove::null_move(); 3],
            size: 0,
        }
    }
    fn get_best(&self) -> &[GameMove] {
        &self.moves[0..self.size]
    }
    fn add_move(&mut self, game_move: GameMove) {
        if self.size == Self::MAX_SIZE {
            // Just give up for now lol
        } else {
            if !self.moves.contains(&game_move) {
                self.moves[self.size] = game_move;
                self.size += 1;
            }
        }
    }
}

pub struct InteractiveSearch<T> {
    pub board: T,
    bounds_table: HashMap<u64, Bounds>,
    tinue_attempts: HashMap<u64, AttackerOutcome>,
    expand: HashSet<u64>,
    view_hist: Vec<(GameMove, RevGameMove)>,
}

impl<T> InteractiveSearch<T>
where
    T: TakBoard,
{
    pub fn new(search: TinueSearch<T>) -> Self {
        let mut expand = HashSet::new();
        expand.insert(search.board.hash());
        InteractiveSearch {
            board: search.board,
            bounds_table: search.bounds_table,
            tinue_attempts: search.tinue_attempts,
            expand,
            view_hist: Vec::new(),
        }
    }
    pub fn change_view(&mut self, line: &str) -> Result<()> {
        for ptn in line.split('/') {
            let m = GameMove::try_from_ptn(ptn, &self.board)
                .ok_or_else(|| anyhow!("Unable to parse ptn!"))?;
            let mut legal_moves = Vec::new();
            generate_all_moves(&self.board, &mut legal_moves);
            if legal_moves.contains(&m) {
                let rev = self.board.do_move(m);
                self.view_hist.push((m, rev));
            } else {
                return Err(anyhow!("Attempted to execute illegal move, breaking!"));
            }
        }
        Ok(())
    }
    pub fn expand_line(&mut self, moves: Vec<&str>) {
        let mut rev_moves = Vec::new();
        for ptn in moves.iter() {
            let m = GameMove::try_from_ptn(ptn, &self.board).unwrap();
            let rev = self.board.do_move(m);
            self.expand.insert(self.board.hash());
            rev_moves.push(rev);
        }
        for rev in rev_moves.into_iter().rev() {
            self.board.reverse_move(rev);
        }
    }
    pub fn reset_view(&mut self) {
        for (_, rev) in self.view_hist.drain(..).rev() {
            self.board.reverse_move(rev);
        }
    }
    pub fn reset_expansion(&mut self) {
        self.expand.clear();
    }
    pub fn print_root(&mut self) {
        let line = self
            .view_hist
            .iter()
            .map(|(m, _)| m.to_ptn::<T>())
            .collect();
        let mut tree = Tree::root(Solved::Root(line));
        if self.view_hist.len() % 2 == 0 {
            self.recurse_attack(&mut tree);
        } else {
            self.recurse_defend(&mut tree);
        }
        println!("{}", tree);
    }
    fn recurse_attack(&mut self, root: &mut Tree<Solved<T>>) {
        let attempt = self.tinue_attempts.get(&self.board.hash());
        match attempt {
            Some(AttackerOutcome::TakThreats(moves)) => {
                for m in moves.clone().into_iter() {
                    // Children will be a defender node
                    let rev = self.board.do_move(m);
                    let bounds = self.bounds_table.get(&self.board.hash()).unwrap();
                    let solved = if bounds.phi == INFINITY {
                        Solved::Proved(m)
                    } else if bounds.phi == 0 {
                        Solved::Disproved(m)
                    } else {
                        Solved::Unknown(m)
                    };
                    let mut child = Tree::root(solved);
                    if self.expand.contains(&self.board.hash()) {
                        self.recurse_defend(&mut child);
                    }
                    root.push(child);
                    self.board.reverse_move(rev);
                }
            }
            Some(AttackerOutcome::NoTakThreats) => {
                root.push(Tree::root(Solved::AttackerNoMoves(PhantomData)));
            }
            Some(AttackerOutcome::HasRoad(m)) => {
                root.push(Tree::root(Solved::AttackerRoad(*m)));
            }
            None => todo!(),
        }
    }
    fn recurse_defend(&mut self, root: &mut Tree<Solved<T>>) {
        let attempt = TinueSearch::defender_responses(&mut self.board, None);
        match attempt {
            DefenderOutcome::CanWin(m) => {
                root.push(Tree::root(Solved::DefenderRoad(m)));
            }
            DefenderOutcome::Defenses(vec) => {
                for m in vec {
                    // Children will be a attacker node
                    let rev = self.board.do_move(m);
                    let bounds = self.bounds_table.get(&self.board.hash()).unwrap();
                    let solved = if bounds.phi == INFINITY {
                        Solved::Disproved(m)
                    } else if bounds.phi == 0 {
                        Solved::Proved(m)
                    } else {
                        Solved::Unknown(m)
                    };
                    let mut child = Tree::root(solved);
                    if self.expand.contains(&self.board.hash()) {
                        self.recurse_attack(&mut child);
                    }
                    root.push(child);
                    self.board.reverse_move(rev);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
enum Solved<T> {
    Proved(GameMove),
    Disproved(GameMove),
    Unknown(GameMove),
    AttackerRoad(GameMove),
    DefenderRoad(GameMove),
    AttackerNoMoves(PhantomData<T>),
    Root(Vec<String>),
}

impl<T> std::fmt::Display for Solved<T>
where
    T: TakBoard,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        use colorful::Color;
        use colorful::Colorful;
        let s = match self {
            Solved::Proved(m) => m.to_ptn::<T>(),
            Solved::Disproved(m) => m.to_ptn::<T>(),
            Solved::Unknown(m) => return write!(f, "{}", m.to_ptn::<T>()),
            Solved::AttackerRoad(m) => format!("{}''", m.to_ptn::<T>()),
            Solved::DefenderRoad(m) => format!("{}''", m.to_ptn::<T>()),
            Solved::AttackerNoMoves(_) => "∅".to_string(),
            Solved::Root(vec) => {
                let move_str = vec.join("/");
                return write!(f, "ROOT({})", move_str);
            }
        };
        let color = match self {
            Solved::Proved(_) | Solved::AttackerRoad(_) => Color::Blue,
            Solved::Disproved(_) | Solved::DefenderRoad(_) | Solved::AttackerNoMoves(_) => {
                Color::Red
            }
            _ => unreachable!(),
        };
        write!(f, "{}", s.color(color))
    }
}

pub struct TinueSearch<T> {
    pub board: T,
    bounds_table: HashMap<u64, Bounds>,
    rev_moves: Vec<RevGameMove>,
    zobrist_hist: Vec<u64>,
    attacker: Color,
    pub nodes: usize,
    top_moves: Vec<TopMoves>,
    tinue_attempts: HashMap<u64, AttackerOutcome>,
    pub replies: HashMap<u64, GameMove>,
    tinue_cache_hits: usize,
    tinue_cache_misses: usize,
    quiet: bool,
    max_nodes: usize,
}

impl<T> TinueSearch<T>
where
    T: TakBoard,
{
    pub fn new(board: T) -> Self {
        let attacker = board.side_to_move();
        Self {
            board,
            bounds_table: HashMap::new(),
            rev_moves: Vec::new(),
            attacker,
            nodes: 0,
            top_moves: vec![TopMoves::new(); 100],
            replies: HashMap::new(),
            tinue_attempts: HashMap::new(),
            tinue_cache_hits: 0,
            tinue_cache_misses: 0,
            zobrist_hist: Vec::new(),
            quiet: false,
            max_nodes: usize::MAX,
        }
    }
    /// Searches for tinue in the provided position, returning None if the search aborted early
    pub fn is_tinue(&mut self) -> Option<bool> {
        let mut root = Child::new(Bounds::root(), GameMove::null_move(), self.board.hash());
        self.mid(&mut root, 0);
        if !self.quiet {
            dbg!(self.nodes);
            dbg!(self.tinue_cache_hits);
            dbg!(self.tinue_cache_misses);
        }
        if self.aborted() {
            return None;
        }
        if root.delta() == INFINITY {
            Some(self.attacker == self.board.side_to_move())
        } else {
            Some(self.attacker != self.board.side_to_move())
        }
    }
    /// Returns true if the tinue search starts with the attacker's move
    pub fn is_attacker(&self) -> bool {
        self.attacker == self.board.side_to_move()
    }
    /// Applies a limit of max_nodes in a builder pattern
    pub fn limit(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }
    /// Suppresses CLI output of the search in a builder pattern
    pub fn quiet(mut self) -> Self {
        self.quiet = true;
        self
    }
    /// Set whether the attacker is the next player in a builder pattern
    pub fn attacker(mut self, next_side: bool) -> Self {
        let color = if next_side {
            self.board.side_to_move()
        } else {
            !self.board.side_to_move()
        };
        self.attacker = color;
        return self;
    }
    /// Returns true if the search aborted due to exceeding max_nodes before finding a solution
    pub fn aborted(&self) -> bool {
        self.nodes > self.max_nodes
    }
    /// Crudely returns the principal variation by looking at the last refuted line
    pub fn principal_variation(&mut self) -> Vec<GameMove> {
        // Todo defender or attacker agnostic
        let root = self.bounds_table.get(&self.board.hash()).unwrap();
        let attacker_wins = {
            // Give attacker benefit of the doubt for these purposes
            if root.delta != 0 {
                self.is_attacker()
            } else {
                !self.is_attacker()
            }
        };
        let mut hist = Vec::new();
        let mut pv = Vec::new();
        // Timeout after 100 ply in case of cycles
        for _ in 0..100 {
            match (self.is_attacker(), attacker_wins) {
                (true, true) | (false, false) => {
                    // We are on the winning side, pick the best valid move
                    let mut moves = Vec::new();
                    if let Some(mv) = self.board.can_make_road(&mut moves, None) {
                        pv.add_move(mv);
                        break;
                    }
                    moves.clear();
                    generate_all_moves(&self.board, &mut moves);
                    let search = if self.is_attacker() {
                        self.board.get_tak_threats(&moves, None)
                    } else {
                        moves
                    };
                    let min = search.into_iter().min_by_key(|&mv| {
                        let rev = self.board.do_move(mv);
                        let lookup = self
                            .bounds_table
                            .get(&self.board.hash())
                            .map(|x| x.delta)
                            .unwrap_or(INFINITY);
                        self.board.reverse_move(rev);
                        lookup
                    });
                    if let Some(mv) = min {
                        hist.push(self.board.do_move(mv));
                        pv.push(mv);
                    } else {
                        break;
                    }
                }
                (true, false) | (false, true) => {
                    // We are on the losing side, follow last analyzed line
                    if let Some(&game_move) = self.replies.get(&self.board.hash()) {
                        pv.push(game_move);
                        let rev = self.board.do_move(game_move);
                        hist.push(rev);
                    } else {
                        break;
                    }
                }
            }
        }
        for rev_move in hist.into_iter().rev() {
            self.board.reverse_move(rev_move);
        }
        pv
    }
    fn mid(&mut self, child: &mut Child, depth: usize) {
        self.nodes += 1;
        if self.nodes > self.max_nodes {
            return;
        }
        if child.game_move != GameMove::null_move() {
            let rev = self.board.do_move(child.game_move);
            self.rev_moves.push(rev);
        }
        self.zobrist_hist.push(self.board.hash());
        assert_eq!(child.zobrist, self.board.hash());
        let side_to_move = self.board.side_to_move();
        let attacker = side_to_move == self.attacker;
        if self.board.flat_game().is_some() {
            let eval = if attacker {
                Bounds::losing()
            } else {
                Bounds::winning()
            };
            child.update_bounds(eval, &mut self.bounds_table);
            self.undo_move();
            return;
        }
        let moves = if attacker {
            let tinue_eval = if let Some(cached_val) = self.tinue_attempts.get(&self.board.hash()) {
                self.tinue_cache_hits += 1;
                cached_val.clone()
            } else {
                self.tinue_cache_misses += 1;
                let outcome = self.tinue_evaluate(depth);
                self.tinue_attempts
                    .insert(self.board.hash(), outcome.clone());
                outcome
            };
            match tinue_eval {
                AttackerOutcome::HasRoad(_m) => {
                    let eval = Bounds::winning();
                    child.update_bounds(eval, &mut self.bounds_table);
                    self.undo_move();
                    return;
                }
                AttackerOutcome::TakThreats(vec) => vec,
                AttackerOutcome::NoTakThreats => {
                    let eval = Bounds::losing();
                    child.update_bounds(eval, &mut self.bounds_table);
                    self.undo_move();
                    return;
                }
            }
        } else {
            match Self::defender_responses(
                &mut self.board,
                self.top_moves.get(depth).map(|t| t.get_best()),
            ) {
                DefenderOutcome::CanWin(m) => {
                    self.top_moves[depth].add_move(m);
                    let eval = Bounds::winning();
                    child.update_bounds(eval, &mut self.bounds_table);
                    self.undo_move();
                    return;
                }
                DefenderOutcome::Defenses(moves) => {
                    if moves.is_empty() {
                        let eval = Bounds::losing();
                        child.update_bounds(eval, &mut self.bounds_table);
                        self.undo_move();
                        return;
                    }
                    moves
                }
            }
        };
        assert!(!moves.is_empty());

        if child.game_move == GameMove::null_move() && !self.quiet {
            let debug_vec: Vec<_> = moves.iter().map(|m| m.to_ptn::<T>()).collect();
            println!("All Tak Threats at the Root: ");
            dbg!(&debug_vec); // Root moves
        }
        let mut child_pns: Vec<_> = moves
            .into_iter()
            .filter_map(|m| self.init_pns(m, depth as u32))
            .collect();
        loop {
            let limit = compute_bounds(&child_pns);
            if child.phi() <= limit.phi || child.delta() <= limit.delta {
                child.update_bounds(limit, &mut self.bounds_table);
                self.undo_move();
                return;
            }
            let (best_idx, second_best_bounds) = Self::select_child(&child_pns);
            child.update_best_child(best_idx, child_pns[best_idx].game_move, &mut self.replies);
            let best_child = &mut child_pns[best_idx];
            let updated_bounds = Bounds {
                phi: child.delta() + best_child.phi() - limit.delta,
                delta: min(child.phi(), second_best_bounds.delta + 1),
            };
            best_child.update_bounds(updated_bounds, &mut self.bounds_table);
            self.mid(best_child, depth + 1);
        }
    }
    fn defender_responses(board: &mut T, hint: Option<&[GameMove]>) -> DefenderOutcome {
        let mut moves = Vec::new();
        if let Some(m) = board.can_make_road(&mut moves, hint) {
            return DefenderOutcome::CanWin(m);
        }
        moves.clear();
        let side = board.side_to_move();
        let enemy = !board.side_to_move();
        let has_cap = board.caps_reserve(side) > 0;
        // Todo optimization: only check if there is only one flat threat
        board.null_move();
        let threat = board.can_make_road(&mut moves, None).unwrap();
        board.rev_null_move();
        moves.clear();
        if threat.is_place_move() {
            let sq = threat.src_index();
            moves.push(GameMove::from_placement(Piece::flat(side), sq));
            moves.push(GameMove::from_placement(Piece::wall(side), sq));
            if has_cap {
                moves.push(GameMove::from_placement(Piece::cap(side), sq));
            }
        } else {
            let wall = crate::Piece::wall(!enemy);
            for item in threat.quantity_iter(T::SIZE) {
                let sq = item.index;
                if board.index(sq).is_empty() {
                    moves.push(GameMove::from_placement(wall, sq));
                    if has_cap {
                        moves.push(GameMove::from_placement(Piece::cap(side), sq));
                    }
                }
            }
        }
        generate_all_stack_moves(board, &mut moves);
        DefenderOutcome::Defenses(moves)
    }
    fn select_child(children: &[Child]) -> (usize, Bounds) {
        let mut c_best_idx = 0;
        let mut best = children[c_best_idx].bounds;
        let mut second_best = Bounds::infinity();
        for (idx, child) in children.iter().enumerate().skip(1) {
            if child.bounds.delta < best.delta {
                c_best_idx = idx;
                second_best = best;
                best = child.bounds;
            } else if child.bounds.delta < second_best.delta {
                second_best = child.bounds;
            }
            if child.bounds.phi == INFINITY {
                return (c_best_idx, second_best);
            }
        }
        (c_best_idx, second_best)
    }
    fn init_pns(&mut self, game_move: GameMove, depth: u32) -> Option<Child> {
        let side_to_move = self.board.side_to_move();
        let attacker = side_to_move == self.attacker;
        let rev = self.board.do_move(game_move);
        let hash = self.board.hash();
        // let default_bounds = Bounds::default();
        let default_bounds = if attacker {
            // Child is defensive node
            Bounds {
                phi: 1,
                delta: 30 + depth * depth,
            }
        } else {
            // Child is offensive node
            if game_move.is_stack_move() {
                Bounds {
                    phi: 10 + depth * depth,
                    delta: 1,
                }
            } else {
                Bounds {
                    phi: 20 + depth * depth,
                    delta: 1,
                }
            }
        };
        let bounds = self.bounds_table.entry(hash).or_insert(default_bounds);

        let child = Child::new(*bounds, game_move, hash);
        self.board.reverse_move(rev);
        if attacker && self.zobrist_hist.contains(&hash) {
            return None;
        }
        Some(child)
    }
    fn undo_move(&mut self) -> Option<()> {
        let m = self.rev_moves.pop()?;
        self.board.reverse_move(m);
        self.zobrist_hist.pop();
        Some(())
    }
    fn tinue_evaluate(&mut self, depth: usize) -> AttackerOutcome {
        let pos = &mut self.board;
        let mut moves = Vec::new();
        if let Some(m) = pos.can_make_road(&mut moves, Some(self.top_moves[depth].get_best())) {
            if m.is_stack_move() {
                self.top_moves[depth].add_move(m);
            }
            return AttackerOutcome::HasRoad(m);
        }
        // Give up
        if depth + 2 >= 98 {
            return AttackerOutcome::NoTakThreats;
        }
        moves.clear();
        crate::move_gen::generate_all_stack_moves(pos, &mut moves);
        crate::move_gen::generate_aggressive_place_moves(pos, &mut moves);
        let tak_threats = pos.get_tak_threats(&moves, Some(self.top_moves[depth + 2].get_best()));
        if tak_threats.is_empty() {
            AttackerOutcome::NoTakThreats
        } else {
            AttackerOutcome::TakThreats(tak_threats)
        }
    }
    pub fn rebuild<W: Write>(
        &mut self,
        writer: &mut W,
        hist: &mut Vec<String>,
        zobrist_hist: &mut HashSet<u64>,
        attacker: bool,
    ) -> Result<()> {
        let mut moves = Vec::new();
        if let Some(mv) = self.board.can_make_road(&mut moves, None) {
            moves = vec![mv];
        } else {
            moves.clear();
            self.board.generate_moves(&mut moves);
            if attacker {
                moves = self.board.get_tak_threats(&moves, None);
            }
        }
        for mv in moves {
            let rev = self.board.do_move(mv);
            hist.push(mv.to_ptn::<T>());
            let zobrist = self.board.zobrist();
            if !zobrist_hist.insert(zobrist) {
                self.board.reverse_move(rev);
                hist.pop();
                continue;
            }
            if let Some(bounds) = self.bounds_table.get(&zobrist) {
                let status = get_status(bounds, attacker);
                let ch = match status {
                    Status::Tinue => "t",
                    Status::NotTinue => "n",
                    Status::Unknown => "u",
                };
                if ch != "u" {
                    writeln!(writer, "{} 1.{}", hist.join(";"), ch)?;
                    self.rebuild(writer, hist, zobrist_hist, !attacker)?;
                }
            }
            self.board.reverse_move(rev);
            hist.pop();
        }
        Ok(())
    }
}

enum Status {
    Tinue,
    NotTinue,
    Unknown,
}

impl std::ops::Not for Status {
    type Output = Status;

    fn not(self) -> Self::Output {
        match self {
            Self::Tinue => Self::NotTinue,
            Self::NotTinue => Self::Tinue,
            Self::Unknown => Self::Unknown,
        }
    }
}

fn get_status(bounds: &Bounds, attacker: bool) -> Status {
    let solved = if bounds.phi == INFINITY {
        Status::Tinue
    } else if bounds.phi == 0 {
        Status::NotTinue
    } else {
        Status::Unknown
    };
    if attacker {
        solved
    } else {
        !solved
    }
}

#[derive(Clone, PartialEq, Debug)]
enum AttackerOutcome {
    HasRoad(GameMove),
    TakThreats(Vec<GameMove>),
    NoTakThreats,
}

#[derive(Clone, PartialEq, Debug)]
enum DefenderOutcome {
    CanWin(GameMove),
    Defenses(Vec<GameMove>),
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::board::Board6;
    #[test]
    fn simple() {
        let s = "x2,2,x2,1/x5,1/x,2,x,1,1,1/x,2,x2,1,x/x,2C,x4/x,2,x4 2 6";
        let board = Board6::try_from_tps(s).unwrap();
        dbg!(&board);
        let mut search = TinueSearch::new(board);
        assert!(search.is_tinue().unwrap());
    }
    #[test]
    fn simple2() {
        let s = "1,1,1,1,1112C,1/x,121C,x,1,2,1/1,2,x,12,1S,x/x,2,2,1221S,x,2/x3,121,x2/2,2,2,1,2,x 1 25";
        let s2 =
            "1,1,1,1,1112C,1/x,x,x,1,2,1/1,2,x,12,1S,x/x,2,2,1221S,x,2/x3,121,x2/2,2,2,1,2,x 1 25";
        let board = Board6::try_from_tps(s).unwrap();
        let mut search = TinueSearch::new(board);
        assert!(search.is_tinue().unwrap());
        let mut search2 = TinueSearch::new(Board6::try_from_tps(s2).unwrap());
        assert!(!search2.is_tinue().unwrap());
    }
    #[test]
    fn see_edge_placement_road() {
        let s = "1,x,1S,x3/1,x,1,x3/x6/212,2,22212C,x,1C,x/x2,2,2,222221,x/21,1,x,2,12,x 2 21";
        let board = Board6::try_from_tps(s).unwrap();
        let m = GameMove::try_from_ptn("f1", &board).unwrap();
        let mut search = TinueSearch::new(board);
        assert_eq!(search.tinue_evaluate(0), AttackerOutcome::HasRoad(m));
    }
    #[test]
    fn defender_counterattack() {
        let s = "x3,1C,x2/x,1,x,1,x2/x,1,1,x,1,x/x3,1,x2/x3,1,x2/2C,2,22,x,2,x 1 9";
        let board = Board6::try_from_tps(s).unwrap();
        let mut search = TinueSearch::new(board);
        assert!(!search.is_tinue().unwrap());
    }
}
