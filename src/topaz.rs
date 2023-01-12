#![allow(clippy::style)]

use crate::Position;
use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver, Sender};
use getopts::Options;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::env;
use std::io::{self, BufRead, BufWriter};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use telnet::Event;
use topaz_tak::board::{Board5, Board6};
use topaz_tak::eval::{Evaluator, Tinue6, Weights5, Weights6, NNUE6};
use topaz_tak::search::book;
use topaz_tak::search::{proof::TinueSearch, search, SearchInfo};
use topaz_tak::*;

lazy_static! {
    static ref SAVED_TPS: HashMap<&'static str, &'static str> = {
        let x = [
        ("alion1", "2,1221122,1,1,1,2S/1,1,1,x,1C,1111212/x2,2,212,2C,11/2,2,x2,1,1/x3,1,1,x/x2,2,21,x,112S 2 32"),
        ("alion2", "2,212221C,2,2,2C,1/1,2,1,1,2,1/12,x,1S,2S,2,1/2,2,2,x2,1/1,2212121S,2,12,1,1S/x,2,2,2,x,1 1 30"),
        ("alion3", "x2,1,21,2,2/1,2,21,1,21,2/1S,2,2,2C,2,2/21S,1,121C,x,1,12/2,2,121,1,1,1/2,2,x3,22S 1 27"),
        ("alion4", "x,1,x4/2,2,1,1,1,1/2221,x,1,21C,x2/2,2,2C,1,2,x/2,2,1,1,1,2/2,x2,2,x,1 2 18"),
        ("alion5", "2,x4,11/x5,221/x,2,2,2,x,221/2,1,12C,1,21C,2/2,x,2,x2,2/x,2,2,2,x,121 1 25"),
        ("empty6", "x6/x6/x6/x6/x6/x6 1 1"),
        ("test5", "2,2,x2,1/2,2,x,1,1/1221S,1,122221C,x,1/1,12,x,2C,2/1S,2,2,x2 1 20"),
        ("test7", concat!("2,2,21S,2,1,1,1/2,1,x,2,1,x,1/2,2,2,2,21112C,121S,x/x2,1112C,2,1,1112S,x/121,22211C,",
            "1S,1,1,121,1221C/x,2,2,2,1,12,2/2,x3,1,122,x 2 50")),
        ("topaz1", "x2,1,x,1212,x/x,221,2212221211C,2S,x2/x,221,1,2,2,x/221,2,12C,1,2,2/22221S,221S,1,1,2,x/12,x,12,1,1,x 1 44"),
        ("opening1", "x2,2,x3/x,2,2,x3/x,1,1,2,2,1/1C,2,12C,1,1,x/x,2,x2,1,1/2,x4,1 1 11"), // Avoid falling way behind?
        ("opening2", "2,x5/x3,1,2,x/2,2,221C,12C,1,2/x,2,x,1,1,x/x2,2,1,1,x/x2,2,x,1,1 1 13"), // Don't give black initiative for free
        ("midgame1", "2,2,2222221C,x3/2,2,2S,12121S,x,2/2,2,1,1,1,1/x,1S,111112C,1,1,x/1,12112S,x4/x,2,x3,1 1 31"), // Tinue avoidance
        ("midgame2", "x4,1,1/1,12S,2,2,1,1/1,1221S,1,21C,1,x/1,21112C,2,1,22221S,2/2,2,2,2S,1,2/x2,21,21,x,2 1 32"),
        ("midgame3", "2,1,1,1,x2/x,2,2,1,x2/x,1,2,1C,1,1/x2,2,1112C,12S,2/x,2,2,1,x,1/2,2,x2,1,1 1 17"),
        ("temp", "x6/x6/x3,2,2,1/x3,2,1,1/x,2,x2,1,1/2,x4,1 2 6"),
        ("start", "x6/x2,2,2,x2/x6/x6/x6/1,x3,1,x 1 3"),
        ("start2", "2,x5/x6/x6/x6/x2,1,x3/1,x5 2 2"),
        ("start3", "x6/x4,2,1/x2,2,2C,1,2/x2,2,x,1,1/x5,1/x6 1 6"),
        ("start4", "2,x4,1/x4,1,1/x2,2,21C,12C,x/1,1,1,2,1,1/x2,2,2,2,2/x6 2 11"),
        ("tt1", "1,x4,2/1,12,1,1,1,2/2,12C,21C,2,21,1/1,1,2,2,2,1/1,x3,2,2/x6 2 16"),
        ("endgame1", "1,2,1,1,1S,212212/112,22,x,1,1S,2/1,2,212C,2,1112S,x/x,2,1,1,12221C,2/1,1S,1S,12,x,22121S/1,12,1,2,x,2 1 46"),
        ("endgame2", "2,x,2,2,1,1/1,2,2,1,12,1/1,2112S,x,1,2,1/21,2,2221S,2,2112C,2/2,121,1,2S,11221C,1/12,222221S,12,1,1,1 1 43"),
        ("endgame3", "x2,21,122,1121S,112S/1S,x,1112,x,2S,x/112C,2S,x,1222221C,2,x/2,x2,1,2121S,x/112,1112111112S,x3,221S/2,2,x2,21,2 1 56"),
        ].into_iter().collect();
        x
    };
}

pub fn main() {
    let args: Vec<String> = env::args().collect();

    if let Some(arg1) = args.get(1) {
        if arg1 == "black" {
            play_game_cmd(false);
        } else if arg1 == "white" {
            play_game_cmd(true);
        } else if arg1 == "analyze" {
            let game = build_tps(&args[2..]).unwrap();
            let mut info = SearchInfo::new(20, 2 << 20);
            match game {
                TakGame::Standard5(mut board) => {
                    let mut eval = Weights5::default();
                    search(&mut board, &mut eval, &mut info);
                }
                TakGame::Standard6(mut board) => {
                    // let mut eval = Weights6::default();
                    let mut board = board.with_komi(0);
                    let mut eval = eval::NNUE6::new();
                    search(&mut board, &mut eval, &mut info);
                }
                _ => todo!(),
            }
            return;
        } else if arg1 == "magic" {
            gen_magics();
            return;
        } else if arg1 == "order" {
            let vals: Vec<_> = SAVED_TPS
                .values()
                .filter_map(|tps| {
                    if Board6::try_from_tps(tps).is_ok() {
                        Some((*tps, 12))
                    } else {
                        None
                    }
                })
                .collect();
            let nodes = search_efficiency(&vals, true).unwrap();
            dbg!(nodes.into_iter().sum::<usize>());
            return;
        } else if arg1 == "book" {
            let book = Arc::new(Mutex::new(book::Book::new(
                search::book::BookMode::Learn,
                HashMap::new(),
            )));
            let count = std::thread::available_parallelism().unwrap().get();
            assert!(count >= 1_usize);
            let openings = vec![&["a1", "f6"], &["a1", "a6"]];
            let mut handles = vec![];
            for (t_id, start) in openings.into_iter().enumerate() {
                let book = Arc::clone(&book);
                println!("Thread: {}", t_id);
                let handle = std::thread::spawn(move || {
                    for i in 0..15 {
                        let data = { book.lock().unwrap().clone() };
                        let (moves, game_res) = play_book_game(start, data);
                        println!("T{} Game {}", i, t_id);
                        let combo: Vec<_> = moves
                            .iter()
                            .copied()
                            .map(|m| m.to_ptn::<Board6>())
                            .collect();
                        let s = combo.join(" ");
                        println!("{}", s);
                        book.lock()
                            .unwrap()
                            .update(Board6::new().with_komi(4), game_res, moves);
                    }
                });
                handles.push(handle);
            }
            if !handles.into_iter().all(|h| h.join().is_ok()) {
                println!("Something went wrong, thread panicked or lock poisoned?");
            }
            book.lock()
                .unwrap()
                .save(&mut std::fs::File::create("book6s.json").unwrap())
                .expect("Failed to write book");
        } else if arg1 == "selfplay" {
            let mut moves = Vec::new();
            let mut game = build_tps(&args[2..]).unwrap();
            let mut info = SearchInfo::new(20, 2 << 20).time_bank(TimeBank::flat(12_000));
            while game.game_result().is_none() {
                match game {
                    TakGame::Standard5(ref mut board) => {
                        let mut eval = Weights5::default();
                        let outcome = search(board, &mut eval, &mut info).unwrap();
                        println!("{}", outcome.best_move().unwrap());
                        let mv = outcome.next().unwrap();
                        board.do_move(mv);
                        info = SearchInfo::new(20, 0)
                            .time_bank(TimeBank::flat(12_000))
                            .take_table(&mut info)
                            .quiet(true);
                        moves.push(mv);
                    }
                    TakGame::Standard6(ref mut board) => {
                        // let mut eval = Weights6::default();
                        let mut eval = NNUE6::default();
                        let outcome = search(board, &mut eval, &mut info).unwrap();
                        println!("{}", outcome.best_move().unwrap());
                        let mv = outcome.next().unwrap();
                        board.do_move(mv);
                        info = SearchInfo::new(20, 0)
                            .time_bank(TimeBank::flat(12_000))
                            .take_table(&mut info)
                            .quiet(true);
                        moves.push(mv);
                    }
                    _ => todo!(),
                }
            }
            let s = match game {
                TakGame::Standard5(ref board) => {
                    format!("{:?}", board)
                }
                TakGame::Standard6(ref board) => {
                    format!("{:?}", board)
                }
                _ => todo!(),
            };
            println!("Final Position: {}", s);
            match game {
                TakGame::Standard5(_) => {
                    for mv in moves {
                        print!("{} ", mv.to_ptn::<Board5>()); // Todo fix
                    }
                }
                TakGame::Standard6(_) => {
                    for mv in moves {
                        print!("{} ", mv.to_ptn::<Board6>()); // Todo fix
                    }
                }
                _ => todo!(),
            }
            println!("{:?}", game.game_result());
            return;
        } else if arg1 == "test" {
            let time = Instant::now();
            // let s = "2,x4,1/x4,1,x/x,2,12C,1,1,x/x,1,2,21C,x2/x,2,2,x3/x2,2,1,x2 1 10";
            // let s = "1,1,2,21,2,2/12,1111112C,2,1S,2,121S/1,x,221C,2,212S,21/1,2,x,2,1S,x/1,112S,2,1,1,1/1,x,112,112S,2,2 1 39";
            // let mut board = Board6::try_from_tps(s).unwrap();
            // let eval = Weights6::default();
            // let mut info = SearchInfo::new(4, 1_000_000);
            // search(&mut board, &eval, &mut info);
            // let pv_move = info.pv_move(&board).unwrap();
            // println!("Computer Choose: {}", pv_move.to_ptn::<Board6>());
            // info.print_cuts();
            // let node_counts = search_efficiency(&["empty6"], 8);
            let examine = vec![("opening1", 12)];
            // let mut board = Board6::try_from_tps(saved_tps("start4").unwrap()).unwrap();
            // let mut eval = Weights6::default();
            // eval.add_noise();
            // eval.evaluate(&board, 0);
            // board.do_move(GameMove::try_from_ptn("d3", &board).unwrap());
            let node_counts = search_efficiency(&examine, false).unwrap();
            // let node_counts =
            //     search_efficiency(&[("opening1", 8), ("opening2", 8), ("midgame1", 6)], false)
            //         .unwrap();
            // let node_counts = search_efficiency(&["alion1", "alion2", "alion3", "topaz1"], 6);
            let nodes: usize = node_counts.into_iter().sum();
            println!(
                "Nodes: {} Nps: {}",
                nodes,
                1000 * nodes as u128 / time.elapsed().as_millis()
            );
            println!("Time: {} ms", time.elapsed().as_millis());
            return;
        } else if arg1 == "tinue" {
            let mut attacker = true;
            let mut nodes = usize::MAX;
            let mut svg = true;
            let game = if let Ok(g) = build_tps(&args[2..]) {
                g
            } else {
                let mut opts = Options::new();
                opts.optopt("t", "tps", "Set position tps", "TPS");
                opts.optopt("n", "nodes", "Sets the node limit for the search", "NODES");
                opts.optflag("h", "help", "Print this help menu");
                opts.optflag("d", "defender", "Set flag if it's the defender's move");
                opts.optflag("i", "interactive", "Use cmd interface instead of svg");
                // let join = args.join(" ");
                // let data = join.split("tinue ").nth(1).unwrap();
                // let vals: Vec<_> = data.split(";").collect();
                let matches = match opts.parse(&args[2..]) {
                    Ok(m) => m,
                    Err(f) => {
                        panic!("{}", f.to_string())
                    }
                };
                if matches.opt_present("h") {
                    println!("{}", opts.usage(""));
                    return;
                }
                dbg!(&args);
                let t = matches.opt_str("t").and_then(|x| {
                    let slice: Vec<_> = x.split_whitespace().collect();
                    build_tps(&slice).ok()
                });
                // dbg!(&tps);
                nodes = matches
                    .opt_str("n")
                    .map(|x| x.trim().parse().unwrap())
                    .unwrap_or(usize::MAX);
                attacker = !matches.opt_present("d");
                svg = !matches.opt_present("i");
                if let Some(game) = t {
                    game // If the game was sent as a single quoted arg
                } else {
                    // If tps is sent as three space separated args
                    args.windows(3).find_map(|s| build_tps(s).ok()).unwrap()
                }
            };
            let res = match game {
                TakGame::Standard5(board) => {
                    let search = crate::search::proof::TinueSearch::new(board)
                        .attacker(attacker)
                        .limit(nodes);
                    proof_interactive(search, svg)
                }
                TakGame::Standard6(board) => {
                    let search = crate::search::proof::TinueSearch::new(board.clone())
                        .attacker(attacker)
                        .limit(nodes);
                    proof_interactive(search, svg)
                }
                TakGame::Standard7(board) => {
                    let search = crate::search::proof::TinueSearch::new(board)
                        .attacker(attacker)
                        .limit(nodes);
                    proof_interactive(search, svg)
                }
                _ => todo!(),
            };
            if res.is_err() {
                println!("Search failed, perhaps max_nodes was exceeded?");
            }
            return;
        } else if arg1 == "playtak" {
            let (s1, r1) = unbounded();
            let (s2, r2) = unbounded();
            playtak_loop(s1, r2);
            play_game_playtak(s2, r1).unwrap();
            return;
        } else {
            println!("Unknown argument: {}", arg1);
        }
    }
    let mut buffer = String::new();
    std::io::stdin()
        .read_line(&mut buffer)
        .expect("Could not read line");
    if buffer.trim() == "tei" {
        // let (s, r) = unbounded();
        tei_loop();
        // identify();
        // let _ = play_game_tei(r);
    } else if buffer == "play white" {
        play_game_cmd(true)
    } else if buffer == "play black" {
        play_game_cmd(false)
    } else {
        println!("Unknown command: {}", buffer);
    }
}

fn play_book_game(st: &[&str], book: book::Book) -> (Vec<GameMove>, GameResult) {
    let mut eval = Weights6::default();
    let mut moves = Vec::new();
    let mut board = Board6::new().with_komi(4);
    for mv in st {
        let mv = GameMove::try_from_ptn(mv, &board).unwrap();
        moves.push(mv);
        board.do_move(mv);
    }
    let mut info = SearchInfo::new(20, 2 << 20)
        .time_bank(TimeBank::flat(12_000))
        .book(book)
        .quiet(true);
    while board.game_result().is_none() {
        let outcome = search(&mut board, &mut eval, &mut info).unwrap();
        // println!("{}", outcome.best_move().unwrap());
        let mv = outcome.next().unwrap();
        board.do_move(mv);
        info = SearchInfo::new(20, 0)
            .time_bank(TimeBank::flat(12_000))
            .take_table(&mut info)
            .take_book(&mut info)
            .quiet(true);
        moves.push(mv);
    }
    return (moves, board.game_result().unwrap());
}

fn build_tps<S: AsRef<str>>(args: &[S]) -> Result<TakGame> {
    let mut rest = String::new();
    for s in args.iter() {
        rest.push_str(s.as_ref());
        rest.push_str(" ");
    }
    rest.pop();
    let tps = if let Some(tps) = saved_tps(rest.as_str()) {
        tps
    } else {
        rest.as_str()
    };
    let game = match TakGame::try_from_tps(tps) {
        Ok(b) => b,
        Err(e) => {
            let message = format!("Unable to create game with tps: \n{}\n{}", tps, e);
            return Err(anyhow::anyhow!(message));
        }
    };
    Ok(game)
}

fn saved_tps(name: &str) -> Option<&str> {
    SAVED_TPS.get(name).map(|x| *x)
}

fn search_efficiency(names: &[(&str, usize)], save: bool) -> Result<Vec<usize>> {
    use std::io::Write;
    let mut vec = Vec::new();
    let old = {
        if let Ok(read_data) = std::fs::read_to_string("node_counts.csv") {
            let map: HashMap<String, usize> = read_data
                .lines()
                .filter_map(|x| {
                    let sp: Vec<_> = x.split(',').collect();
                    if sp.len() == 3 {
                        Some((sp[0].to_string(), sp[2].parse().unwrap()))
                    } else {
                        None
                    }
                })
                .collect();
            Some(map)
        } else {
            None
        }
    };
    for (name, depth) in names {
        let tps = if let Some(tps) = saved_tps(name) {
            tps
        } else {
            name
        };
        let mut board = Board6::try_from_tps(tps).unwrap();
        let mut eval = Weights6::default();
        let mut info = SearchInfo::new(*depth, 2 << 20);
        search(&mut board, &mut eval, &mut info);
        dbg!(info.stats);
        // for idx in 0..36 {
        //     let dummy_move = GameMove::from_placement(Piece::WhiteFlat, idx);
        //     let ptn_idx = dummy_move.to_ptn::<Board6>();
        //     println!("{}: {:?}", ptn_idx, info.hist_moves.square_data(idx));
        // }
        vec.push(info.nodes);
        if let Some(old_nodes) = old.as_ref().and_then(|x| x.get(*name)) {
            let diff = (info.nodes as f64 - *old_nodes as f64) / *old_nodes as f64;
            println!("{}: {}", name, diff);
        }
    }
    if save {
        let mut f = std::fs::File::create("node_counts.csv")?;
        for ((name, depth), nodes) in names.iter().zip(vec.iter()) {
            write!(&mut f, "{},{},{}\n", name, depth, nodes)?;
        }
    }
    Ok(vec)
}

fn proof_interactive<T: TakBoard>(mut search: TinueSearch<T>, svg: bool) -> Result<()> {
    let time = Instant::now();
    let tinue = search
        .is_tinue()
        .ok_or_else(|| anyhow::anyhow!("Search ended improperly"))?;
    if tinue {
        println!("Tinue Found!")
    } else {
        println!("No Tinue Found.");
    }
    let pv = search.principal_variation();
    for m in pv.into_iter().map(|m| m.to_ptn::<T>()) {
        println!("{}", m);
    }

    let seconds = time.elapsed().as_secs();
    println!("Done in {} seconds", seconds);
    if svg {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open("proof-data.txt")?;
        let mut file = BufWriter::new(file);
        let mut hist = Vec::new();
        let mut zobrist_hist = std::collections::HashSet::new();
        search
            .rebuild(
                &mut file,
                &mut hist,
                &mut zobrist_hist,
                search.is_attacker(),
            )
            .unwrap();
    } else {
        let mut interactive = crate::search::proof::InteractiveSearch::new(search);
        let mut first = true;
        interactive.print_root();
        loop {
            let mut opts = Options::new();
            let mut buffer = String::new();
            // opts.optopt("o", "", "set output file name", "NAME");
            opts.optopt("m", "move", "Move the root of the tree", "PTN/PTN");
            opts.optopt(
                "e",
                "expand",
                "Expand the tree of a certain move",
                "PTN/PTN",
            );
            // opts.optflag("v", "verbose", "Expand all children, even explored ones");
            opts.optflag("h", "help", "Print the help text");
            opts.optflag("q", "quit", "Quit");
            opts.optflag(
                "r",
                "reset",
                "Resets the view back to the default root view",
            );
            if first {
                println!("{}", opts.usage(""));
                first = false;
            }
            io::stdin().lock().read_line(&mut buffer)?;
            let matches = opts.parse(buffer.split_whitespace())?;
            if matches.opt_present("q") {
                break;
            }
            if matches.opt_present("h") {
                println!("{}", opts.usage(""));
                continue;
            }
            if matches.opt_present("r") {
                interactive.reset_expansion();
                interactive.reset_view();
            }
            if let Some(v) = matches.opt_str("m") {
                let res = interactive.change_view(&v);
                if res.is_err() {
                    println!("Failed to change view, resetting to default!");
                    interactive.reset_view();
                }
            }
            if let Some(s) = matches.opt_str("e") {
                interactive.expand_line(s.split('/').collect());
            }
            // interactive.expand_line(vec!["c1".to_string(), "b1>".to_string()]);
            interactive.print_root();
        }
    }

    Ok(())
}

fn play_game_cmd(mut computer_turn: bool) {
    let mut board = Board6::new();
    let mut eval = Weights6::default();
    while let None = board.game_result() {
        println!("{:?}", &board);
        if computer_turn {
            let mut info = SearchInfo::new(6, 5000);
            search(&mut board, &mut eval, &mut info);
            let pv_move = info.pv_move(&board).unwrap();
            println!("Computer Choose: {}", pv_move.to_ptn::<Board6>());
            board.do_move(pv_move);
        } else {
            let stdin = io::stdin();
            let line = stdin.lock().lines().next().unwrap().unwrap();
            if line == "q" {
                return;
            } else if line == "tps" {
                println!("{:?}", &board);
                continue;
            }
            let ptn_move = GameMove::try_from_ptn(&line, &board).unwrap();
            if !board.legal_move(ptn_move) {
                println!("Illegal Move Attempted!");
                continue;
            }
            board.do_move(ptn_move);
        }
        computer_turn = !computer_turn;
    }
}

struct TimeLeft {
    wtime: u64,
    btime: u64,
    winc: u64,
    binc: u64,
}

impl TimeLeft {
    pub fn new(tei_str: &str) -> Self {
        let mut ret = Self {
            wtime: 1000,
            btime: 1000,
            winc: 0,
            binc: 0,
        };
        for (field, val) in tei_str
            .split_whitespace()
            .zip(tei_str.split_whitespace().skip(1))
        {
            match (field, val.parse()) {
                ("wtime", Ok(val)) => ret.wtime = val,
                ("btime", Ok(val)) => ret.btime = val,
                ("winc", Ok(val)) => ret.winc = val,
                ("binc", Ok(val)) => ret.binc = val,
                _ => {}
            }
        }
        ret
    }
    fn has_time(&self, color: Color) -> (u64, u64) {
        let (time_bank, inc) = match color {
            Color::White => (self.wtime, self.winc),
            Color::Black => (self.btime, self.binc),
        };
        (time_bank, inc)
    }
}

fn play_game_tei<E: Evaluator + Default>(
    receiver: Receiver<TeiCommand>,
    init: GameInitializer,
) -> Result<()> {
    let (mut board, mut eval) = init.get_board::<E>();
    let mut info = SearchInfo::new(init.max_depth, init.hash_size);
    // let mut eval = Box::new(crate::eval::Weights5::default());

    loop {
        let message = receiver.recv()?;
        match message {
            TeiCommand::Go(s) => {
                let flats_left =
                    board.pieces_reserve(Color::White) + board.pieces_reserve(Color::Black);
                let time_left = TimeLeft::new(&s);
                let (clock_time, clock_inc) = time_left.has_time(board.side_to_move());
                let use_time = TimeBank::init(flats_left as u64, clock_time, clock_inc);
                info = SearchInfo::new(init.max_depth, 0)
                    .take_table(&mut info)
                    .time_bank(use_time)
                    .abort_depth(8);
                if board.ply() == 8 || board.ply() == 9 {
                    eval = E::default();
                }
                let res = search(&mut board, &mut eval, &mut info);
                if let Some(outcome) = res {
                    println!("info {}", outcome);
                    println!(
                        "bestmove {}",
                        outcome
                            .best_move()
                            .expect("Could not find best move!")
                            .trim_end_matches('*')
                    );
                } else {
                    println!("Something went wrong, search failed!");
                }
            }
            TeiCommand::Position(s) => {
                board = init.get_board::<E>().0;
                for m in s.split_whitespace() {
                    if let Some(m) = GameMove::try_from_ptn(m, &board) {
                        board.do_move(m);
                    }
                }
            }
            TeiCommand::NewGame(_size) => {
                info.clear_tt();
                if init.add_noise {
                    cfg_if::cfg_if! {
                        if #[cfg(feature = "random")] {
                            println!("Adding noise!");
                            // eval.add_noise();
                        } else {
                            panic!("Unable to add noise because no rng was compiled!");
                        }
                    }
                }
            }
            TeiCommand::Quit => {
                break;
            }
            _ => println!("Unknown command: {:?}", message),
        }
    }
    Ok(())
}

fn identify() {
    println!("id name Topaz");
    println!("id author Justin Kur");
    println!("option name HalfKomi type spin default 0 min 0 max 12");
    println!("teiok");
}

fn tei_loop() {
    let (sender, r) = unbounded();
    let mut receiver = Some(r);
    let mut buffer = String::new();
    let mut init = GameInitializer::new(2 << 22, 80, 0, false);
    identify();
    loop {
        std::io::stdin()
            .read_line(&mut buffer)
            .expect("Could not read line");
        let line = buffer.trim();
        if line == "tei" {
            identify();
        } else if line == "isready" {
            println!("readyok");
        } else if line == "quit" {
            sender.send(TeiCommand::Quit).unwrap();
            break;
        } else if line == "stop" {
            sender.send(TeiCommand::Stop).unwrap();
        } else if line.starts_with("position") {
            sender.send(TeiCommand::Position(line.to_string())).unwrap();
        } else if line.starts_with("go") {
            sender.send(TeiCommand::Go(line.to_string())).unwrap();
        } else if line.starts_with("teinewgame") {
            let size = line
                .split_whitespace()
                .nth(1)
                .expect("Bad new game command?")
                .parse()
                .expect("Failed to parse size!");
            if let Some(recv) = receiver.take() {
                let init = init.clone();
                thread::spawn(move || match size {
                    5 => play_game_tei::<Weights5>(recv, init).unwrap(),
                    6 => play_game_tei::<eval::NNUE6>(recv, init).unwrap(),
                    // 6 => play_game_tei::<Weights6>(recv, init).unwrap(),
                    _ => unimplemented!(),
                });
            }
            sender.send(TeiCommand::NewGame(size)).unwrap();
        } else if line.starts_with("setoption") {
            let mut iter = line.split_whitespace();
            let name = iter.nth(2).unwrap();
            let value = iter.nth(1).unwrap();
            if name == "HalfKomi" {
                init.komi = value.parse().unwrap();
                println!("Setting komi to {}", init.komi);
            }
        } else {
            println!("Unknown Tei Command: {}", buffer);
        }
        buffer.clear();
    }
}

#[derive(Clone)]
struct GameInitializer {
    hash_size: usize,
    max_depth: usize,
    komi: u8,
    add_noise: bool,
}

impl GameInitializer {
    fn new(hash_size: usize, max_depth: usize, komi: u8, add_noise: bool) -> Self {
        Self {
            hash_size,
            max_depth,
            komi,
            add_noise,
        }
    }
    fn get_board<E: Evaluator + Default>(&self) -> (E::Game, E) {
        (E::Game::start_position().with_komi(self.komi), E::default())
        // crate::board::Board5::new().with_komi(self.komi)
    }
}

const PLAYTAK_KOMI: u8 = 4;

fn play_game_playtak(server_send: Sender<String>, server_recv: Receiver<TeiCommand>) -> Result<()> {
    const MAX_DEPTH: usize = 32;
    const KOMI: u8 = PLAYTAK_KOMI;
    const MAX_OPENING_LENGTH: usize = 0;
    let mut move_cache = Vec::new();
    let mut board = Board6::new().with_komi(KOMI);
    let mut info = SearchInfo::new(MAX_DEPTH, 2 << 22);
    let book = playtak_book();
    // let mut eval = Weights6::default();
    let mut eval = crate::eval::NNUE6::default();
    // eval.add_noise();
    // let eval = Evaluator6 {};
    'outer: loop {
        let message = server_recv.recv()?;
        match message {
            TeiCommand::Go(_) => {
                if let Some(ref book) = book {
                    if board.ply() <= MAX_OPENING_LENGTH {
                        let symmetries = board.symmetries();
                        for (sym_idx, pos) in symmetries.into_iter().enumerate() {
                            if let Some(mv_str) = book.get(&pos.hash()) {
                                let mv = GameMove::try_from_ptn(mv_str.as_str(), &pos);
                                if let Some(mv) = mv {
                                    let fixed_mv = mv.reverse_symmetry::<Board6>(sym_idx);
                                    if board.legal_move(fixed_mv) {
                                        server_send.send(fixed_mv.to_ptn::<Board6>()).unwrap();
                                        continue 'outer;
                                    }
                                }
                            }
                        }
                    }
                }
                let use_time = 12_000; // Todo better time management
                info = SearchInfo::new(MAX_DEPTH, 0)
                    .take_table(&mut info)
                    .take_book(&mut info)
                    .time_bank(TimeBank::flat(use_time))
                    .abort_depth(50);
                let res = search(&mut board, &mut eval, &mut info);
                if let Some(outcome) = res {
                    server_send
                        .send(outcome.best_move().expect("could not find best move!"))
                        .unwrap();
                } else {
                    println!("Something went wrong, search failed!");
                }
            }
            TeiCommand::Position(s) => {
                move_cache.clear();
                board = Board6::new().with_komi(KOMI);
                for m in s.split(',') {
                    if let Some(m) = GameMove::try_from_playtak(m, &board) {
                        move_cache.push(m);
                        board.do_move(m);
                    }
                }
            }
            TeiCommand::Quit => {
                break;
            }
            TeiCommand::NewGame(_size) => {
                // Record poor opening outcome
                // if last_score < -400 {
                //     if let Some(side) = my_side {
                //         let res = match side {
                //             Color::White => GameResult::BlackWin,
                //             Color::Black => GameResult::WhiteWin,
                //         };
                //         if let Some(op_book) = info.book_mut() {
                //             op_book.update(Board6::new().with_komi(KOMI), res, move_cache);
                //             save_playtak_book(op_book)?;
                //         }
                //     }
                // } else if last_score == 0 && board.move_num() >= 30 {
                //     let res = GameResult::Draw;
                //     if let Some(op_book) = info.book_mut() {
                //         op_book.update(Board6::new().with_komi(KOMI), res, move_cache);
                //         save_playtak_book(op_book)?;
                //     }
                // }
                info.clear_tt();
                board = Board6::new().with_komi(KOMI);
                move_cache = Vec::new();
            }
            _ => println!("Unknown command: {:?}", message),
        }
    }
    Ok(())
}

fn playtak_loop(engine_send: Sender<TeiCommand>, engine_recv: Receiver<String>) {
    // let mut opp = "TakticianBot";
    let mut opp = "WilemBot";
    let login_s = if let Some((user, pass)) = playtak_auth() {
        format!("Login {} {}\n", user, pass)
    } else {
        "Login Guest".to_string()
    };
    std::thread::spawn(move || {
        let mut com = telnet::Telnet::connect(("playtak.com", 10_000), 2048).unwrap();
        let mut counter = 0;
        let mut playing = false;
        let mut waiting_for_engine = false;
        let mut my_color = None;
        let mut game_id = None;
        let mut goal = None;
        let mut live_seek = false;

        let mut moves = Vec::new();
        loop {
            match com.read_nonblocking() {
                Ok(event) => match event {
                    Event::Data(buffer) => {
                        let s = std::str::from_utf8(&(*buffer)).unwrap();
                        print!("{}", s);
                        for line in s.lines() {
                            if line.starts_with("Welcome!") {
                                println!("Logging in");
                                com.write(login_s.as_bytes()).unwrap();
                            } else if line.starts_with("Game#") {
                                let rest = line.split_once(' ').map(|x| x.1);
                                if let Some(rest) = rest {
                                    if rest.starts_with('P') || rest.starts_with('M') {
                                        moves.push(rest.to_string());
                                    } else if rest.starts_with("Over") {
                                        // Todo maybe a different command?
                                        engine_send.send(TeiCommand::NewGame(0)).unwrap();
                                        counter = 0;
                                        playing = false;
                                        waiting_for_engine = false;
                                        my_color = None;
                                        game_id = None;
                                        goal = None;
                                        live_seek = false;
                                        moves.clear();
                                    }
                                } else {
                                    dbg!(line);
                                }
                            } else if line.starts_with("Seek new") {
                                if line.contains(opp) {
                                    goal =
                                        Some(line.split_whitespace().nth(2).unwrap().to_string());
                                    println!("Goal: {:?}", goal);
                                }
                            } else if line.starts_with("Game Start") {
                                engine_send.send(TeiCommand::NewGame(0)).unwrap();
                                game_id = line.split_whitespace().nth(2).map(|x| x.to_string());
                                my_color = line.split_whitespace().nth(7).map(|x| x.to_lowercase());
                                playing = true;
                                dbg!(&game_id);
                                dbg!(&my_color);
                                opp = "NONE";
                            }
                        }
                    }
                    Event::NoData => {
                        thread::sleep(std::time::Duration::from_secs(1));
                        counter += 1;
                        if counter >= 30 {
                            counter = 0;
                            println!("Pinging server!");
                            com.write("PING\n".as_bytes()).unwrap();
                        }
                        if !playing {
                            if let Some(ref id) = goal {
                                let s = format!("Accept {}\n", id);
                                com.write(s.as_bytes()).unwrap();
                                goal = None;
                            } else if !live_seek && counter >= 5 {
                                let s = format!("Seek 6 900 30 A {} 30 1 0 0 \n", PLAYTAK_KOMI);
                                com.write(s.as_bytes()).unwrap();
                                live_seek = true;
                            }
                        } else {
                            let color = my_color.as_ref().unwrap().as_str();

                            let my_turn = match color {
                                "white" => moves.len() % 2 == 0,
                                "black" => moves.len() % 2 == 1,
                                _ => panic!("Weird color?"),
                            };
                            if waiting_for_engine {
                                if let Ok(response) = engine_recv.try_recv() {
                                    let color = if moves.len() % 2 == 0 {
                                        Color::White
                                    } else {
                                        Color::Black
                                    };
                                    let m = GameMove::try_from_ptn_m(&response, 6, color).unwrap();
                                    let playtak_m = m.to_playtak::<Board6>();
                                    let message = format!(
                                        "Game#{} {}\n",
                                        game_id.as_ref().unwrap(),
                                        playtak_m
                                    );
                                    println!("Sending: {}", message);
                                    com.write(message.as_bytes()).unwrap();
                                    waiting_for_engine = false;
                                    moves.push(playtak_m);
                                    thread::sleep(std::time::Duration::from_secs(1));
                                }
                                continue;
                            }
                            if my_turn {
                                engine_send
                                    .send(TeiCommand::Position(moves.join(",")))
                                    .unwrap();
                                engine_send.send(TeiCommand::Go("".to_string())).unwrap();
                                waiting_for_engine = true;
                            }
                        }
                    }
                    _ => {
                        println!("Got here");
                        dbg!(event);
                        break;
                    }
                },
                Err(e) => {
                    dbg!(e);
                    break;
                }
            }
            // println!("Thread sleepy!");
        }
    });
}

fn playtak_auth() -> Option<(String, String)> {
    dotenv::dotenv().ok()?;
    let mut username = None;
    let mut password = None;
    for (key, value) in env::vars() {
        if key == "PLAYTAK_USERNAME" {
            username = Some(value);
        } else if key == "PLAYTAK_PASSWORD" {
            password = Some(value);
        }
    }
    Some((username?, password?))
}

fn playtak_book() -> Option<HashMap<u64, String>> {
    use std::io::Read;
    dotenv::dotenv().ok()?;
    let mut pt_book = None;
    for (key, value) in env::vars() {
        if key == "PLAYTAK_BOOK" {
            pt_book = Some(value);
        }
    }
    let pt_book = pt_book?;
    let mut map = HashMap::new();
    if let Ok(mut file) = std::fs::File::open(&pt_book) {
        let mut book_data = String::new();
        file.read_to_string(&mut book_data).ok()?;
        for line in book_data.lines() {
            let mut split = line.split(",");
            let hash: u64 = split.next()?.parse().ok()?;
            let ptn = split.next()?.to_string();
            map.insert(hash, ptn);
        }
        Some(map)
    } else {
        None
    }
}

fn save_playtak_book(book: &book::Book) -> Result<()> {
    use anyhow::anyhow;
    use miniserde::json;
    use std::io::Write;
    dotenv::dotenv()?;
    let mut pt_book = None;
    for (key, value) in env::vars() {
        if key == "PLAYTAK_BOOK" {
            pt_book = Some(value);
        }
    }
    let pt_book = pt_book.ok_or_else(|| anyhow!("Could not find book path to write to"))?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&pt_book)?;
    let write_string = json::to_string(book);
    write!(&mut file, "{}", write_string)?;
    file.flush()?;
    Ok(())
}

fn gen_magics() {
    use rand_core::SeedableRng;
    let mut seed: [u8; 32] = [0; 32];
    getrandom::getrandom(&mut seed).unwrap();
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::from_seed(seed);
    topaz_tak::move_gen::magic::generate_move_magic6(&mut rng);
}
