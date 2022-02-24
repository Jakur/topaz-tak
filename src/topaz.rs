use crate::eval::Evaluator6;
use anyhow::Result;
use board_game_traits::Position;
use crossbeam_channel::{unbounded, Receiver, Sender};
use getopts::Options;
use std::env;
use std::io::{self, BufRead};
use std::thread;
use std::time::Instant;
use telnet::Event;
use topaz_tak::board::Board6;
use topaz_tak::eval::Weights6;
use topaz_tak::search::{proof::TinueSearch, search, SearchInfo};
use topaz_tak::*;

pub fn main() {
    let args: Vec<String> = env::args().collect();

    if let Some(arg1) = args.get(1) {
        if arg1 == "black" {
            play_game_cmd(false);
        } else if arg1 == "white" {
            play_game_cmd(true);
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
            let examine = vec![("temp", 8)];
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
            let mut rest = String::new();
            for s in args[2..].iter() {
                rest.push_str(s);
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
                    println!("Unable to create game with tps: \n{}\n{}", tps, e);
                    return;
                }
            };
            match game {
                TakGame::Standard5(board) => {
                    let search = crate::search::proof::TinueSearch::new(board);
                    proof_interactive(search).unwrap();
                }
                TakGame::Standard6(board) => {
                    let search = crate::search::proof::TinueSearch::new(board);
                    proof_interactive(search).unwrap();
                }
                TakGame::Standard7(board) => {
                    let search = crate::search::proof::TinueSearch::new(board);
                    proof_interactive(search).unwrap();
                }
                _ => todo!(),
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

fn saved_tps(name: &str) -> Option<&str> {
    let s = match name {
        "alion1" => "2,1221122,1,1,1,2S/1,1,1,x,1C,1111212/x2,2,212,2C,11/2,2,x2,1,1/x3,1,1,x/x2,2,21,x,112S 2 32",
        "alion2" => "2,212221C,2,2,2C,1/1,2,1,1,2,1/12,x,1S,2S,2,1/2,2,2,x2,1/1,2212121S,2,12,1,1S/x,2,2,2,x,1 1 30",
        "alion3" => "x2,1,21,2,2/1,2,21,1,21,2/1S,2,2,2C,2,2/21S,1,121C,x,1,12/2,2,121,1,1,1/2,2,x3,22S 1 27",
        "alion4" => "x,1,x4/2,2,1,1,1,1/2221,x,1,21C,x2/2,2,2C,1,2,x/2,2,1,1,1,2/2,x2,2,x,1 2 18",
        "alion5" => "2,x4,11/x5,221/x,2,2,2,x,221/2,1,12C,1,21C,2/2,x,2,x2,2/x,2,2,2,x,121 1 25",
        "empty6" => "x6/x6/x6/x6/x6/x6 1 1",
        "test5" => "2,2,x2,1/2,2,x,1,1/1221S,1,122221C,x,1/1,12,x,2C,2/1S,2,2,x2 1 20",
        "test7" => concat!("2,2,21S,2,1,1,1/2,1,x,2,1,x,1/2,2,2,2,21112C,121S,x/x2,1112C,2,1,1112S,x/121,22211C,", 
            "1S,1,1,121,1221C/x,2,2,2,1,12,2/2,x3,1,122,x 2 50"),
        "topaz1" => "x2,1,x,1212,x/x,221,2212221211C,2S,x2/x,221,1,2,2,x/221,2,12C,1,2,2/22221S,221S,1,1,2,x/12,x,12,1,1,x 1 44",
        "opening1" => "x2,2,x3/x,2,2,x3/x,1,1,2,2,1/1C,2,12C,1,1,x/x,2,x2,1,1/2,x4,1 1 11", // Avoid falling way behind?
        "opening2" => "2,x5/x3,1,2,x/2,2,221C,12C,1,2/x,2,x,1,1,x/x2,2,1,1,x/x2,2,x,1,1 1 13", // Don't give black initiative for free
        "midgame1" => "2,2,2222221C,x3/2,2,2S,12121S,x,2/2,2,1,1,1,1/x,1S,111112C,1,1,x/1,12112S,x4/x,2,x3,1 1 31", // Tinue avoidance
        "midgame2" => "x4,1,1/1,12S,2,2,1,1/1,1221S,1,21C,1,x/1,21112C,2,1,22221S,2/2,2,2,2S,1,2/x2,21,21,x,2 1 32",
        "midgame3" => "2,1,1,1,x2/x,2,2,1,x2/x,1,2,1C,1,1/x2,2,1112C,12S,2/x,2,2,1,x,1/2,2,x2,1,1 1 17",
        "temp" => "x2,2,112,x,1/2,21S,2,12,x2/x2,2,112,x,2/x2,1112111112C,1,1,1/2,221C,x,1,1,x/2,x,1,2,x2 1 31",
        _ => {return None}
    };
    Some(s)
}

fn search_efficiency(names: &[(&str, usize)], save: bool) -> Result<Vec<usize>> {
    use std::collections::HashMap;
    use std::io::Write;
    let mut vec = Vec::new();
    let old = {
        if let Ok(read_data) = std::fs::read_to_string("node_counts.csv") {
            let map: HashMap<String, usize> = read_data
                .lines()
                .filter_map(|x| {
                    let sp: Vec<_> = x.split(",").collect();
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
        let tps = saved_tps(name).unwrap();
        let mut board = Board6::try_from_tps(tps).unwrap();
        let eval = Weights6::default();
        let mut info = SearchInfo::new(*depth, 10_000_000);
        search(&mut board, &eval, &mut info);
        for idx in 0..36 {
            let dummy_move = GameMove::from_placement(Piece::WhiteFlat, idx);
            let ptn_idx = dummy_move.to_ptn::<Board6>();
            println!("{}: {:?}", ptn_idx, info.hist_moves.square_data(idx));
        }
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

fn proof_interactive<T: TakBoard>(mut search: TinueSearch<T>) -> Result<()> {
    let time = Instant::now();
    let tinue = search.is_tinue().unwrap();
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
            interactive.expand_line(s.split("/").collect());
        }
        // interactive.expand_line(vec!["c1".to_string(), "b1>".to_string()]);
        interactive.print_root();
    }
    Ok(())
}

fn play_game_cmd(mut computer_turn: bool) {
    let mut board = Board6::new();
    let eval = Evaluator6 {};
    while let None = board.game_result() {
        println!("{:?}", &board);
        if computer_turn {
            let mut info = SearchInfo::new(6, 5000);
            search(&mut board, &eval, &mut info);
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
    fn use_time(&self, est_plies: usize, side_to_move: Color) -> u64 {
        let (time_bank, inc) = match side_to_move {
            Color::White => (self.wtime, self.winc),
            Color::Black => (self.btime, self.binc),
        };
        let use_bank = time_bank / (est_plies + 2) as u64 / 1000;
        use_bank + inc / 1000
    }
}

fn play_game_tei(receiver: Receiver<TeiCommand>, init: GameInitializer) -> Result<()> {
    let mut board = init.get_board();
    let mut info = SearchInfo::new(init.max_depth, init.hash_size);
    let mut eval = Weights6::default();

    loop {
        let message = receiver.recv()?;
        match message {
            TeiCommand::Go(s) => {
                let low_flats = std::cmp::min(
                    board.pieces_reserve(Color::White),
                    board.pieces_reserve(Color::Black),
                );
                let est_plies = low_flats * 2;
                let time_left = TimeLeft::new(&s);
                let use_time = time_left.use_time(est_plies, board.side_to_move());
                info = SearchInfo::new(init.max_depth, 0)
                    .take_table(&mut info)
                    .max_time(use_time);
                if board.ply() == 8 || board.ply() == 9 {
                    eval = Weights6::default();
                }
                let res = search(&mut board, &eval, &mut info);
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
                board = init.get_board();
                for m in s.split_whitespace() {
                    if let Some(m) = GameMove::try_from_ptn(m, &board) {
                        board.do_move(m);
                    }
                }
            }
            TeiCommand::NewGame(_size) => {
                if init.add_noise {
                    println!("Adding noise!");
                    eval.add_noise();
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
    println!("option name Komi type spin default 0 min 0 max 12");
    println!("teiok");
}

fn tei_loop() {
    let (sender, r) = unbounded();
    let mut receiver = Some(r);
    let mut buffer = String::new();
    let mut init = GameInitializer::new(1_000_000, 8, 0, true);
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
                thread::spawn(move || {
                    play_game_tei(recv, init).unwrap();
                });
            }
            sender.send(TeiCommand::NewGame(size)).unwrap();
        } else if line.starts_with("setoption") {
            let mut iter = line.split_whitespace();
            let name = iter.nth(2).unwrap();
            let value = iter.nth(1).unwrap();
            if name == "Komi" {
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
    fn get_board(&self) -> Board6 {
        Board6::new().with_komi(self.komi)
    }
}

fn play_game_playtak(server_send: Sender<String>, server_recv: Receiver<TeiCommand>) -> Result<()> {
    const MAX_DEPTH: usize = 8;
    const KOMI: u8 = 0;
    let mut board = Board6::new().with_komi(KOMI);
    let mut info = SearchInfo::new(MAX_DEPTH, 5_000_000);
    let eval = Weights6::default();
    // eval.add_noise();
    // let eval = Evaluator6 {};
    loop {
        let message = server_recv.recv()?;
        match message {
            TeiCommand::Go(_) => {
                let use_time = 15; // Todo better time management
                info = SearchInfo::new(MAX_DEPTH, 0)
                    .take_table(&mut info)
                    .max_time(use_time);
                let res = search(&mut board, &eval, &mut info);
                if let Some(outcome) = res {
                    server_send
                        .send(outcome.best_move().expect("could not find best move!"))
                        .unwrap();
                } else {
                    println!("Something went wrong, search failed!");
                }
            }
            TeiCommand::Position(s) => {
                board = Board6::new().with_komi(KOMI);
                for m in s.split(",") {
                    if let Some(m) = GameMove::try_from_playtak(m, &board) {
                        board.do_move(m);
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

fn playtak_loop(engine_send: Sender<TeiCommand>, engine_recv: Receiver<String>) {
    static OPP: &'static str = "TakticianBot";
    let (user, pass) = playtak_auth().expect("Could not read properly formatted .env file");
    std::thread::spawn(move || {
        let mut com = telnet::Telnet::connect(("playtak.com", 10_000), 2048).unwrap();
        let mut counter = 0;
        let mut played = false;
        let mut waiting_for_engine = false;
        let mut my_color = None;
        let mut game_id = None;
        let mut goal = None;
        // let mut played = true;
        // let mut waiting_for_engine = false;
        // let mut my_color = Some("black".to_string());
        // let mut game_id = Some("481821".to_string());
        // let mut goal = None;
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
                                let login_s = format!("Login {} {}\n", user, pass);
                                com.write(login_s.as_bytes()).unwrap();
                            } else if line.starts_with("Game#") {
                                let rest = line.splitn(2, " ").nth(1);
                                if let Some(rest) = rest {
                                    if rest.starts_with("P") || rest.starts_with("M") {
                                        moves.push(rest.to_string());
                                    } else if rest.starts_with("Over") {
                                        engine_send.send(TeiCommand::Quit).unwrap();
                                        break;
                                    }
                                } else {
                                    dbg!(line);
                                }
                            } else if line.starts_with("Seek new") {
                                if line.contains(OPP) {
                                    goal =
                                        Some(line.split_whitespace().nth(2).unwrap().to_string());
                                    println!("Goal: {:?}", goal);
                                }
                            } else if line.starts_with("Game Start") {
                                game_id = line.split_whitespace().nth(2).map(|x| x.to_string());
                                my_color = line.split_whitespace().nth(7).map(|x| x.to_lowercase());
                                played = true;
                                dbg!(&game_id);
                                dbg!(&my_color);
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
                        if !played {
                            if let Some(ref id) = goal {
                                let s = format!("Accept {}\n", id);
                                com.write(s.as_bytes()).unwrap();
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
