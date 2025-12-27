# Topaz Tak

An engine for the board game [Tak](https://en.wikipedia.org/wiki/Tak_(game)). As of December 2025, it is likely the strongest Tak playing engine on consumer hardware. On fast time controls (10+0.1), it performs well against the GPU-based [TakZero](https://github.com/ViliamVadocz/takzero), scoring +344-222=98, +64 elo [+39, +89] with 4 threads against a 3080 GPU. 

## Installation 
1) Clone the repository
2) Download the Topaz network (quantised.bin) from the most recent directory of the following [Google Drive](https://drive.google.com/drive/u/0/folders/1IRMfgvkGH1CW79B184hHs1-ri0bmIPWv). Copy the quantised.bin file into the src directory `src/quantised.bin`.
3) With a Rust compiler, run `cargo build --release`

## Usage 
The program will launch by default in [tei mode](https://github.com/MortenLohne/racetrack). This mode can be used for analysis in conjunction with [ptn.ninja](https://ptn.ninja/). Alternatively, single positions can be analyzed using `./topaz analyze [tps]`. Additionally, the program includes a proof-number solver for Tinue positions, which can be run using `./topaz tinue [tps]` 

### Game
The primary battleground for Tak engines has been 6x6 Tak with 2 Komi for black. Currently, the engine is only optimized for these settings, although 5x5 and 7x7 Tak are playable with weaker search and evaluation functions.  

### Evaluation
Topaz uses an efficiently updatable neural network trained on roughly 1 billion Tak positions generated from self-play using randomized opening positions. 
     
### Search
- [x] Negamax with alpha-beta pruning
- [X] Null Move Pruning 
- [X] Principal Variation Search 
- [x] Late Move Reductions
- [X] Forward and Reverse Futility Pruning
- [X] History Move Orderings
- [X] Static Evaluation Correction History 
- [X] Automated tuning
- [X] Multithreading with LazySMP 
- [X] Multipv Support (Limited)


### Protocol 
- [x] [tei protocol](https://github.com/MortenLohne/racetrack)
- [X] playtak.com server support (Limited)
