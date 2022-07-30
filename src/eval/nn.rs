use vectrix::Matrix;

const LABELS: [&'static str; 27] = [
    "empty",
    "loc_friendly",
    "loc_enemy",
    "flat_friendly",
    "wall_friendly",
    "cap_friendly",
    "flat_enemy",
    "wall_enemy",
    "cap_enemy",
    "cap_status_friendly",
    "cap_status_enemy",
    "cap_height_friendly",
    "cap_height_enemy",
    "stack_score_friendly",
    "stack_score_enemy",
    "stack_count_friendly",
    "stack_count_enemy",
    "reserves_friendly",
    "reserves_enemy",
    "comps_friendly",
    "comps_enemy",
    "flat_placement_r_friendly",
    "flat_placement_r_enemy",
    "one_gap_r_friendly",
    "one_gap_r_enemy",
    "cs_friendly",
    "cs_enemy",
];
static WEIGHTS1: &'static str = include_str!("data.mat");

pub struct SmallNN {
    weights1: Matrix<f32, 27, 27>,
    bias1: Matrix<f32, 1, 27>,
    weights2: Matrix<f32, 27, 1>,
    bias2: f32,
}

impl SmallNN {
    pub fn new() -> Self {
        let weights1: Matrix<f32, 27, 27> = WEIGHTS1.lines().map(|x| x.parse().unwrap()).collect();
        let bias1 = [
            0.0258, 0.0669, -0.1591, -0.1901, 0.0286, -0.1532, -0.1522, -0.0804, -0.0165, 0.0483,
            0.1439, -0.0990, -0.0800, 0.1574, -0.0428, 0.0781, 0.1630, -0.0742, 0.0189, -0.1840,
            -0.0190, -0.0156, -0.0920, -0.1787, -0.1422, -0.1142, -0.1785,
        ]
        .into_iter()
        .collect();
        let weights2 = [
            -0.1671, 0.2680, -0.2025, -0.3668, 0.0685, 0.7048, 0.3593, -0.2475, -0.5606, -0.0394,
            -0.0993, 0.0679, 0.1888, 0.0456, 0.1498, 0.2755, 0.0901, 0.2747, 0.1193, -0.2441,
            -0.4877, 0.0966, 0.1771, 0.1527, 0.2814, -0.1136, -0.0810,
        ]
        .into_iter()
        .collect();
        let bias2 = 0.1593;
        Self {
            weights1,
            bias1,
            weights2,
            bias2,
        }
    }
    // pub fn flip_color(input: &mut [i32; 27]) {
    //     const STACK_SCORE: usize = find(LABELS, "stack_score_friendly");
    //     input.swap(1, 2);
    //     input.swap(3, 6);
    //     input.swap(4, 7);
    //     input.swap(5, 8);

    //     // Swaps
    //     for i in 9..LABELS.len() / 2 {
    //         let idx = 1 + i * 2;
    //         input.swap(idx, idx + 1);
    //     }
    //     // Fix the poor feature construction of stack score
    //     input[STACK_SCORE] *= -1;
    //     input[STACK_SCORE + 1] *= -1;
    // }
    pub fn prepare_data(input: [i32; 27]) -> Matrix<f32, 1, 27> {
        const FLAT_P: usize = find(LABELS, "flat_placement_r_friendly");
        const LOC: usize = find(LABELS, "loc_friendly");
        const STACK_SCORE: usize = find(LABELS, "stack_score_friendly");
        const EMPTY: usize = find(LABELS, "empty");
        let mut out: Matrix<f32, 1, 27> = input.into_iter().map(|x| x as f32).collect();
        out[FLAT_P] = f32::min(16.0, out[FLAT_P]);
        out[FLAT_P + 1] = f32::min(16.0, out[FLAT_P + 1]);
        out[LOC] /= 30.0;
        out[LOC + 1] /= 30.0;
        out[STACK_SCORE] /= 20.0;
        out[STACK_SCORE + 1] /= 20.0;
        out[EMPTY] /= 5.0;
        out
    }
    pub fn forward(&self, input: Matrix<f32, 1, 27>) -> i32 {
        let out = input * self.weights1;
        let mut out = out + self.bias1;
        Self::relu(out.as_mut_slice());
        let out = out * self.weights2;
        let out = out[0] + self.bias2;
        // Undo normalization and return to integer
        (out * 100.0) as i32
    }
    fn relu(matrix_slice: &mut [f32]) {
        for val in matrix_slice {
            *val = f32::max(*val, 0.0);
        }
    }
}

const fn find(vec: [&'static str; 27], search: &'static str) -> usize {
    let mut idx = 0;
    let search = search.as_bytes();
    'outer: while idx < vec.len() {
        let bytes = vec[idx].as_bytes();
        if bytes.len() == search.len() {
            let mut inner = 0;
            while inner < search.len() {
                if bytes[inner] != search[inner] {
                    idx += 1;
                    continue 'outer;
                }
                inner += 1;
            }
            return idx;
        }
        idx += 1;
    }
    return usize::MAX;
}
