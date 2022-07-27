use vectrix::Matrix;

const LABELS: [&'static str; 27] = [
    "empty",
    "loc_white",
    "loc_black",
    "flat_white",
    "wall_white",
    "cap_white",
    "flat_black",
    "wall_black",
    "cap_black",
    "cap_status_white",
    "cap_status_black",
    "cap_height_white",
    "cap_height_black",
    "stack_score_white",
    "stack_score_black",
    "stack_count_white",
    "stack_count_black",
    "reserves_white",
    "reserves_black",
    "comps_white",
    "comps_black",
    "flat_placement_r_white",
    "flat_placement_r_black",
    "one_gap_r_white",
    "one_gap_r_black",
    "cs_white",
    "cs_black",
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
            -0.16706082224845886,
            0.2679905593395233,
            -0.20251516997814178,
            -0.366769015789032,
            0.06854448467493057,
            0.7047578692436218,
            0.3592526614665985,
            -0.2475123256444931,
            -0.5606487989425659,
            -0.03941737487912178,
            -0.0992896556854248,
            0.0679473802447319,
            0.18884554505348206,
            0.04559066891670227,
            0.14979691803455353,
            0.2755080759525299,
            0.09012038260698318,
            0.274671345949173,
            0.11925443261861801,
            -0.24413184821605682,
            -0.4876846671104431,
            0.0966338962316513,
            0.17707756161689758,
            0.15273012220859528,
            0.2814192473888397,
            -0.11357637494802475,
            -0.08102928847074509,
        ]
        .into_iter()
        .collect();
        let bias2 = 0.15925399959087372;
        Self {
            weights1,
            bias1,
            weights2,
            bias2,
        }
    }
    pub fn flip_color(input: &mut [i32; 27]) {
        const STACK_SCORE: usize = find(LABELS, "stack_score_white");
        input.swap(1, 2);
        input.swap(3, 6);
        input.swap(4, 7);
        input.swap(5, 8);

        // Swaps
        for i in 9..LABELS.len() / 2 {
            let idx = 1 + i * 2;
            input.swap(idx, idx + 1);
        }
        // Fix the poor feature construction of stack score
        input[STACK_SCORE] *= -1;
        input[STACK_SCORE + 1] *= -1;
    }
    pub fn prepare_data(input: [i32; 27]) -> Matrix<f32, 1, 27> {
        const FLAT_P: usize = find(LABELS, "flat_placement_r_white");
        const LOC: usize = find(LABELS, "loc_white");
        const STACK_SCORE: usize = find(LABELS, "stack_score_white");
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
