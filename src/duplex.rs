use super::SearchError;
use crate::tree::{SuffixTree, ROOT_NODE};
use logspace::LogSpace;
use ndarray::{ArrayBase, Axis, Data, Ix1, Ix2, Ix3};
use ndarray_stats::QuantileExt;

mod logspace {
    use std::ops::{Add, AddAssign, Mul, MulAssign};

    #[cfg(feature = "fastexp")]
    fn exp(a: f32) -> f32 {
        use crate::fastexp::FastExp;
        a.fastexp()
    }
    #[cfg(not(feature = "fastexp"))]
    fn exp(a: f32) -> f32 {
        a.exp()
    }

    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct LogSpace(f32);

    impl LogSpace {
        pub fn new(val: f32) -> Self {
            LogSpace(val.ln())
        }
        pub fn zero() -> Self {
            LogSpace(std::f32::NEG_INFINITY)
        }
        pub fn one() -> Self {
            LogSpace(0.0)
        }
        pub fn max(self, other: Self) -> Self {
            if self.0 < other.0 {
                other
            } else {
                self
            }
        }
    }

    impl Add for LogSpace {
        type Output = Self;
        fn add(self, other: Self) -> Self {
            fn add_internal(big: f32, small: f32) -> f32 {
                if small == std::f32::NEG_INFINITY {
                    // -inf is the additive unit (it represents zero probability)
                    big
                } else {
                    big + exp(small - big).ln_1p()
                }
            }
            // order operands by magnitude to ensure a+b produces the same answer as b+a
            // NB: the comparison is done such that if there is a NaN, it will end up being the
            // first argument to add_internal(), and thus be propagated properly (fastexp() doesn't
            // propagate NaNs).
            if self.0 <= other.0 {
                LogSpace(add_internal(other.0, self.0))
            } else {
                LogSpace(add_internal(self.0, other.0))
            }
        }
    }
    impl AddAssign for LogSpace {
        fn add_assign(&mut self, other: Self) {
            *self = self.add(other);
        }
    }
    impl Mul for LogSpace {
        type Output = Self;
        fn mul(self, other: Self) -> Self {
            LogSpace(self.0 + other.0)
        }
    }
    impl MulAssign for LogSpace {
        fn mul_assign(&mut self, other: Self) {
            *self = self.mul(other);
        }
    }
}

/// The probabilities for a node in the search tree.
#[derive(Clone, Copy, Debug)]
struct ProbPair {
    /// The cumulative probability of the labelling so far for paths without any leading blank
    /// labels.
    label: LogSpace,
    /// The cumulative probability of the labelling so far for paths with one or more leading
    /// blank labels.
    gap: LogSpace,
}

impl ProbPair {
    fn zero() -> Self {
        ProbPair {
            label: LogSpace::zero(),
            gap: LogSpace::zero(),
        }
    }

    fn with_label(label: LogSpace) -> Self {
        ProbPair {
            label,
            gap: LogSpace::zero(),
        }
    }

    fn with_gap(gap: LogSpace) -> Self {
        ProbPair {
            label: LogSpace::zero(),
            gap,
        }
    }

    /// The total probability of the labelling so far.
    ///
    /// This sums the probabilities of the paths with and without leading blank labels.
    fn probability(&self) -> LogSpace {
        self.label + self.gap
    }
}

impl std::ops::AddAssign for ProbPair {
    fn add_assign(&mut self, other: Self) {
        self.label += other.label;
        self.gap += other.gap;
    }
}

/// A node in the labelling tree to build from.
#[derive(Clone, Copy, Debug)]
struct SearchPoint {
    /// The node search should progress from.
    node: i32,
    /// The transition state for crf.
    state: usize,
    /// The probability information for the paths in network 1 discovered so far that could produce
    /// the labelling indicated by `node`.
    prob_1: ProbPair,
    /// The maximum probability found in network 2 for the labelling indicated by `node`.
    prob_2_max: LogSpace,
}

impl SearchPoint {
    /// The total probability of the labelling so far.
    fn probability(&self) -> LogSpace {
        self.prob_1.probability() * self.prob_2_max
    }
}

#[derive(Debug)]
struct SecondaryProbs {
    offset: isize,
    probs: Vec<ProbPair>,
    max_prob: LogSpace,
}

impl SecondaryProbs {
    fn with_offset(offset: isize) -> Self {
        SecondaryProbs {
            offset,
            probs: Vec::new(),
            max_prob: LogSpace::zero(),
        }
    }

    fn get(&self, at: isize) -> ProbPair {
        let index = at - self.offset;
        if index < 0 {
            ProbPair::zero()
        } else {
            let index = index as usize;
            if index < self.probs.len() {
                self.probs[index as usize]
            } else {
                ProbPair::zero()
            }
        }
    }

    fn discard_until(&mut self, keep_from: isize) {
        if keep_from > self.offset {
            let first_index = (keep_from - self.offset) as usize;
            if first_index < self.probs.len() {
                self.probs.drain(0..first_index);
            } else {
                self.probs.clear();
            }
            self.offset = keep_from;
        }
    }

    fn update_max(&mut self, lower_bound: isize, upper_bound: isize) {
        assert!(lower_bound <= upper_bound);

        let len = self.probs.len() as isize;
        let begin = (lower_bound - self.offset).clamp(0, len) as usize;
        let end = (upper_bound - self.offset).clamp(begin as isize, len) as usize;
        let mut max_prob = LogSpace::zero();
        for prob in &self.probs[begin..end] {
            max_prob = max_prob.max(prob.probability());
        }
        self.max_prob = max_prob;
    }

    fn end(&self) -> isize {
        assert!(self.probs.len() <= (std::isize::MAX as usize));
        self.offset + (self.probs.len() as isize)
    }
}

fn build_secondary_probs<D: Data<Elem = LogSpace>>(
    network_output: &ArrayBase<D, Ix2>,
    parent_probs: &SecondaryProbs,
    label: usize,
    is_repeat: bool,
    lower_bound: usize,
    upper_bound: usize,
) -> SecondaryProbs {
    assert!(lower_bound < upper_bound);
    assert!(upper_bound <= (std::isize::MAX as usize));
    assert!(upper_bound <= network_output.shape()[0]);
    assert!(label + 1 < network_output.shape()[1]);

    let mut probs = SecondaryProbs::with_offset(lower_bound as isize);
    probs.probs.reserve(upper_bound - lower_bound);
    let mut last_probs = ProbPair::zero();

    for (idx, labelling_probs) in (lower_bound..upper_bound).zip(
        network_output
            .slice(s![lower_bound..upper_bound, ..])
            .outer_iter(),
    ) {
        let gap_prob = last_probs.probability() * labelling_probs[0];
        let prev_parent_probs = parent_probs.get((idx as isize) - 1);
        let label_prob = if is_repeat {
            labelling_probs[label + 1] * (last_probs.label + prev_parent_probs.gap)
        } else {
            labelling_probs[label + 1] * (last_probs.label + prev_parent_probs.probability())
        };
        last_probs = ProbPair {
            label: label_prob,
            gap: gap_prob,
        };
        probs.probs.push(last_probs);
        probs.max_prob = probs.max_prob.max(last_probs.probability());
    }
    probs
}

fn crf_build_secondary_probs<D: Data<Elem = LogSpace>>(
    network_output: &ArrayBase<D, Ix3>,
    parent_probs: &SecondaryProbs,
    label: usize,
    tstate: usize,
    lower_bound: usize,
    upper_bound: usize,
) -> SecondaryProbs {
    assert!(lower_bound < upper_bound);
    assert!(upper_bound <= (std::isize::MAX as usize));
    assert!(upper_bound <= network_output.shape()[0]);
    assert!(label + 1 < network_output.shape()[2]);

    let mut probs = SecondaryProbs::with_offset(lower_bound as isize);
    probs.probs.reserve(upper_bound - lower_bound);

    let mut last_probs = ProbPair::zero();

    for (idx, labelling_probs) in (lower_bound..upper_bound).zip(
        network_output
            .slice(s![lower_bound..upper_bound, .., ..])
            .outer_iter(),
    ) {
        let gap_prob = last_probs.probability() * labelling_probs[[tstate, 0]];
        let prev_parent_probs = parent_probs.get((idx as isize) - 1);
        let label_prob = labelling_probs[[tstate, label + 1]]
            * (last_probs.label + prev_parent_probs.probability());

        last_probs = ProbPair {
            label: label_prob,
            gap: gap_prob,
        };

        probs.probs.push(last_probs);
        probs.max_prob = probs.max_prob.max(last_probs.probability());
    }
    probs
}

fn crf_extend_secondary_probs<D: Data<Elem = LogSpace>>(
    p2: &ArrayBase<D, Ix3>,
    probs: &mut SecondaryProbs,
    parent_probs: &SecondaryProbs,
    label: usize,
    tstate: usize,
    lower_bound: usize,
    upper_bound: usize,
) {
    assert!(upper_bound <= (std::isize::MAX as usize));
    assert!(lower_bound <= upper_bound);

    // discard everything below lower_bound-1, and make sure max_prob only covers from lower_bound
    if (lower_bound as isize) > probs.offset {
        // one before the lower_bound is useful for calculating child probabilities, don't need
        // anything before that
        probs.discard_until((lower_bound as isize) - 1);
        if probs.probs.is_empty() {
            probs.offset = lower_bound as isize;
        }
        probs.update_max(lower_bound as isize, upper_bound as isize);
    }

    let current_end = probs.end();
    assert!(current_end >= 0);
    let current_end = current_end as usize;
    assert!(current_end < upper_bound);

    let mut last_probs = probs.probs.last().copied().unwrap_or(ProbPair::zero());
    probs.probs.reserve(upper_bound - current_end);
    for (idx, labelling_probs) in
        (current_end..upper_bound).zip(p2.slice(s![current_end..upper_bound, .., ..]).outer_iter())
    {
        let gap_prob = last_probs.probability() * labelling_probs[[tstate, 0]];
        let prev_parent_probs = parent_probs.get((idx as isize) - 1);

        let label_prob = labelling_probs[[tstate, label + 1]]
            * (last_probs.label + prev_parent_probs.probability());

        last_probs = ProbPair {
            label: label_prob,
            gap: gap_prob,
        };
        probs.probs.push(last_probs);
        probs.max_prob = probs.max_prob.max(last_probs.probability());
    }
}

fn extend_secondary_probs<D: Data<Elem = LogSpace>>(
    probs: &mut SecondaryProbs,
    network_output: &ArrayBase<D, Ix2>,
    parent_probs: &SecondaryProbs,
    label: usize,
    is_repeat: bool,
    lower_bound: usize,
    upper_bound: usize,
) {
    assert!(upper_bound <= (std::isize::MAX as usize));
    assert!(lower_bound <= upper_bound);

    // discard everything below lower_bound-1, and make sure max_prob only covers from lower_bound
    if (lower_bound as isize) > probs.offset {
        // one before the lower_bound is useful for calculating child probabilities, don't need
        // anything before that
        probs.discard_until((lower_bound as isize) - 1);
        if probs.probs.is_empty() {
            probs.offset = lower_bound as isize;
        }
        probs.update_max(lower_bound as isize, upper_bound as isize);
    }

    let current_end = probs.end();
    assert!(current_end >= 0);
    let current_end = current_end as usize;
    assert!(current_end < upper_bound);

    let mut last_probs = probs.probs.last().copied().unwrap_or(ProbPair::zero());
    probs.probs.reserve(upper_bound - current_end);
    for (idx, labelling_probs) in (current_end..upper_bound).zip(
        network_output
            .slice(s![current_end..upper_bound, ..])
            .outer_iter(),
    ) {
        let gap_prob = last_probs.probability() * labelling_probs[0];
        let prev_parent_probs = parent_probs.get((idx as isize) - 1);
        let label_prob = if is_repeat {
            labelling_probs[label + 1] * (last_probs.label + prev_parent_probs.gap)
        } else {
            labelling_probs[label + 1] * (last_probs.label + prev_parent_probs.probability())
        };
        last_probs = ProbPair {
            label: label_prob,
            gap: gap_prob,
        };
        probs.probs.push(last_probs);
        probs.max_prob = probs.max_prob.max(last_probs.probability());
    }
}

fn root_probs<D: Data<Elem = LogSpace>>(
    gap_probs: &ArrayBase<D, Ix1>,
    upper_bound: usize,
) -> SecondaryProbs {
    let mut probs = SecondaryProbs {
        offset: -1,
        probs: Vec::new(),
        max_prob: LogSpace::one(),
    };
    probs.probs.reserve(upper_bound + 1);

    let mut cur_prob = LogSpace::one();
    probs.probs.push(ProbPair::with_gap(cur_prob));

    for &prob in gap_probs.slice(s![..upper_bound]) {
        cur_prob *= prob;
        probs.probs.push(ProbPair::with_gap(cur_prob));
    }

    probs
}

fn crf_root_probs<D: Data<Elem = LogSpace>>(
    gap_probs: &ArrayBase<D, Ix3>,
    init_state: usize,
    upper_bound: usize,
) -> SecondaryProbs {
    let mut probs = SecondaryProbs {
        offset: -1,
        probs: Vec::new(),
        max_prob: LogSpace::one(),
    };

    let n_state = gap_probs.shape()[1];
    let n_base = gap_probs.shape()[2] - 1;

    probs.probs.reserve(upper_bound + 1);

    let mut cur_prob = LogSpace::one();
    probs.probs.push(ProbPair::with_gap(cur_prob));

    let mut state = init_state;
    for pr in gap_probs
        .slice(s![0..upper_bound, .., ..])
        .axis_iter(Axis(0))
    {
        cur_prob *= pr[[state as usize, 0]];
        probs.probs.push(ProbPair::with_gap(cur_prob));
        state = (state * n_base) % n_state;
    }

    probs
}

pub fn beam_search<D: Data<Elem = f32>, E: Data<Elem = usize>>(
    network_output_1_real: &ArrayBase<D, Ix2>,
    network_output_2_real: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    envelope: &ArrayBase<E, Ix2>,
    beam_size: usize,
    beam_cut_threshold_real: f32,
    collapse_repeats: bool,
) -> Result<String, SearchError> {
    let network_output_1 = network_output_1_real.map(|&x| LogSpace::new(x));
    let network_output_2 = network_output_2_real.map(|&x| LogSpace::new(x));
    let beam_cut_threshold = LogSpace::new(beam_cut_threshold_real);

    assert_eq!(network_output_1.shape()[1], network_output_2.shape()[1]);
    assert_eq!(network_output_1.shape()[0], envelope.shape()[0]);
    assert_eq!(envelope.shape()[1], 2);
    assert_eq!(network_output_1.shape()[1], alphabet.len());

    // alphabet size minus the blank label
    let alphabet_size = alphabet.len() - 1;

    let mut suffix_tree = SuffixTree::new(alphabet_size);
    let mut beam = vec![SearchPoint {
        node: ROOT_NODE,
        state: 0,
        prob_1: ProbPair {
            label: LogSpace::zero(),
            gap: LogSpace::one(),
        },
        prob_2_max: LogSpace::one(),
    }];
    let mut next_beam = Vec::new();

    let root_secondary_probs =
        root_probs(&network_output_2.index_axis(Axis(1), 0), envelope[(0, 1)]);

    let network_2_len = network_output_2.shape()[0];
    let mut last_upper_bound = 0;

    for (labelling_probs, bounds) in network_output_1.outer_iter().zip(envelope.outer_iter()) {
        next_beam.clear();

        let (lower_t, upper_t) = (bounds[0].max(0), bounds[1].min(network_2_len));
        if lower_t >= upper_t || lower_t > last_upper_bound {
            return Err(SearchError::InvalidEnvelope);
        }

        if upper_t > last_upper_bound {
            // need to extend secondary probs for anything still in the search beam

            beam.sort_by_key(|x| x.node); // parents before children
            let mut placeholder = SecondaryProbs::with_offset(0);
            for &SearchPoint { node, .. } in &beam {
                if let Some(info) = suffix_tree.info(node) {
                    // we need to swap the data out before editing to satisfy Rust's borrowing rules
                    let mut has_data = false;
                    if let Some(data) = suffix_tree.get_data_ref_mut(node) {
                        std::mem::swap(data, &mut placeholder);
                        has_data = true;
                    }
                    if has_data {
                        extend_secondary_probs(
                            &mut placeholder,
                            &network_output_2,
                            suffix_tree
                                .get_data_ref(info.parent)
                                .unwrap_or_else(|| &root_secondary_probs),
                            info.label,
                            suffix_tree.label(info.parent) == Some(info.label),
                            lower_t,
                            upper_t,
                        );
                        std::mem::swap(
                            suffix_tree.get_data_ref_mut(node).unwrap(),
                            &mut placeholder,
                        );
                    }
                }
            }
        }

        last_upper_bound = upper_t;

        for &tip in &beam {
            let tip_label = suffix_tree.label(tip.node);
            // add N to beam
            if labelling_probs[0] > beam_cut_threshold {
                next_beam.push(SearchPoint {
                    prob_1: ProbPair::with_gap(tip.prob_1.probability() * labelling_probs[0]),
                    ..tip
                });
            }

            for (label, &prob) in labelling_probs.iter().skip(1).enumerate() {
                if prob < beam_cut_threshold {
                    continue;
                }
                if collapse_repeats && Some(label) == tip_label {
                    next_beam.push(SearchPoint {
                        prob_1: ProbPair::with_label(tip.prob_1.label * prob),
                        ..tip
                    });
                    let new_node_idx = suffix_tree.get_child(tip.node, label).or_else(|| {
                        if tip.prob_1.gap > LogSpace::zero() {
                            let secondary_probs = build_secondary_probs(
                                &network_output_2,
                                suffix_tree
                                    .get_data_ref(tip.node)
                                    .unwrap_or_else(|| &root_secondary_probs),
                                label,
                                true,
                                lower_t,
                                upper_t,
                            );
                            Some(suffix_tree.add_node(tip.node, label, secondary_probs))
                        } else {
                            None
                        }
                    });

                    if let Some(idx) = new_node_idx {
                        next_beam.push(SearchPoint {
                            node: idx,
                            prob_1: ProbPair::with_label(tip.prob_1.gap * prob),
                            ..tip
                        });
                    }
                } else {
                    let new_node_idx =
                        suffix_tree.get_child(tip.node, label).unwrap_or_else(|| {
                            let secondary_probs = build_secondary_probs(
                                &network_output_2,
                                suffix_tree
                                    .get_data_ref(tip.node)
                                    .unwrap_or_else(|| &root_secondary_probs),
                                label,
                                false,
                                lower_t,
                                upper_t,
                            );
                            suffix_tree.add_node(tip.node, label, secondary_probs)
                        });

                    next_beam.push(SearchPoint {
                        node: new_node_idx,
                        prob_1: ProbPair::with_label(tip.prob_1.probability() * prob),
                        ..tip
                    });
                }
            }
        }

        std::mem::swap(&mut beam, &mut next_beam);

        const DELETE_MARKER: i32 = i32::min_value();
        beam.sort_by_key(|x| x.node);
        let mut last_key: i32 = DELETE_MARKER;
        let mut last_key_pos = 0;
        for i in 0..beam.len() {
            let beam_item = beam[i];
            if beam_item.node == last_key {
                beam[last_key_pos].prob_1 += beam_item.prob_1;
                beam[i].node = DELETE_MARKER;
            } else {
                last_key_pos = i;
                last_key = beam_item.node;
            }
        }

        beam.retain(|x| x.node != DELETE_MARKER);
        for beam_item in &mut beam {
            let node = beam_item.node;
            if let Some(data) = suffix_tree.get_data_ref(node) {
                beam_item.prob_2_max = data.max_prob;
            }
        }
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.probability())
                .partial_cmp(&(a.probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal // don't really care
                })
        });
        if has_nans {
            return Err(SearchError::IncomparableValues);
        }
        beam.truncate(beam_size);
        if beam.is_empty() {
            // we've run out of beam (probably the threshold is too high)
            return Err(SearchError::RanOutOfBeam);
        }
    }

    let mut sequence = String::new();

    if beam[0].node != ROOT_NODE {
        for label in suffix_tree.iter_from_no_data(beam[0].node) {
            sequence.push_str(&alphabet[label + 1]);
        }
    }

    Ok(sequence.chars().rev().collect())
}

pub fn crf_beam_search<D: Data<Elem = f32>, E: Data<Elem = usize>>(
    network_output_1_real: &ArrayBase<D, Ix3>,
    init_state_1: &ArrayBase<D, Ix1>,
    network_output_2_real: &ArrayBase<D, Ix3>,
    init_state_2: &ArrayBase<D, Ix1>,
    alphabet: &[String],
    envelope: &ArrayBase<E, Ix2>,
    beam_size: usize,
    beam_cut_threshold_real: f32,
) -> Result<String, SearchError> {
    let network_output_1 = network_output_1_real.map(|&x| LogSpace::new(x));
    let network_output_2 = network_output_2_real.map(|&x| LogSpace::new(x));
    let beam_cut_threshold = LogSpace::new(beam_cut_threshold_real);

    assert_eq!(network_output_1.shape()[1], network_output_2.shape()[1]);
    assert_eq!(network_output_1.shape()[2], network_output_2.shape()[2]);
    assert_eq!(network_output_1.shape()[2], alphabet.len());
    assert_eq!(network_output_2.shape()[2], alphabet.len());
    assert_eq!(network_output_1.shape()[0], envelope.shape()[0]);
    assert_eq!(envelope.shape()[1], 2);

    let n_state = network_output_1.shape()[1];
    let n_base = network_output_1.shape()[2] - 1;

    let mut suffix_tree = SuffixTree::new(n_base);
    let mut beam = vec![SearchPoint {
        node: ROOT_NODE,
        state: init_state_1.argmax().unwrap(),
        prob_1: ProbPair {
            label: LogSpace::zero(),
            gap: LogSpace::one(),
        },
        prob_2_max: LogSpace::one(),
    }];

    let mut next_beam = Vec::new();

    let root_secondary_probs = crf_root_probs(
        &network_output_2.view(),
        init_state_2.argmax().unwrap(),
        envelope[(0, 1)],
    );

    let network_2_len = network_output_2.shape()[0];
    let mut last_upper_bound = 0;

    for (probs, bounds) in network_output_1.outer_iter().zip(envelope.outer_iter()) {
        next_beam.clear();

        let (lower_t, upper_t) = (bounds[0].max(0), bounds[1].min(network_2_len));
        if lower_t >= upper_t || lower_t > last_upper_bound {
            return Err(SearchError::InvalidEnvelope);
        }

        if upper_t > last_upper_bound {
            // need to extend secondary probs for anything still in the search beam
            beam.sort_by_key(|x| x.node); // parents before children
            let mut placeholder = SecondaryProbs::with_offset(0);
            for &SearchPoint { node, state, .. } in &beam {
                if let Some(info) = suffix_tree.info(node) {
                    // we need to swap the data out before editing to satisfy Rust's borrowing rules
                    let mut has_data = false;

                    if let Some(data) = suffix_tree.get_data_ref_mut(node) {
                        std::mem::swap(data, &mut placeholder);
                        has_data = true;
                    }

                    if has_data {
                        crf_extend_secondary_probs(
                            &network_output_2,
                            &mut placeholder,
                            suffix_tree
                                .get_data_ref(info.parent)
                                .unwrap_or_else(|| &root_secondary_probs),
                            info.label,
                            state,
                            lower_t,
                            upper_t,
                        );
                        std::mem::swap(
                            suffix_tree.get_data_ref_mut(node).unwrap(),
                            &mut placeholder,
                        );
                    }
                }
            }
        }

        last_upper_bound = upper_t;

        for &tip in &beam {
            let labelling_probs = probs.slice(s![tip.state, ..]);

            // add N to beam
            if labelling_probs[0] > beam_cut_threshold {
                next_beam.push(SearchPoint {
                    prob_1: ProbPair::with_gap(tip.prob_1.probability() * labelling_probs[0]),
                    ..tip
                });
            }

            for (label, &prob) in labelling_probs.iter().skip(1).enumerate() {
                if prob < beam_cut_threshold {
                    continue;
                }

                let new_node_idx = suffix_tree.get_child(tip.node, label).unwrap_or_else(|| {
                    let secondary_probs = crf_build_secondary_probs(
                        &network_output_2,
                        suffix_tree
                            .get_data_ref(tip.node)
                            .unwrap_or_else(|| &root_secondary_probs),
                        label,
                        tip.state,
                        lower_t,
                        upper_t,
                    );
                    suffix_tree.add_node(tip.node, label, secondary_probs)
                });

                next_beam.push(SearchPoint {
                    node: new_node_idx,
                    state: (tip.state * n_base) % n_state + (label),
                    prob_1: ProbPair::with_label(tip.prob_1.probability() * prob),
                    ..tip
                });
            }
        }

        std::mem::swap(&mut beam, &mut next_beam);

        const DELETE_MARKER: i32 = i32::min_value();
        beam.sort_by_key(|x| x.node);
        let mut last_key: i32 = DELETE_MARKER;
        let mut last_key_pos = 0;
        for i in 0..beam.len() {
            let beam_item = beam[i];
            if beam_item.node == last_key {
                beam[last_key_pos].prob_1 += beam_item.prob_1;
                beam[i].node = DELETE_MARKER;
            } else {
                last_key_pos = i;
                last_key = beam_item.node;
            }
        }

        beam.retain(|x| x.node != DELETE_MARKER);
        for beam_item in &mut beam {
            let node = beam_item.node;
            if let Some(data) = suffix_tree.get_data_ref(node) {
                beam_item.prob_2_max = data.max_prob;
            }
        }
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.probability())
                .partial_cmp(&(a.probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal // don't really care
                })
        });
        if has_nans {
            return Err(SearchError::IncomparableValues);
        }
        beam.truncate(beam_size);
        if beam.is_empty() {
            // we've run out of beam (probably the threshold is too high)
            return Err(SearchError::RanOutOfBeam);
        }
    }

    let mut sequence = String::new();

    if beam[0].node != ROOT_NODE {
        for label in suffix_tree.iter_from_no_data(beam[0].node) {
            sequence.push_str(&alphabet[label + 1]);
        }
    }

    Ok(sequence.chars().rev().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::PartialEq;

    #[test]
    fn test_secondary_probs_get() {
        let mut p = SecondaryProbs {
            offset: 0,
            probs: vec![
                ProbPair::with_gap(LogSpace::new(0.1)),
                ProbPair::with_gap(LogSpace::new(0.2)),
                ProbPair::with_gap(LogSpace::new(0.3)),
            ],
            max_prob: LogSpace::zero(),
        };
        assert!(!p.probs[0].gap.eq(&p.probs[1].gap));
        assert!(!p.probs[0].gap.eq(&p.probs[2].gap));
        assert!(!p.probs[1].gap.eq(&p.probs[2].gap));

        assert!(p.get(-1).gap.eq(&LogSpace::zero()));
        assert!(p.get(0).gap.eq(&p.probs[0].gap));
        assert!(p.get(1).gap.eq(&p.probs[1].gap));
        assert!(p.get(2).gap.eq(&p.probs[2].gap));
        assert!(p.get(3).gap.eq(&LogSpace::zero()));

        p.offset = 3;
        assert!(p.get(-1).gap.eq(&LogSpace::zero()));
        assert!(p.get(0).gap.eq(&LogSpace::zero()));
        assert!(p.get(2).gap.eq(&LogSpace::zero()));
        assert!(p.get(3).gap.eq(&p.probs[0].gap));
        assert!(p.get(4).gap.eq(&p.probs[1].gap));
        assert!(p.get(5).gap.eq(&p.probs[2].gap));
        assert!(p.get(6).gap.eq(&LogSpace::zero()));

        p.offset = -1;
        assert!(p.get(-2).gap.eq(&LogSpace::zero()));
        assert!(p.get(-1).gap.eq(&p.probs[0].gap));
        assert!(p.get(0).gap.eq(&p.probs[1].gap));
        assert!(p.get(1).gap.eq(&p.probs[2].gap));
        assert!(p.get(2).gap.eq(&LogSpace::zero()));

        p.probs.clear();
        p.offset = 0;
        assert!(p.get(-1).gap.eq(&LogSpace::zero()));
        assert!(p.get(0).gap.eq(&LogSpace::zero()));
        assert!(p.get(1).gap.eq(&LogSpace::zero()));

        p.offset = -1;
        assert!(p.get(-2).gap.eq(&LogSpace::zero()));
        assert!(p.get(-1).gap.eq(&LogSpace::zero()));
        assert!(p.get(0).gap.eq(&LogSpace::zero()));

        p.offset = 4;
        assert!(p.get(3).gap.eq(&LogSpace::zero()));
        assert!(p.get(4).gap.eq(&LogSpace::zero()));
        assert!(p.get(5).gap.eq(&LogSpace::zero()));
    }

    #[test]
    fn test_secondary_probs_update_max_empty() {
        let mut p = SecondaryProbs {
            offset: 0,
            probs: vec![],
            max_prob: LogSpace::zero(),
        };

        p.max_prob = LogSpace::one();
        p.update_max(0, 0);
        assert!(p.max_prob.eq(&LogSpace::zero()));

        p.max_prob = LogSpace::one();
        p.update_max(-1, 0);
        assert!(p.max_prob.eq(&LogSpace::zero()));

        p.max_prob = LogSpace::one();
        p.update_max(0, 1);
        assert!(p.max_prob.eq(&LogSpace::zero()));

        p.max_prob = LogSpace::one();
        p.update_max(-1, 1);
        assert!(p.max_prob.eq(&LogSpace::zero()));

        p.max_prob = LogSpace::one();
        p.update_max(std::isize::MIN, std::isize::MAX);
        assert!(p.max_prob.eq(&LogSpace::zero()));
    }

    #[test]
    fn test_secondary_probs_update_max_values() {
        let mut p = SecondaryProbs {
            offset: 2,
            probs: vec![
                ProbPair::with_gap(LogSpace::new(0.1)),
                ProbPair::with_label(LogSpace::new(0.3)),
                ProbPair::with_label(LogSpace::new(0.2)),
                ProbPair::with_label(LogSpace::new(0.4)),
                ProbPair::with_gap(LogSpace::new(0.5)),
            ],
            max_prob: LogSpace::zero(),
        };

        p.max_prob = LogSpace::one();
        p.update_max(0, 0);
        assert!(p.max_prob.eq(&LogSpace::zero()));

        p.max_prob = LogSpace::one();
        p.update_max(0, 2);
        assert!(p.max_prob.eq(&LogSpace::zero()));

        p.max_prob = LogSpace::one();
        p.update_max(0, 3);
        assert!(p.max_prob.eq(&LogSpace::new(0.1)));

        p.max_prob = LogSpace::one();
        p.update_max(2, 2);
        assert!(p.max_prob.eq(&LogSpace::zero()));

        p.max_prob = LogSpace::one();
        p.update_max(2, 3);
        assert!(p.max_prob.eq(&LogSpace::new(0.1)));

        p.max_prob = LogSpace::one();
        p.update_max(2, 4);
        assert!(p.max_prob.eq(&LogSpace::new(0.3)));

        p.max_prob = LogSpace::one();
        p.update_max(2, 5);
        assert!(p.max_prob.eq(&LogSpace::new(0.3)));

        p.max_prob = LogSpace::one();
        p.update_max(2, 6);
        assert!(p.max_prob.eq(&LogSpace::new(0.4)));

        p.max_prob = LogSpace::one();
        p.update_max(2, 7);
        assert!(p.max_prob.eq(&LogSpace::new(0.5)));

        p.max_prob = LogSpace::one();
        p.update_max(6, 7);
        assert!(p.max_prob.eq(&LogSpace::new(0.5)));

        p.max_prob = LogSpace::one();
        p.update_max(7, 7);
        assert!(p.max_prob.eq(&LogSpace::zero()));

        p.max_prob = LogSpace::one();
        p.update_max(2, 10);
        assert!(p.max_prob.eq(&LogSpace::new(0.5)));

        p.max_prob = LogSpace::one();
        p.update_max(3, 10);
        assert!(p.max_prob.eq(&LogSpace::new(0.5)));

        p.max_prob = LogSpace::one();
        p.update_max(8, 10);
        assert!(p.max_prob.eq(&LogSpace::zero()));
    }
}
