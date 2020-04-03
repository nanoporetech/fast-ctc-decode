use super::SearchError;
use crate::tree::{SuffixTree, ROOT_NODE};
use ndarray::{ArrayBase, Axis, Data, Ix1, Ix2};

/// The probabilities for a node in the search tree.
#[derive(Clone, Copy, Debug)]
struct ProbPair {
    /// The cumulative probability of the labelling so far for paths without any leading blank
    /// labels.
    label: f32,
    /// The cumulative probability of the labelling so far for paths with one or more leading
    /// blank labels.
    gap: f32,
}

impl ProbPair {
    fn zero() -> Self {
        ProbPair {
            label: 0.0,
            gap: 0.0,
        }
    }

    fn with_label(label: f32) -> Self {
        ProbPair { label, gap: 0.0 }
    }

    fn with_gap(gap: f32) -> Self {
        ProbPair { label: 0.0, gap }
    }

    /// The total probability of the labelling so far.
    ///
    /// This sums the probabilities of the paths with and without leading blank labels.
    fn probability(&self) -> f32 {
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
    /// The probability information for the paths in network 1 discovered so far that could produce
    /// the labelling indicated by `node`.
    prob_1: ProbPair,
    /// The maximum probability found in network 2 for the labelling indicated by `node`.
    prob_2_max: f32,
}

impl SearchPoint {
    /// The total probability of the labelling so far.
    fn probability(&self) -> f32 {
        self.prob_1.probability() * self.prob_2_max
    }
}

#[derive(Debug)]
struct SecondaryProbs {
    offset: usize,
    probs: Vec<ProbPair>,
    max_prob: f32,
}

impl SecondaryProbs {
    fn with_offset(offset: usize) -> Self {
        SecondaryProbs {
            offset,
            probs: Vec::new(),
            max_prob: f32::NEG_INFINITY,
        }
    }
}

fn build_secondary_probs<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    parent_probs: &SecondaryProbs,
    label: usize,
    is_repeat: bool,
    lower_bound: usize,
    upper_bound: usize,
) -> SecondaryProbs {
    let mut probs = SecondaryProbs::with_offset(lower_bound);
    probs.probs.reserve(upper_bound - lower_bound);
    let mut last_probs = ProbPair::zero();
    for (idx, labelling_probs) in ((lower_bound..upper_bound).rev()).zip(
        network_output
            .slice(s![lower_bound..upper_bound;-1, ..])
            .outer_iter(),
    ) {
        let gap_prob = last_probs.probability() * labelling_probs[0];
        let prev_idx = idx + 1; // we're going backwards
        let prev_parent_probs = if prev_idx < parent_probs.offset {
            ProbPair::zero()
        } else {
            let parent_idx = prev_idx - parent_probs.offset;
            if parent_idx >= parent_probs.probs.len() {
                ProbPair::zero()
            } else {
                parent_probs.probs[parent_idx]
            }
        };
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
    probs.probs.reverse(); // we added them in the wrong order
    probs
}

fn root_probs<D: Data<Elem = f32>>(
    gap_probs: &ArrayBase<D, Ix1>,
    lower_bound: usize,
) -> SecondaryProbs {
    let mut probs = SecondaryProbs {
        offset: lower_bound,
        probs: Vec::new(),
        max_prob: 1.0,
    };
    probs.probs.reserve(1 + gap_probs.len() - lower_bound);
    // this is the only "out of bounds" probability that isn't just zero
    probs.probs.push(ProbPair::with_gap(1.0));
    let mut cur_prob = 1.0;
    for prob in gap_probs.slice(s![lower_bound..;-1]) {
        cur_prob *= prob;
        probs.probs.push(ProbPair::with_gap(cur_prob));
    }
    probs.probs.reverse();
    probs
}

pub fn beam_search<D: Data<Elem = f32>, E: Data<Elem = usize>>(
    network_output_1: &ArrayBase<D, Ix2>,
    network_output_2: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    envelope: &ArrayBase<E, Ix2>,
    beam_size: usize,
    beam_cut_threshold: f32,
) -> Result<String, SearchError> {
    assert_eq!(network_output_1.shape()[1], network_output_2.shape()[1]);
    assert_eq!(network_output_1.shape()[0], envelope.shape()[0]);
    assert_eq!(envelope.shape()[1], 2);
    assert_eq!(network_output_1.shape()[1], alphabet.len());

    // alphabet size minus the blank label
    let alphabet_size = alphabet.len() - 1;

    let mut suffix_tree = SuffixTree::new(alphabet_size);
    let mut beam = vec![SearchPoint {
        node: ROOT_NODE,
        prob_1: ProbPair {
            label: 0.0,
            gap: 1.0,
        },
        prob_2_max: 1.0,
    }];
    let mut next_beam = Vec::new();
    let mut root_secondary_probs = root_probs(
        &network_output_2.index_axis(Axis(1), 0),
        envelope[[envelope.nrows() - 1, 0]],
    );
    let network_2_len = network_output_2.shape()[0];

    for (labelling_probs, bounds) in network_output_1
        .slice(s![..;-1, ..])
        .outer_iter()
        .zip(envelope.outer_iter().rev())
    {
        let (lower_t, upper_t) = (bounds[0].max(0), bounds[1].min(network_2_len));
        next_beam.clear();

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
                if Some(label) == tip_label {
                    next_beam.push(SearchPoint {
                        prob_1: ProbPair::with_label(tip.prob_1.label * prob),
                        ..tip
                    });
                    let new_node_idx = suffix_tree.get_child(tip.node, label).or_else(|| {
                        if tip.prob_1.gap > 0.0 {
                            let secondary_probs = build_secondary_probs(
                                network_output_2,
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
                                network_output_2,
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
        let scale_1 = beam[0].prob_1.probability();
        let scale_2 = beam[0].prob_2_max;
        for x in &mut beam {
            x.prob_1.label /= scale_1;
            x.prob_1.gap /= scale_1;
            let secondary = suffix_tree
                .get_data_ref_mut(x.node)
                .unwrap_or(&mut root_secondary_probs);
            for val in &mut secondary.probs {
                val.label /= scale_2;
                val.gap /= scale_2;
            }
            secondary.max_prob /= scale_2;
            x.prob_2_max = secondary.max_prob;
        }
    }

    let mut sequence = String::new();

    for label in suffix_tree.iter_from_no_data(beam[0].node) {
        sequence.push_str(&alphabet[label + 1]);
    }

    Ok(sequence)
}
