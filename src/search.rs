use super::SearchError;
use crate::tree::{SuffixTree, ROOT_NODE};
use ndarray::{ArrayBase, Data, Ix2};

/// A node in the labelling tree to build from.
#[derive(Clone, Copy, Debug)]
struct SearchPoint {
    /// The node search should progress from.
    node: i32,
    /// The cumulative probability of the labelling so far for paths without any leading blank
    /// labels.
    label_prob: f32,
    /// The cumulative probability of the labelling so far for paths with one or more leading
    /// blank labels.
    gap_prob: f32,
}

impl SearchPoint {
    /// The total probability of the labelling so far.
    ///
    /// This sums the probabilities of the paths with and without leading blank labels.
    fn probability(&self) -> f32 {
        self.label_prob + self.gap_prob
    }
}

pub fn beam_search<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
) -> Result<(String, Vec<usize>), SearchError> {
    // alphabet size minus the blank label
    let alphabet_size = alphabet.len() - 1;
    let duration = network_output.nrows();

    let mut suffix_tree = SuffixTree::new(alphabet_size);
    let mut beam = vec![SearchPoint {
        node: ROOT_NODE,
        label_prob: 0.0,
        gap_prob: 1.0,
    }];
    let mut next_beam = Vec::new();

    for (idx, pr) in network_output.slice(s![..;-1, ..]).outer_iter().enumerate() {
        next_beam.clear();

        // forward index in time
        let fidx = duration - idx - 1;

        for &SearchPoint {
            node,
            label_prob,
            gap_prob,
        } in &beam
        {
            let tip_label = suffix_tree.label(node);
            // add N to beam
            if pr[0] > beam_cut_threshold {
                next_beam.push(SearchPoint {
                    node,
                    label_prob: 0.0,
                    gap_prob: (label_prob + gap_prob) * pr[0],
                });
            }

            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                if pr_b < beam_cut_threshold {
                    continue;
                }
                if Some(label) == tip_label {
                    next_beam.push(SearchPoint {
                        node,
                        label_prob: label_prob * pr_b,
                        gap_prob: 0.0,
                    });
                    let new_node_idx = suffix_tree.get_child(node, label).or_else(|| {
                        if gap_prob > 0.0 {
                            Some(suffix_tree.add_node(node, label, fidx))
                        } else {
                            None
                        }
                    });

                    if let Some(idx) = new_node_idx {
                        next_beam.push(SearchPoint {
                            node: idx,
                            label_prob: gap_prob * pr_b,
                            gap_prob: 0.0,
                        });
                    }
                } else {
                    let new_node_idx = suffix_tree
                        .get_child(node, label)
                        .unwrap_or_else(|| suffix_tree.add_node(node, label, fidx));

                    next_beam.push(SearchPoint {
                        node: new_node_idx,
                        label_prob: (label_prob + gap_prob) * pr_b,
                        gap_prob: 0.0,
                    });
                }
            }
        }
        std::mem::swap(&mut beam, &mut next_beam);

        const DELETE_MARKER: i32 = i32::min_value();
        beam.sort_by_key(|x| x.node);
        let mut last_key = DELETE_MARKER;
        let mut last_key_pos = 0;
        for i in 0..beam.len() {
            let beam_item = beam[i];
            if beam_item.node == last_key {
                beam[last_key_pos].label_prob += beam_item.label_prob;
                beam[last_key_pos].gap_prob += beam_item.gap_prob;
                beam[i].node = DELETE_MARKER;
            } else {
                last_key_pos = i;
                last_key = beam_item.node;
            }
        }

        beam.retain(|x| x.node != DELETE_MARKER);
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
        let top = beam[0].probability();
        for mut x in &mut beam {
            x.label_prob /= top;
            x.gap_prob /= top;
        }
    }

    let mut path = Vec::new();
    let mut sequence = String::new();

    for (label, time) in suffix_tree.iter_from(beam[0].node) {
        path.push(time);
        sequence.push_str(&alphabet[label + 1]);
    }

    Ok((sequence, path))
}
