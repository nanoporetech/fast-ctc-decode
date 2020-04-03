use super::SearchError;
use crate::vec2d::Vec2D;
use ndarray::{ArrayBase, Data, Ix2};

/// An element in a possible labelling.
#[derive(Clone, Copy, Debug)]
struct LabelNode {
    /// The index into the alphabet of this label.
    ///
    /// Note that blanks are not represented by a LabelNode - this is an actual label.
    label: usize,
    /// The index of the next LabelNode.
    ///
    /// Can also be considered the parent edge in the tree of labelling suffixes.
    next: i32,
    /// The last(?) sample to which the label applies.
    time: usize,
}

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
    alphabet: &Vec<String>,
    beam_size: usize,
    beam_cut_threshold: f32,
) -> Result<(String, Vec<usize>), SearchError> {
    // alphabet_size minus the blank label
    let alphabet_size = alphabet.len() - 1;
    let duration = network_output.nrows();

    // suffix_tree and suffix_children, between them, describe a tree of labelling suffixes
    // (partial labellings pinned to the end of the network output). Zipping the two vectors
    // together gives all the nodes in the tree, with the first entry being the root node and the
    // next field of LabelNode giving the index of each node's parent.
    let mut suffix_tree = vec![LabelNode {
        label: 0,
        next: 0,
        time: 0,
    }];
    // suffix_children is a 2D array.
    //
    // Invariants:
    //
    // suffix_tree.len() == suffix_children.len()
    //
    // For all n, l where suffix_children[n][l] != -1:
    //     suffix_tree[suffix_children[n][l]].label == l (child edge label matches child label)
    //     suffix_tree[suffix_children[n][l]].next == n  (child's parent pointer is correct)
    //
    // For all n > 0:
    //     suffix_children[suffix_tree[n].next][suffix_tree[n].label] == n
    //     (the parent node has a child edge back to this node labelled correctly)
    let mut suffix_children: Vec2D<i32> = Vec2D::new(alphabet_size);
    suffix_children.add_row_with_value(-1);
    let mut beam = vec![SearchPoint {
        node: 0,
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
            // add N to beam
            if pr[0] > beam_cut_threshold {
                next_beam.push(SearchPoint {
                    node,
                    label_prob: 0.0,
                    gap_prob: (label_prob + gap_prob) * pr[0],
                });
            }

            for (label, &pr_b) in (1..=alphabet_size).zip(pr.iter().skip(1)) {
                if pr_b < beam_cut_threshold {
                    continue;
                }
                if label == suffix_tree[node as usize].label {
                    next_beam.push(SearchPoint {
                        node,
                        label_prob: label_prob * pr_b,
                        gap_prob: 0.0,
                    });
                    let mut new_node_idx = suffix_children[[node as usize, label - 1]];
                    if new_node_idx == -1 && gap_prob > 0.0 {
                        new_node_idx = suffix_tree.len() as i32;
                        suffix_tree.push(LabelNode {
                            label: label,
                            next: node,
                            time: fidx,
                        });
                        suffix_children[[node as usize, label - 1]] = new_node_idx;
                        suffix_children.add_row_with_value(-1);
                    }

                    next_beam.push(SearchPoint {
                        node: new_node_idx,
                        label_prob: gap_prob * pr_b,
                        gap_prob: 0.0,
                    });
                } else {
                    let mut new_node_idx = suffix_children[[node as usize, label - 1]];
                    if new_node_idx == -1 {
                        new_node_idx = suffix_tree.len() as i32;
                        suffix_tree.push(LabelNode {
                            label: label,
                            next: node,
                            time: fidx,
                        });
                        suffix_children[[node as usize, label - 1]] = new_node_idx;
                        suffix_children.add_row_with_value(-1);
                    }

                    next_beam.push(SearchPoint {
                        node: new_node_idx,
                        label_prob: (label_prob + gap_prob) * pr_b,
                        gap_prob: 0.0,
                    });
                }
            }
        }
        std::mem::swap(&mut beam, &mut next_beam);

        beam.sort_by_key(|x| x.node);
        let mut last_key: i32 = -1;
        let mut last_key_pos = 0;
        for i in 0..beam.len() {
            let beam_item = beam[i];
            if beam_item.node == last_key {
                beam[last_key_pos].label_prob += beam_item.label_prob;
                beam[last_key_pos].gap_prob += beam_item.gap_prob;
                beam[i].node = -1;
            } else {
                last_key_pos = i;
                last_key = beam_item.node;
            }
        }

        beam.retain(|x| x.node != -1);
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

    let mut node_idx = beam[0].node;
    let mut path = Vec::new();
    let mut sequence = String::new();

    while node_idx != 0 {
        let node = &suffix_tree[node_idx as usize];
        path.push(node.time);
        sequence.push_str(&alphabet[node.label]);
        node_idx = node.next;
    }

    Ok((sequence, path))
}
