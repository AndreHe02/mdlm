import torch
import numpy as np


def forward_deletions(x, move_chance, bos, eos, pad_token_id):
    xt = np.full_like(x, pad_token_id, dtype=x.dtype)
    del_indices = np.random.rand(*x.shape) < move_chance

    j = np.zeros(x.shape[0], dtype=np.long)
    for i in range(x.shape[1]):
        keep_indices = ~del_indices[:, i] | (x[:, i] == bos) | (x[:, i] == eos)
        xt[keep_indices, j[keep_indices]] = x[keep_indices, i]
        j += keep_indices

    return xt


def remove_masked_tokens(x, mask_token_id, pad_token_id):
    # the insertion model will predict mask in place of tokens that are deleted
    # we should remove these tokens from the input sequence
    x_ = np.full_like(x, pad_token_id, dtype=x.dtype)
    i = np.zeros(x.shape[0], dtype=np.long)
    for j in range(x.shape[1]):
        keep_indices = (x[:, j] != mask_token_id) & (x[:, j] != pad_token_id)
        x_[keep_indices, i[keep_indices]] = x[keep_indices, j]
        i += keep_indices

    max_len = i.max()
    return x_[:, :max_len]


def compute_insertion_probs(x, y, vocab_size, del_rate, pad_token_id, mask_token_id):

    batch_size = x.shape[0]
    max_len_x = x.shape[1]
    max_len_y = y.shape[1]

    # Compute effective lengths
    x_mask = x != pad_token_id
    y_mask = y != pad_token_id

    effective_len_x = x_mask.sum(axis=1)  # Shape: (batch_size,)
    effective_len_y = y_mask.sum(axis=1)  # Shape: (batch_size,)

    # Initialize pointers
    i = np.zeros(batch_size, dtype=np.long)
    j = np.zeros(batch_size, dtype=np.long)

    # initialize insertion probabilities
    insertion_cands = np.zeros(
        (
            batch_size,
            max_len_y + 1,
            vocab_size,
        ),
        dtype=np.float32,
    )
    # insertion_cands[:, :, mask_idx] = 1.0

    # print("ins probs shape")
    # print(insertion_cands.shape)

    processing = i < effective_len_x

    while processing.any():
        # Get batch indices for sequences being processed
        b_indices = processing.nonzero()[0].squeeze()

        # Ensure indices are tensors
        if len(b_indices.shape) == 0:
            b_indices = b_indices[np.newaxis]

        # For sequences where both i and j are valid
        i_valid = i[b_indices] < effective_len_x[b_indices]
        j_valid = j[b_indices] < effective_len_y[b_indices]
        both_valid = i_valid & j_valid

        if both_valid.any():
            bv_indices = b_indices[both_valid]
            x_i = x[bv_indices, i[bv_indices]]
            y_j = y[bv_indices, j[bv_indices]]

            matches = x_i == y_j
            match_indices = bv_indices[matches]

            i[match_indices] += 1
            j[match_indices] += 1

            mismatch_indices = bv_indices[~matches]
            insertion_cands[
                mismatch_indices,
                j[mismatch_indices],
                x[mismatch_indices, i[mismatch_indices]],
            ] += 1.0
            i[mismatch_indices] += 1

        j_done = (j[b_indices] == effective_len_y[b_indices]) & (
            i[b_indices] < effective_len_x[b_indices]
        )
        if j_done.any():
            jd_indices = b_indices[j_done]
            insertion_cands[
                jd_indices, j[jd_indices], x[jd_indices, i[jd_indices]]
            ] += 1
            i[jd_indices] += 1

        processing = i < effective_len_x

    # Compute actual insertion probabilities for the intermediate state
    # we interpret x as x_0, y as x_t. We want to compute the probabilities
    # that the tokens absent in x_t are present in x_{t-1}
    # del_rate is the independent probability of the token being deleted (a function of t-1, t handled elsewhere)
    # refer to my ipad notes for the derivation for the formula
    ins_rate = 1.0 - del_rate
    num_cands = insertion_cands.sum(axis=-1)
    token_ins_probs = ins_rate / (del_rate + num_cands * ins_rate)
    token_del_probs = del_rate / (del_rate + num_cands * ins_rate)
    insertion_probs = insertion_cands * token_ins_probs[:, :, np.newaxis]
    insertion_probs[:, :, mask_token_id] = token_del_probs

    loss_mask = y != pad_token_id

    return insertion_probs, loss_mask


def test_insertion_cands(x, y, vocab_size):
    i = 0
    j = 0
    insp = np.zeros((len(y) + 1, vocab_size))
    while i < len(x):
        if x[i] == y[j]:
            i += 1
            j += 1
        else:
            insp[j, x[i]] += 1
            i += 1
    return insp


import time

if __name__ == "__main__":
    # generate a batch x seq_len tensor of random integers
    batch_size = 1
    seq_len = 10
    vocab_size = 20

    start = time.time()
    # for i in range(1):
    x = np.random.randint(0, high=vocab_size - 1, size=(batch_size, seq_len))

    move_chance = 0.7
    # del_indices = np.random.rand(*x.shape) < move_chance
    # xt = np.full_like(x, -1, dtype=x.dtype)
    # for i in range(x.shape[0]):
    #     j = 0
    #     for k in range(x.shape[1]):
    #         if not del_indices[i, k]:
    #             xt[i, j] = x[i, k]
    #             j += 1

    # y = xt

    y = forward_deletions(x, move_chance)
    print(x)
    print(y)

    # alignments = find_alignments_with_padding(x[0].tolist(), xt[0].tolist())
    # alignments = greedy_alignment(x[0], xt[0])
    # print(x[0])
    # print(xt[0])
    # print(alignments)
    # print(x[0][alignments])
    # print(time.time() - start）

    start = time.time()
    ins_probs = compute_insertion_probs(x, y, vocab_size, 0.1, mask_idx=-1)
    print(x)
    print(y)
    print(ins_probs)

    # print(test_insertion_cands(x[0], y[0], vocab_size))

    print(time.time() - start)

    # for b in range(x.shape[0]):
    #     target = test_insertion_cands(x[b], y[b], vocab_size)
    #     print(target)
    #     assert np.allclose(
    #         (ins_probs[b] > 0), target
    #     ), f"Sequence {b}: Insertion probs mismatch"

    # for b in range(x.size(0)):
    #     kept_idx = alignments[b]
    #     x_seq = x[b]
    #     y_seq = y[b]

    #     # print(x_seq)
    #     # print(y_seq)
    #     # print(x_seq[kept_idx])

    #     effective_len_y = (y_seq != -1).sum().item()
    #     y_effective = y_seq[:effective_len_y]
    #     x_gathered = x_seq[kept_idx]
    #     assert torch.equal(
    #         x_gathered, y_effective
    #     ), f"Sequence {b}: Gathered x does not match y"
    #     print(f"Sequence {b}: Assertion passed.")

    # target_probs, target_mask = construct_target_probs_batch(
    #     x, y, alignments, vocab_size=11, mask_rate=0.1, mask_index=-1, unmask_rate=0.9
    # )

    # print(time.time() - start)

    # print(x[0])
    # print(y[0])
    # print(alignments[0])
    # print(target_probs[0])
    # print(target_mask[0])
