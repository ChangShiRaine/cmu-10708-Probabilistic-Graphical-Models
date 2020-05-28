def _viterbi_decode(self, emissions, mask):
    """Compute the viterbi algorithm to find the most probable sequence of labels
    given a sequence of emissions.
    Args:
        emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
        mask (Torch.FloatTensor): (batch_size, seq_len)
    Returns:
        torch.Tensor: the viterbi score for the for each batch.
            Shape of (batch_size,)
        list of lists of ints: the best viterbi sequence of labels for each batch
    """
    batch_size, seq_length, nb_labels = emissions.shape

    # in the first iteration, BOS will have all the scores and then, the max
    alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

    backpointers = []

    for i in range(1, seq_length):
        alpha_t = []
        backpointers_t = []

        for tag in range(nb_labels):

            # get the emission for the current tag and broadcast to all labels
            e_scores = emissions[:, i, tag]
            e_scores = e_scores.unsqueeze(1)

            # transitions from something to our tag and broadcast to all batches
            t_scores = self.transitions[:, tag]
            t_scores = t_scores.unsqueeze(0)

            # combine current scores with previous alphas
            scores = e_scores + t_scores + alphas

            # so far is exactly like the forward algorithm,
            # but now, instead of calculating the logsumexp,
            # we will find the highest score and the tag associated with it
            max_score, max_score_tag = torch.max(scores, dim=-1)

            # add the max score for the current tag
            alpha_t.append(max_score)

            # add the max_score_tag for our list of backpointers
            backpointers_t.append(max_score_tag)

        # create a torch matrix from alpha_t
        # (bs, nb_labels)
        new_alphas = torch.stack(alpha_t).t()
        alphas = new_alphas 

        # append the new backpointers
        backpointers.append(backpointers_t)

    # add the scores for the final transition
    last_transition = self.transitions[:, self.EOS_TAG_ID]
    end_scores = alphas + last_transition.unsqueeze(0)

    # get the final most probable score and the final most probable tag
    max_final_scores, max_final_tags = torch.max(end_scores, dim=1)



    # find the best sequence of labels for each sample in the batch
    best_sequences = []

    mask = torch.ones(emissions.shape[:2], dtype=torch.float)
    emission_lengths = mask.int().sum(dim=1)
    for i in range(batch_size):

        # recover the original sentence length for the i-th sample in the batch
        sample_length = emission_lengths[i].item()

        # recover the max tag for the last timestep
        sample_final_tag = max_final_tags[i].item()

        # limit the backpointers until the last but one
        # since the last corresponds to the sample_final_tag
        sample_backpointers = backpointers[: sample_length - 1]

        # follow the backpointers to build the sequence of labels
        sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

        # add this path to the list of best sequences
        best_sequences.append(sample_path)

    return max_final_scores, best_sequences



    def _find_best_path(self, sample_id, best_tag, backpointers):
    """Auxiliary function to find the best path sequence for a specific sample.
        Args:
            sample_id (int): sample index in the range [0, batch_size)
            best_tag (int): tag which maximizes the final score
            backpointers (list of lists of tensors): list of pointers with
            shape (seq_len_i-1, nb_labels, batch_size) where seq_len_i
            represents the length of the ith sample in the batch
        Returns:
            list of ints: a list of tag indexes representing the bast path
    """

    # add the final best_tag to our best path
    best_path = [best_tag]

    # traverse the backpointers in backwards
    for backpointers_t in reversed(backpointers):

        # recover the best_tag at this timestep
        best_tag = backpointers_t[best_tag][sample_id].item()

        # append to the beginning of the list so we don't need to reverse it later
        best_path.insert(0, best_tag)

    return best_path