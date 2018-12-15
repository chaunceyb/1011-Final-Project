# based on OpenNMT

import torch
from tqdm import tqdm


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size):
        """Initialize params."""
        self.size = size
        self.done = False
        # The score for each translation on the beam.
        self.scores = torch.FloatTensor(size).zero_().to(device)  # K : P(Y1 Y2 ... Y_Step)

        # The backpointers at each time-step. i.e. which beam does the current come from
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [torch.LongTensor(size).fill_(PAD_IDX).to(device)]
        self.nextYs[0][0] = SOS_IDX

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam. i.e. what lastest Y do the beams predict"""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def advance(self, workd_lk):  # K x C
        """Advance the beam."""
        num_words = workd_lk.size(1)  # C

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)  # K -> K x 1 -> K x C
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words  # Get beam number
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)  # Get C (token index) number

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == EOS_IDX:
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]

def do_beam_translate(args, cons, encoder, decoder, dataloader, src_lang, tgt_lang,
                      beam_size, output = False, max_len=100):
    global device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX
    device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = cons

    def read_source(id_list):
        ls = [src_lang.id2token[i] for i in id_list if i != PAD_IDX]
        return ls

    def read_target(id_list):
        ls = [tgt_lang.id2token[i] for i in id_list if i != UNK_IDX]
        try:
            leng = ls.index('<EOS>')
            ls = ls[:leng]
        except:
            pass
        return ls

    encoder.eval()
    decoder.eval()
    # Use a list to store all output tuple(source_list, target_org_list, pred_list)
    res = []
    print_res = ''
    ref_streams = []
    sys_stream = []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            if args.model == 'selfenc':
                encoder_outputs, encoder_hidden = encoder(x['sen'], x['mask'].unsqueeze(-2))
            else:
                encoder_outputs, encoder_hidden = encoder(x['sen'], x['sen_len'])
            n_layers = decoder.n_layers

            bsize = len(y['sen'])
            beam = [Beam(beam_size) for x in range(bsize)]
            # Expand beam_size times
            if args.rnn == 'gru':
                encoder_hidden.unsqueeze_(2)
                encoder_hidden = encoder_hidden.repeat(1, 1, beam_size, 1).view(n_layers, bsize * beam_size, -1)
                decoder_hidden = encoder_hidden
            elif args.rnn == 'lstm':
                [i.unsqueeze_(2) for i in encoder_hidden]
                encoder_hidden = [i.repeat(1, 1, beam_size, 1).view(n_layers, bsize * beam_size, -1) for i in encoder_hidden]
                decoder_hidden = encoder_hidden
            encoder_output_len = encoder_outputs.size(1)
            encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, beam_size, 1, 1).view(bsize * beam_size,
                                                                                      encoder_output_len, -1)

            encoder_mask = (x['mask']).unsqueeze(1).repeat(1, beam_size, 1).view(bsize * beam_size, -1)

            for t in range(max_len):
                if all([b.done for b in beam]): break  # Stop before max_len
                decoder_input = torch.stack([b.get_current_state() for b in beam]).view(-1).unsqueeze(1)
                y_hat, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs, encoder_hidden, encoder_mask)
                y_hat = y_hat.squeeze(1).view(bsize, beam_size, -1)
                if args.rnn == 'gru':
                    decoder_hidden = decoder_hidden.view(n_layers, bsize, beam_size, -1)
                    for j, b in enumerate(beam):
                        b.advance(y_hat[j])
                        decoder_hidden[:, j] = decoder_hidden[:, j, b.get_current_origin()]
                    decoder_hidden = decoder_hidden.view(n_layers, bsize * beam_size, -1)
                elif args.rnn == 'lstm':
                    decoder_hidden = [i.view(n_layers, bsize, beam_size, -1) for i in decoder_hidden]
                    for j, b in enumerate(beam):
                        b.advance(y_hat[j])
                        decoder_hidden[0][:, j] = decoder_hidden[0][:, j, b.get_current_origin()]
                        decoder_hidden[1][:, j] = decoder_hidden[1][:, j, b.get_current_origin()]
                    decoder_hidden = [i.view(n_layers, bsize * beam_size, -1) for i in decoder_hidden]
            pred = [b.get_hyp(0) for b in beam]
            this_res = [(read_source(x),
                     y.split(),
                     read_target(z)) \
                    for (x, y, z,) in
                    zip(x['sen'], y['sen_org'], pred)]
            ref_streams += [y.split() for y in y['sen_org']]
            sys_stream += [read_target(z) for z in pred]
            res += this_res
            if output:
                for tup in this_res:
                    print_res += '\n\nSOURCE:  ' + ' '.join(tup[0])
                    print_res += '\nTARGET:  ' + ' '.join(tup[1])
                    print_res += '\nPREDICT: ' + ' '.join(tup[2])

    if output:
        return res, ref_streams, sys_stream, print_res
    else:
        return res, ref_streams, sys_stream