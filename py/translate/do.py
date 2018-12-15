import torch
from torch import optim
import numpy as np
import time
from tqdm import tqdm
from translate.masked_nll_loss import masked_nll_loss
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')
import seaborn as sns
import pickle as pkl
import translate.beam
import translate.BLEU


def print_sen(sen_idx, lang):
    '''
    Given a list of indicies, print the sentence
    :param sen_idx: A list of token indicies
    :param lang: A language class
    :return: no return, print a sentence
    '''
    sen = [lang.id2token[i] for i in sen_idx if (i != PAD_IDX and i != EOS_IDX)]
    print(' '.join(sen))


def train_batch(args, x, y, encoder, decoder, log, encoder_optimizer, decoder_optimizer, src_lang, tgt_lang,
                teacher_forcing_ratio=1, display_ex=False, epoch=None):
    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    if args.model == 'selfenc':
        encoder_outputs, encoder_hidden = encoder(x['sen'], x['mask'].unsqueeze(-2))
    else:
        encoder_outputs, encoder_hidden = encoder(x['sen'], x['sen_len'])

    batch_size = len(y['sen_len'])
    max_target_length = max(y['sen_len'])
    # Initialize the decoder input

    decoder_input = torch.LongTensor([SOS_IDX] * batch_size).unsqueeze(1).to(device)
    decoder_hidden = encoder_hidden  # L x B x H
    use_teacher_forcing = np.random.rand() < teacher_forcing_ratio

    if use_teacher_forcing:
        decoder_input = torch.cat((decoder_input, y['sen'][:, :-1]), dim=1)  # B x tgt_len
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs, encoder_hidden,  x['mask']
        )

        loss = masked_nll_loss(
            decoder_output,  # B x tgt_len x output_vocab
            y['sen'],  # -> B x tgt_len
            y['sen_len'],
            y['mask']
        )
    else:  # no teacher forcing
        raise NotImplementedError('Stick with teacher forcing for now')
        all_decoder_outputs = torch.zeros(batch_size, max_target_length, decoder.output_size).to(device)
        all_decoder_attn = torch.zeros(batch_size, max_target_length, max(x['sen_len'])).to(device)
        y_hat_mask = torch.zeros(*y['mask'].size()).byte().to(device)
        y_hat_eos = torch.ByteTensor([False] * batch_size).to(device)
        y_hat = torch.zeros(*y['sen'].size()).long().to(device)
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs, x['mask']
            )
            all_decoder_outputs[:, t, :] = decoder_output.squeeze(1)  # Store this step's outputs
            all_decoder_attn[:, t, :] = decoder_attn.squeeze(1)
            _, topi = decoder_output.topk(1, dim=2)  # B x tgt_len x output_vocab
            decoder_input = topi.squeeze().detach()
            y_hat[:, t] = decoder_input
            y_hat_eos = y_hat_eos | ((decoder_input == EOS_IDX) & (y['sen'][:, t] == EOS_IDX))
            y_hat_mask[:, t] = ~y_hat_eos
            decoder_input.unsqueeze_(1)

        loss = masked_nll_loss(
            all_decoder_outputs,  # B x tgt_len x output_vocab
            y['sen'],  # -> B x tgt_len
            y_hat_mask.sum(dim=1),
            y_hat_mask
        )

    # TODO Implement Real Non-Teacher Forcing

    loss = torch.sum(loss)
    loss.backward()
    loss_value = loss.item()
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.1)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.1)
    encoder_optimizer.step()
    decoder_optimizer.step()

    try:
        if display_ex:
            # print('TRAINING epoch {}'.format(log.epochs))
            print('Teacher forcing example')
            print('<REFERENCE>: ',end='')
            print_sen(y['sen'][0], tgt_lang)
            print('<PREDICT>  : ', end = '')
            _, y_hat = decoder_output.topk(1, 2)
            y_hat = y_hat.squeeze(2).masked_fill(1 - y['mask'], 0).detach()
            print_sen(y_hat[0], tgt_lang)
            # plt.figure()
            # sns.heatmap(decoder_attn[0].detach().to('cpu').numpy())
            # plt.savefig('Train-{}.png'.format(epoch))
            # plt.close()
    except:
        print('Print fail')

    del decoder_output
    del decoder_hidden
    del decoder_attn

    return loss_value / batch_size  # sum of a batch


# def val_batch(x, y, encoder, decoder, rtn_ex=False, epoch=None):
#     encoder.eval()
#     decoder.eval()
#     batch_size = len(y['sen_len'])
#     with torch.no_grad():
#         encoder_outputs, encoder_hidden = encoder(x['sen'], x['sen_len'])
#         # Val is different from eval: there is no target length for eval
#         max_target_length = max(y['sen_len'])
#         # Initialize the decoder input
#         decoder_input = torch.LongTensor([SOS_IDX] * batch_size).unsqueeze(1).to(device)
#         decoder_hidden = encoder_hidden  # L x B x H
#         all_decoder_outputs = torch.zeros(batch_size, max_target_length, decoder.output_size).to(device)
#         all_decoder_attn = torch.zeros(batch_size, max_target_length, max(x['sen_len'])).to(device)
#         y_hat_mask = torch.zeros(*y['mask'].size()).byte().to(device)
#         y_hat_eos = torch.ByteTensor([False] * batch_size).to(device)
#         y_hat = torch.zeros(*y['sen'].size()).long().to(device)
#         for t in range(max_target_length):
#             decoder_output, decoder_hidden, decoder_attn = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs, x['mask']
#             )
#             all_decoder_outputs[:, t, :] = decoder_output.squeeze(1)  # Store this step's outputs
#             all_decoder_attn[:, t, :] = decoder_attn.squeeze(1)
#             _, topi = decoder_output.topk(1, dim=2)  # B x tgt_len x output_vocab
#             # TODO When not using teacher forcing, if decoder outputs <EOS> stops (or change y_mask)
#             decoder_input = topi.squeeze().detach()
#             y_hat[:, t] = decoder_input
#             y_hat_eos = y_hat_eos | ((decoder_input == EOS_IDX) & (y['sen'][:, t] == EOS_IDX))
#             y_hat_mask[:, t] = ~y_hat_eos
#             decoder_input.unsqueeze_(1)
#
#         loss = masked_nll_loss(
#             all_decoder_outputs,  # B x tgt_len x output_vocab
#             y['sen'],  # -> B x tgt_len
#             y_hat_mask.sum(dim=1),
#             y_hat_mask
#         )
#         loss = torch.sum(loss).item()
#     if rtn_ex == True:
#         plt.figure()
#         sns.heatmap(all_decoder_attn[1].detach().to('cpu').numpy())
#         plt.savefig('Val-{}.png'.format(epoch))
#         plt.close()
#         # pdb.set_trace()
#         return loss / batch_size, y['sen'], y_hat.masked_fill_(1 - y_hat_mask, 0)
#     return loss / batch_size

def val_epoch(args, cons, val_loader, encoder, decoder, src_lang, tgt_lang, display_ex = True):
    res, ref_streams, sys_stream = translate.beam.do_beam_translate(args, cons, encoder, decoder, val_loader,
                                                                    src_lang, tgt_lang, beam_size=5, output=False)
    print('VALIDATION  with beam size of 5')
    if display_ex:
        print('<SOURCE>   : ' + ' '.join(res[0][0]))
        print('<REFERENCE>: ' + ' '.join(res[0][1]))
        print('<PREDICT>  : ' + ' '.join(res[0][2]))
        print(' ')
        print('<SOURCE>   : ' + ' '.join(res[50][0]))
        print('<REFERENCE>: ' + ' '.join(res[50][1]))
        print('<PREDICT>  : ' + ' '.join(res[50][2]))
    bleu = translate.BLEU.corpus_bleu(sys_stream, ref_streams, smooth='none')
    print('BLEU SCORE: {:.6f}'.format(bleu.score))
    return bleu.score, res

def val_plot_align(args, cons, val_loader, encoder, decoder, src_lang, tgt_lang):
    global device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX
    device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = cons
    zhfont = matplotlib.font_manager.FontProperties(fname=r'../../simhei.ttf', size=14)
    loss_total = 0
    for i, (x, y) in enumerate(val_loader):
        encoder.eval()
        decoder.eval()
        batch_size = len(y['sen_len'])
        with torch.no_grad():
            encoder_outputs, encoder_hidden = encoder(x['sen'], x['sen_len'])
            # Val is different from eval: there is no target length for eval
            max_target_length = max(y['sen_len'])
            # Initialize the decoder input
            decoder_input = torch.LongTensor([SOS_IDX] * batch_size).unsqueeze(1).to(device)
            decoder_hidden = encoder_hidden  # L x B x H
            all_decoder_outputs = torch.zeros(batch_size, max_target_length, decoder.output_size).to(device)
            all_decoder_attn = torch.zeros(batch_size, max_target_length, max(x['sen_len'])).to(device)
            y_hat_mask = torch.zeros(*y['mask'].size()).byte().to(device)
            y_hat_eos = torch.ByteTensor([False] * batch_size).to(device)
            y_hat = torch.zeros(*y['sen'].size()).long().to(device)
            for t in range(max_target_length):
                decoder_output, decoder_hidden, decoder_attn = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, encoder_hidden, x['mask']
                )
                all_decoder_outputs[:, t, :] = decoder_output.squeeze(1)  # Store this step's outputs
                all_decoder_attn[:, t, :] = decoder_attn.squeeze(1)
                _, topi = decoder_output.topk(1, dim=2)  # B x tgt_len x output_vocab
                decoder_input = topi.squeeze().detach()
                y_hat[:, t] = decoder_input
                y_hat_eos = y_hat_eos | ((decoder_input == EOS_IDX) & (y['sen'][:, t] == EOS_IDX))
                y_hat_mask[:, t] = ~y_hat_eos
                decoder_input.unsqueeze_(1)

            loss = masked_nll_loss(
                all_decoder_outputs,  # B x tgt_len x output_vocab
                y['sen'],  # -> B x tgt_len
                y_hat_mask.sum(dim=1),
                y_hat_mask
            )
            loss = torch.sum(loss).item()
        if True:
            plt.figure()
            ind = 2
            src_sen = x['sen'][ind]
            tmpx, tmpy = all_decoder_outputs[ind].topk(1, dim=1)
            tgt_sen = tmpy.squeeze()

            src_len = x['sen_len'][ind]
            try:
                tgt_len = ([bool(i == EOS_IDX) for i in tgt_sen]).index(True)
            except:
                tgt_len = len(tgt_sen)

            tgt_sen = tgt_sen[:tgt_len]
            src_sen = src_sen[:src_len]

            print(' '.join([src_lang.id2token[i] for i in src_sen]))

            fig, ax = plt.subplots(figsize=(10, 8))
            plt.rcParams['font.sans-serif'] = ['simhei']

            sns.heatmap(all_decoder_attn[ind][:tgt_len, :src_len].detach().to('cpu').numpy(),
                        xticklabels = [src_lang.id2token[i] for i in src_sen],
                        yticklabels = [tgt_lang.id2token[i] for i in tgt_sen],
                        cmap = 'gray',  ax=ax)

            spec = (args.language, args.model, args.rnn)

            plt.savefig('../display/{}_{}_{}_align.eps'.format(*spec))
            plt.savefig('../display/{}_{}_{}_align.png'.format(*spec))
            plt.close()
            # pdb.set_trace()
        loss =  loss / batch_size
        loss_total += float(loss)
        break
    return






def do_train(args, cons, train_loader, val_loader, encoder, decoder, encoder_optimizer, decoder_optimizer,
             log, src_lang, tgt_lang, n_epochs, learning_rate=0.00001):
    global device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX
    device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = cons

    print_loss_total = 0  # Reset every print_every

    for epoch in range(n_epochs):
        print('TRAINING epoch = {}'.format(log.next_epoch()))
        len_loader = len(train_loader)

        for i, (x, y) in enumerate(tqdm(train_loader)):
            loss = train_batch(args, x, y, encoder, decoder, log, encoder_optimizer,
                               decoder_optimizer, src_lang, tgt_lang, teacher_forcing_ratio=1, display_ex = i == len_loader - 1)
            torch.cuda.empty_cache()
            print_loss_total += float(loss)

        print_loss_avg = print_loss_total / len(train_loader)
        print_loss_total = 0
        print('train loss: {:.6f}'.format(print_loss_avg))

        bleu_score, _ = val_epoch(args, cons, val_loader, encoder, decoder, src_lang, tgt_lang)

        log.update(print_loss_avg, bleu_score)

        if not args.readonly:
            spec = (args.language, args.model, args.rnn)
            torch.save(encoder.state_dict(), '../state/{}_{}_{}/encoder_state.pkl'.format(*spec))
            torch.save(decoder.state_dict(), '../state/{}_{}_{}/decoder_state.pkl'.format(*spec))
            torch.save(encoder_optimizer.state_dict(), '../state/{}_{}_{}/encoder_optimizer_state.pkl'.format(*spec))
            torch.save(decoder_optimizer.state_dict(), '../state/{}_{}_{}/decoder_optimizer_state.pkl'.format(*spec))
            pkl.dump(log, open('../state/{}_{}_{}/log.pkl'.format(*spec), 'wb'), pkl.HIGHEST_PROTOCOL)
    return

def eval(args, cons, loader, encoder, decoder, src_lang, tgt_lang, display_ex = True, group_by_len = False):
    global device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX
    device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = cons
    if not args.output:
        res, ref_streams, can_stream = translate.beam.do_beam_translate(args, cons, encoder, decoder, loader,
                                                                    src_lang, tgt_lang, beam_size=args.evalbeam, output=False)
    else:
        res, ref_streams, can_stream, print_res = translate.beam.do_beam_translate(args, cons, encoder, decoder, loader,
                                                                        src_lang, tgt_lang, beam_size=args.evalbeam, output=True)
        spec = (args.language, args.model, args.rnn)
        print(print_res, file = open('../display/{}_{}_{}_eval_translate.txt'.format(*spec), 'w'))
        bleu = translate.BLEU.corpus_bleu(can_stream, ref_streams, smooth='none')
        print('BLEU SCORE: {:.6f}'.format(bleu.score))
        return bleu.score, res
    print('EVALUATION  with beam size of {}'.format(args.evalbeam))
    if display_ex:
        print('<SOURCE>   : ' + ' '.join(res[0][0]))
        print('<REFERENCE>: ' + ' '.join(res[0][1]))
        print('<PREDICT>  : ' + ' '.join(res[0][2]))
        print(' ')
        print('<SOURCE>   : ' + ' '.join(res[50][0]))
        print('<REFERENCE>: ' + ' '.join(res[50][1]))
        print('<PREDICT>  : ' + ' '.join(res[50][2]))
    if group_by_len:
        streams_len = len(ref_streams)
        ref_len = [len(t) for t in ref_streams]
        bins = [(i, i+3) for i in range(0, 45)]
        bins.append((45, float('inf')))
        bleu_grouped = dict()
        for bin in bins:
            ref_inbin = [l >= bin[0] and l < bin[1] for l in ref_len]
            ref_grouped = [ref_streams[i] for i in range(streams_len) if ref_inbin[i]]
            can_grouped = [can_stream[i] for i in range(streams_len) if ref_inbin[i]]
            bleu = translate.BLEU.corpus_bleu(can_grouped, ref_grouped, smooth='none')
            bleu_grouped[(bin[0] + bin[1]) / 2.0] = bleu.score
        spec = (args.language, args.model, args.rnn)
        pkl.dump(bleu_grouped, open('../display/{}_{}_{}_bleu_grouped.pkl'.format(*spec), 'wb'), pkl.HIGHEST_PROTOCOL)

    bleu = translate.BLEU.corpus_bleu(can_stream, ref_streams, smooth='none')
    print('BLEU SCORE: {:.6f}'.format(bleu.score))
    return bleu.score, res
