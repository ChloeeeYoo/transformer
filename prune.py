import math
import re
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time

import torch.nn.utils.prune as prune


def load_record(path):
    f = open(path, 'r')
    losses = f.read()
    losses = re.sub('\\]', '', losses)
    losses = re.sub('\\[', '', losses)
    losses = re.sub('\\,', '', losses)
    losses = losses.split(' ')
    losses = [float(i) for i in losses]
    return losses, len(losses)


def load_weight(model):
    model = torch.load("./saved/model-3.661655992269516.pt")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



train_losses, train_count = load_record('./result/train_loss.txt')
test_losses, _ = load_record('./result/test_loss.txt')
bleus, _ = load_record('./result/bleu.txt')
epoch -= train_count

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)


print(f'The model has {count_parameters(model):,} trainable parameters')
#load_weight(model=model)

model.load_state_dict(torch.load("./saved/model-3.6249868869781494.pt"))

optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)



def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        #print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    
    
    parameters_to_prune = ()
    for i in range(6):
            parameters_to_prune += (
                (model.encoder.layers[i].attention.w_q, 'weight'),
                (model.encoder.layers[i].attention.w_k, 'weight'),
                (model.encoder.layers[i].attention.w_v, 'weight'),
               # (model.encoder.layers[i].attention.w_concat, 'weight'),
                (model.encoder.layers[i].ffn.linear1, 'weight'),
                (model.encoder.layers[i].ffn.linear2, 'weight'),
                (model.decoder.layers[i].self_attention.w_q, 'weight'),
                (model.decoder.layers[i].self_attention.w_k, 'weight'),
                (model.decoder.layers[i].self_attention.w_v, 'weight'),
               # (model.decoder.layers[i].self_attention.w_concat, 'weight'),
                (model.decoder.layers[i].enc_dec_attention.w_q, 'weight'),
                (model.decoder.layers[i].enc_dec_attention.w_k, 'weight'),
                (model.decoder.layers[i].enc_dec_attention.w_v, 'weight'),
               # (model.decoder.layers[i].enc_dec_attention.w_concat, 'weight'),
                (model.decoder.layers[i].ffn.linear1, 'weight'),
                (model.decoder.layers[i].ffn.linear2, 'weight'),
 
            )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.7,
    )

    for i in range(6):
        print( #encoder_attention 
            "Sparsity in e_layer{}.w_q: {:.2f}%".format(i,
                100. * float(torch.sum(model.encoder.layers[i].attention.w_q.weight == 0))
                / float(model.encoder.layers[i].attention.w_q.weight.nelement())
            )
        )
        print(
            "Sparsity in layer{}.w_k: {:.2f}%".format(i,
                100. * float(torch.sum(model.encoder.layers[i].attention.w_k.weight == 0))
                / float(model.encoder.layers[i].attention.w_k.weight.nelement())
            )
        )
        print(
            "Sparsity in layer{}.w_v: {:.2f}%".format(i,
                100. * float(torch.sum(model.encoder.layers[i].attention.w_v.weight == 0))
                / float(model.encoder.layers[i].attention.w_v.weight.nelement())
            )
        )
        print(
            "Sparsity in layer{}.w_concat: {:.2f}%".format(i,
                100. * float(torch.sum(model.encoder.layers[i].attention.w_concat.weight == 0))
                / float(model.encoder.layers[i].attention.w_concat.weight.nelement())
            )
        )
        print( #encoder_ffn
            "Sparsity in layer{}.ffn layer1: {:.2f}%".format(i,
                100. * float(torch.sum(model.encoder.layers[i].ffn.linear1.weight == 0))
                / float(model.encoder.layers[i].ffn.linear1.weight.nelement())
            )
        )
        print(
            "Sparsity in layer{}.ffn layer2: {:.2f}%".format(i,
                100. * float(torch.sum(model.encoder.layers[i].ffn.linear2.weight == 0))
                / float(model.encoder.layers[i].ffn.linear2.weight.nelement())
            )
        )
        print( #decoder_self_attention
            "Sparsity in d_self_layer{}.w_q: {:.2f}%".format(i,
                100. * float(torch.sum(model.decoder.layers[i].self_attention.w_q.weight == 0))
                / float(model.decoder.layers[i].self_attention.w_q.weight.nelement())
            )
        )
        print(
            "Sparsity in layer{}.w_k: {:.2f}%".format(i,
                100. * float(torch.sum(model.decoder.layers[i].self_attention.w_k.weight == 0))
                / float(model.decoder.layers[i].self_attention.w_k.weight.nelement())
            )
        )
        print(
            "Sparsity in layer{}.w_v: {:.2f}%".format(i,
                100. * float(torch.sum(model.decoder.layers[i].self_attention.w_v.weight == 0))
                / float(model.decoder.layers[i].self_attention.w_v.weight.nelement())
            )
        )
        print(
            "Sparsity in layer{}.w_concat: {:.2f}%".format(i,
                100. * float(torch.sum(model.decoder.layers[i].self_attention.w_concat.weight == 0))
                / float(model.decoder.layers[i].self_attention.w_concat.weight.nelement())
            )
        )
        print( #decoder_enc_dec_attention
            "Sparsity in d_enc_dec_layer{}.w_q: {:.2f}%".format(i,
                100. * float(torch.sum(model.decoder.layers[i].enc_dec_attention.w_q.weight == 0))
                / float(model.decoder.layers[i].enc_dec_attention.w_q.weight.nelement())
            )
        )
        print(
            "Sparsity in layer{}.w_k: {:.2f}%".format(i,
                100. * float(torch.sum(model.decoder.layers[i].enc_dec_attention.w_k.weight == 0))
                / float(model.decoder.layers[i].enc_dec_attention.w_k.weight.nelement())
            )
        )
        print(
            "Sparsity in layer{}.w_v: {:.2f}%".format(i,
                100. * float(torch.sum(model.decoder.layers[i].enc_dec_attention.w_v.weight == 0))
                / float(model.decoder.layers[i].enc_dec_attention.w_v.weight.nelement())
            )
        )
        print(
            "Sparsity in layer{}.w_concat: {:.2f}%".format(i,
                100. * float(torch.sum(model.decoder.layers[i].enc_dec_attention.w_concat.weight == 0))
                / float(model.decoder.layers[i].enc_dec_attention.w_concat.weight.nelement())
            )
        )
        print( #decoder_ffn
            "Sparsity in layer{}.ffn layer1: {:.2f}%".format(i,
                100. * float(torch.sum(model.decoder.layers[i].ffn.linear1.weight == 0))
                / float(model.decoder.layers[i].ffn.linear1.weight.nelement())
            )
        )
        print(
            "Sparsity in layer{}.ffn layer2: {:.2f}%".format(i,
                100. * float(torch.sum(model.decoder.layers[i].ffn.linear2.weight == 0))
                / float(model.decoder.layers[i].ffn.linear2.weight.nelement())
            )
        )
       
   # module = model.encoder.layers[0].attention.w_v
   # prune.l1_unstructured(module, name='weight', amount=0.3)
   # print(list(module.named_parameters()))
   # print(list(module.named_buffers()))
   # print(module.weight)
   


    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1 + train_count} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
