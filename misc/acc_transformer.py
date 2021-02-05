"""
Author:		 Jey Han Lau
Date:		   Jul 19
"""

import torch
import numpy as np
from transformers import XLNetTokenizer, XLNetLMHeadModel
from scipy.special import softmax
from argparse import ArgumentParser

def config():
    parser = ArgumentParser()
    # basic
    parser.add_argument('--file_dir', type=str, default=None, help="data directory")
    parser.add_argument('--ids_file', type=str, default=None, help="list of ids to eval")
    parser.add_argument('--id', type=str, default=None, help="single setting to evaluate")
    parser.add_argument('--parsed_file', type=str, default=None, help='')
    parser.add_argument('--accept_name', type=str, default='xlnet', help='bert or xlnet')

    args = parser.parse_args()

    model_name = 'xlnet-large-cased'
    args.tokenizer = XLNetTokenizer.from_pretrained(model_name)
    args.acpt_model = XLNetLMHeadModel.from_pretrained(model_name)

    args.device = torch.device('cuda:0')
    args.acpt_model.to(args.device)
    args.acpt_model.eval()

    return args

# global
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> """


def readlines(file_name):
    ret = []
    for line in open(file_name, 'r'):
        ret.append(line.strip())
    return ret

# ###########
# # functions#
# ###########
# def model_score(tokenize_input, tokenize_context, model, tokenizer, device, model_name='xlnet', use_context=False):
#     if model_name.startswith("gpt"):
#         if not use_context:
#             # prepend the sentence with <|endoftext|> token, so that the loss is computed correctly
#             tensor_input = torch.tensor([[50256] + tokenizer.convert_tokens_to_ids(tokenize_input)], device=device)
#             loss = model(tensor_input, labels=tensor_input)
#         else:
#             tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_context + tokenize_input)], device=device)
#             labels = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_context + tokenize_input)], device=device)
#             # -1 label for context (loss not computed over these tokens)
#             labels[:, :len(tokenize_context)] = -1
#             loss = model(tensor_input, labels=labels)
#         return float(loss[0]) * -1.0 * len(tokenize_input)
#     elif model_name.startswith("bert"):
#         batched_indexed_tokens = []
#         batched_segment_ids = []
#         if not use_context:
#             tokenize_combined = ["[CLS]"] + tokenize_input + ["[SEP]"]
#         else:
#             tokenize_combined = ["[CLS]"] + tokenize_context + tokenize_input + ["[SEP]"]
#         for i in range(len(tokenize_input)):
#             # Mask a token that we will try to predict back with `BertForMaskedLM`
#             masked_index = i + 1 + (len(tokenize_context) if use_context else 0)
#             tokenize_masked = tokenize_combined.copy()
#             tokenize_masked[masked_index] = '[MASK]'
#
#             # Convert token to vocabulary indices
#             indexed_tokens = tokenizer.convert_tokens_to_ids(tokenize_masked)
#             # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
#             segment_ids = [0] * len(tokenize_masked)
#
#             batched_indexed_tokens.append(indexed_tokens)
#             batched_segment_ids.append(segment_ids)
#
#         # Convert inputs to PyTorch tensors
#         tokens_tensor = torch.tensor(batched_indexed_tokens, device=device)
#         segment_tensor = torch.tensor(batched_segment_ids, device=device)
#
#         # Predict all tokens
#         with torch.no_grad():
#             outputs = model(tokens_tensor, token_type_ids=segment_tensor)
#             predictions = outputs[0]
#         # go through each word and sum their logprobs
#         lp = 0.0
#         for i in range(len(tokenize_input)):
#             masked_index = i + 1 + (len(tokenize_context) if use_context else 0)
#             predicted_score = predictions[i, masked_index]
#             predicted_prob = softmax(predicted_score.cpu().numpy())
#             lp += np.log(predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])[0]])
#         return lp
#     elif model_name.startswith("xlnet"):
#         xlnet_bidir=True
#         tokenize_ptext = tokenizer.tokenize(PADDING_TEXT.lower())
#         if not use_context:
#             tokenize_input2 = tokenize_ptext + tokenize_input
#         else:
#             tokenize_input2 = tokenize_ptext + tokenize_context + tokenize_input
#         # go through each word and sum their logprobs
#         lp = 0.0
#         for max_word_id in range((len(tokenize_input2) - len(tokenize_input)), (len(tokenize_input2))):
#             sent = tokenize_input2[:]
#             input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(sent)], device=device)
#             perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
#             # if not bidir, mask target word + right/future words
#             if not xlnet_bidir:
#                 perm_mask[:, :, max_word_id:] = 1.0
#                 # if bidir, mask only the target word
#             else:
#                 perm_mask[:, :, max_word_id] = 1.0
#
#             target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
#             target_mapping[0, 0, max_word_id] = 1.0
#             with torch.no_grad():
#                 outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
#                 next_token_logits = outputs[0]
#
#             word_id = tokenizer.convert_tokens_to_ids([tokenize_input2[max_word_id]])[0]
#             predicted_prob = softmax((next_token_logits[0][-1]).cpu().numpy())
#             lp += np.log(predicted_prob[word_id])
#         return lp

def bert_model_score(tokenize_input, args):
    batched_indexed_tokens = []
    batched_segment_ids = []
    # if not args.use_context:
    tokenize_combined = ["[CLS]"] + tokenize_input + ["[SEP]"]
    # else:
    #     tokenize_combined = ["[CLS]"] + tokenize_context + tokenize_input + ["[SEP]"]
    for i in range(len(tokenize_input)):
        # Mask a token that we will try to predict back with `BertForMaskedLM`
        masked_index = i + 1
        tokenize_masked = tokenize_combined.copy()
        tokenize_masked[masked_index] = '[MASK]'

        # Convert token to vocabulary indices
        indexed_tokens = args.tokenizer.convert_tokens_to_ids(tokenize_masked)
        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        segment_ids = [0] * len(tokenize_masked)

        batched_indexed_tokens.append(indexed_tokens)
        batched_segment_ids.append(segment_ids)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor(batched_indexed_tokens, device=args.device)
    segment_tensor = torch.tensor(batched_segment_ids, device=args.device)

    # Predict all tokens
    with torch.no_grad():
        outputs = args.acpt_model(tokens_tensor, token_type_ids=segment_tensor)
        predictions = outputs[0]
    # go through each word and sum their logprobs
    lp = 0.0
    for i in range(len(tokenize_input)):
        masked_index = i + 1
        predicted_score = predictions[i, masked_index]
        predicted_prob = softmax(predicted_score.cpu().numpy())
        lp += np.log(predicted_prob[args.tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])[0]])
    return lp

def xlnet_model_score(tokenize_input, args):
    xlnet_bidir = True
    tokenize_context = None
    tokenize_ptext = args.tokenizer.tokenize(PADDING_TEXT.lower())
    # if not args.use_context:
    tokenize_input2 = tokenize_ptext + tokenize_input
    # else:
    # tokenize_input2 = tokenize_ptext + tokenize_context + tokenize_input
    # go through each word and sum their logprobs
    lp = 0.0
    for max_word_id in range((len(tokenize_input2) - len(tokenize_input)), (len(tokenize_input2))):
        sent = tokenize_input2[:]
        input_ids = torch.tensor([args.tokenizer.convert_tokens_to_ids(sent)], device=args.device)
        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=args.device)
        # if not bidir, mask target word + right/future words
        if not xlnet_bidir:
            perm_mask[:, :, max_word_id:] = 1.0
            # if bidir, mask only the target word
        else:
            perm_mask[:, :, max_word_id] = 1.0

        target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=args.device)
        target_mapping[0, 0, max_word_id] = 1.0
        with torch.no_grad():
            outputs = args.acpt_model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
            next_token_logits = outputs[0]

        word_id = args.tokenizer.convert_tokens_to_ids([tokenize_input2[max_word_id]])[0]
        predicted_prob = softmax((next_token_logits[0][-1]).cpu().numpy())
        lp += np.log(predicted_prob[word_id])
    return lp

def evaluate_accept(references, translations, args):

    scores = []

    for ref, translation in zip(references, translations):
        tokenize_input = args.tokenizer.tokenize(ref)
        text_len = len(tokenize_input)
        if args.accept_name == 'bert':
            # compute sentence logprob
            lp = bert_model_score(tokenize_input, args)
        elif args.accept_name == 'xlnet':
            lp = xlnet_model_score(tokenize_input, args)
            # acceptability measures
        penalty = ((5 + text_len) ** 0.8 / (5 + 1) ** 0.8)
        score_ref = lp / penalty
        # print(score_ref)

        tokenize_input = args.tokenizer.tokenize(translation)
        text_len = len(tokenize_input)
        if args.accept_name == 'bert':
            # compute sentence logprob
            lp = bert_model_score(tokenize_input, args)
        elif args.accept_name == 'xlnet':
            lp = xlnet_model_score(tokenize_input, args)
        # acceptability measures
        penalty = ((5 + text_len) ** 0.8 / (5 + 1) ** 0.8)
        score_trans = lp / penalty
        # print(score_trans)
        scores.append(score_trans-score_ref)

    return scores


# ######
# # main#
# ######
# def main(args):
#     # system scores
#     mean_lps = []
#     # Load pre-trained model and tokenizer
#     args.model_name = 'bert-base-uncased'
#     if args.model_name.startswith("gpt"):
#         model = GPT2LMHeadModel.from_pretrained(args.model_name)
#         tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
#     elif args.model_name.startswith("bert"):
#         model = BertForMaskedLM.from_pretrained(args.model_name)
#         tokenizer = BertTokenizer.from_pretrained(args.model_name,
#                                                   do_lower_case=(True if "uncased" in args.model_name else False))
#     elif args.model_name.startswith("xlnet"):
#         tokenizer = XLNetTokenizer.from_pretrained(args.model_name)
#         model = XLNetLMHeadModel.from_pretrained(args.model_name)
#     else:
#         print("Supported models: gpt, bert and xlnet only.")
#         raise SystemExit
#
#     # put model to device (GPU/CPU)
#     device = torch.device(args.device)
#     model.to(device)
#
#     # eval mode; no dropout
#     model.eval()
#
#     text = "400 was great ! he was very helpful and i incubates comparitive back and visit again . i really playmobile the theme and the staff was very friendly ! a herb to guppy !"
#
#     tokenize_input = tokenizer.tokenize(text)
#     text_len = len(tokenize_input)
#
#     # compute sentence logprob
#     lp = model_score(tokenize_input, None, model, tokenizer, device, args.model_name)
#
#     # acceptability measures
#     penalty = ((5 + text_len) ** 0.8 / (5 + 1) ** 0.8)
#     print("score=", lp / penalty)



if __name__ == "__main__":
    args = config()

    if args.ids_file is not None:
        ids = readlines(args.ids_file)
        for id in ids:
            comps = id.split()
            src_sents = readlines(args.file_dir+'/'+comps[1]+'/s2s_output/src_changed.txt')
            tgt_sents = readlines(args.file_dir+'/'+comps[1]+'/s2s_output/adv_changed.txt')
            scores = evaluate_accept(src_sents, tgt_sents, args)
            print(comps[0]+': '+str(sum(scores)/len(scores)))
            with open(args.file_dir+'/'+comps[1]+'/s2s_output/accept_changed.txt', 'w') as output_file:
                for score in scores:
                    output_file.write(str(score)+'\n')
    elif args.id is not None:
        src_sents = readlines(args.file_dir + '/' + args.id + '/s2s_output/src_changed.txt')
        tgt_sents = readlines(args.file_dir + '/' + args.id + '/s2s_output/adv_changed.txt')
        scores = evaluate_accept(src_sents, tgt_sents, args)
        print('accept score: ' + str(sum(scores) / len(scores)))
        with open(args.file_dir + '/' + args.id + '/s2s_output/accept_changed.txt', 'w') as output_file:
            for score in scores:
                output_file.write(str(score) + '\n')
    elif args.parsed_file is not None:
        src_sents, tgt_sents = [], []
        is_src = True
        for line in open(args.parsed_file, 'r'):
            if line.strip() != '':
                comps = line.strip().split('\t')
                if is_src:
                    src_sents.append(comps[1])
                else:
                    tgt_sents.append(comps[1])
                is_src = (not is_src)
        scores = evaluate_accept(src_sents, tgt_sents, args)
        print('accept score: ' + str(sum(scores) / len(scores)))

