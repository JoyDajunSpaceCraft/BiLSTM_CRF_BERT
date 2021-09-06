from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys
import datetime
import time
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import copy
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from utils import NerProcessor, convert_examples_to_features, get_Dataset
from models import BERT_BiLSTM_CRF
import conlleval

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule

logger = logging.getLogger(__name__)

valid_curve = []
checkpoint_interval = 5
filename = "store iteration"
# train_loss_data = []
# train_loss_step = []
# eval_loss_data = []
# eval_loss_step =[]

# set the random seed for repeat
def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
  return tensor.detach().cpu().tolist()


def boolean_string(s):
  if s not in {'False', 'True'}:
    raise ValueError('Not a valid boolean string')
  return s == 'True'


def evaluate(args, data, model, id2label, all_ori_tokens, writer):
  # model = torch.load("/content/gdrive/MyDrive/BiLSTM_CRF_BERT/model/clue_bilstm/training_args.bin")
  model.eval()
  sampler = SequentialSampler(data)
  dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)

  logger.info("***** Running eval *****")
  logger.info(f" Num examples = {len(data)}")
  logger.info(f" Batch size = {args.eval_batch_size}")
  pred_labels = []
  ori_labels = []

  steps = 0
  ev_loss = 0.0
  logging_ev_loss = 0.0
  eval_list = []
  steps_list = []
  for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(dataloader, desc="Evaluating")):
    input_ids = input_ids.to(args.device)
    input_mask = input_mask.to(args.device)
    segment_ids = segment_ids.to(args.device)
    label_ids = label_ids.to(args.device)
    # loss = model(input_ids, label_ids, segment_ids, input_mask)

    # not add grad into eval
    with torch.no_grad():
      loss = model(input_ids, label_ids, segment_ids, input_mask)
      logits = model.predict(input_ids, segment_ids, input_mask)
    # logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
    # logits = logits.detach().cpu().numpy()

    for l in logits:
      pred_labels.append([id2label[idx] for idx in l])

    for l in label_ids:
      ori_labels.append([id2label[idx.item()] for idx in l])

    # logits is a list
    ev_loss += loss

    if (b_i + 1) % 1 == 0:
      steps += 1
      if args.logging_steps > 0 and steps % 7 == 0:
        ev_loss_avg = (ev_loss - logging_ev_loss) / args.logging_steps
        eval_list.append(float(ev_loss_avg))
        steps_list.append(steps)
        print("write into eval")
        writer.add_scalar("Eval/loss", ev_loss_avg, global_step=steps)
        # eval_loss_data.append(ev_loss_avg)
        # eval_loss_step.append(steps)
        logging_ev_loss = ev_loss
    print("Eval done")

  # plt.plot(steps_list, eval_list, label='eval')
  # plt.legend(loc='upper right')
  # plt.ylabel('loss value')
  # plt.xlabel('Iteration')
  # plt.savefig(fname="eval.png")
  # plt.show()

  eval_list = []
  for ori_tokens, oril, prel in zip(all_ori_tokens, ori_labels, pred_labels):
    for ot, ol, pl in zip(ori_tokens, oril, prel):
      if ot in ["[CLS]", "[SEP]"]:
        continue
      eval_list.append(f"{ot} {ol} {pl}\n")
    eval_list.append("\n")

  # eval the model
  counts = conlleval.evaluate(eval_list)
  conlleval.report(counts)

  # namedtuple('Metrics', 'tp fp fn prec rec fscore')
  overall, by_type = conlleval.metrics(counts)

  return overall, by_type



def main():
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--train_file", default=None, type=str)
  parser.add_argument("--eval_file", default=None, type=str)
  parser.add_argument("--test_file", default=None, type=str)
  parser.add_argument("--model_name_or_path", default=None, type=str)
  parser.add_argument("--output_dir", default=None, type=str)

  ## other parameters
  parser.add_argument("--config_name", default="", type=str,
                      help="Pretrained config name or path if not the same as model_name")
  parser.add_argument("--tokenizer_name", default="", type=str,
                      help="Pretrained tokenizer name or path if not the same as model_name")
  parser.add_argument("--cache_dir", default="", type=str,
                      help="Where do you want to store the pre-trained models downloaded from s3")

  parser.add_argument("--max_seq_length", default=256, type=int)
  parser.add_argument("--do_train", default=False, type=boolean_string)
  parser.add_argument("--do_eval", default=False, type=boolean_string)
  parser.add_argument("--do_test", default=False, type=boolean_string)
  parser.add_argument("--train_batch_size", default=8, type=int)
  parser.add_argument("--eval_batch_size", default=8, type=int)
  parser.add_argument("--learning_rate", default=3e-5, type=float)
  parser.add_argument("--num_train_epochs", default=10, type=float)
  parser.add_argument("--warmup_proprotion", default=0.1, type=float)
  parser.add_argument("--use_weight", default=1, type=int)
  parser.add_argument("--local_rank", type=int, default=-1)
  parser.add_argument("--seed", type=int, default=2019)
  parser.add_argument("--fp16", default=False)
  parser.add_argument("--loss_scale", type=float, default=0)
  parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
  parser.add_argument("--warmup_steps", default=0, type=int)
  parser.add_argument("--adam_epsilon", default=1e-8, type=float)
  parser.add_argument("--max_steps", default=-1, type=int)
  parser.add_argument("--do_lower_case", action='store_true')
  parser.add_argument("--logging_steps", default=500, type=int)
  parser.add_argument("--clean", default=False, type=boolean_string, help="clean the output dir")

  parser.add_argument("--need_birnn", default=False, type=boolean_string)
  parser.add_argument("--rnn_dim", default=128, type=int)

  args = parser.parse_args()

  device = torch.device("cuda")
  # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_
  args.device = device
  n_gpu = torch.cuda.device_count()

  logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                      datefmt='%m/%d/%Y %H:%M:%S',
                      level=logging.INFO)

  logger.info(f"device: {device} n_gpu: {n_gpu}")

  if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
      args.gradient_accumulation_Iteration))

  # now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H')
  # tmp_dir = args.output_dir + '/' +str(now_time) + '_ernie'
  # if not os.path.exists(tmp_dir):
  #     os.makedirs(tmp_dir)
  # args.output_dir = tmp_dir
  if args.clean and args.do_train:
    # logger.info("清理")
    if os.path.exists(args.output_dir):
      def del_file(path):
        ls = os.listdir(path)
        for i in ls:
          c_path = os.path.join(path, i)
          print(c_path)
          if os.path.isdir(c_path):
            del_file(c_path)
            os.rmdir(c_path)
          else:
            os.remove(c_path)

      try:
        del_file(args.output_dir)
      except Exception as e:
        print(e)
        print('pleace remove the files of output dir and data.conf')
        exit(-1)

  if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  if not os.path.exists(os.path.join(args.output_dir, "eval")):
    os.makedirs(os.path.join(args.output_dir, "eval"))

  writer = SummaryWriter(logdir=os.path.join(args.output_dir, "eval"), comment="Linear")

  processor = NerProcessor()
  label_list = processor.get_labels(args)
  num_labels = len(label_list)
  args.label_list = label_list

  if os.path.exists(os.path.join(args.output_dir, "label2id.pkl")):
    with open(os.path.join(args.output_dir, "label2id.pkl"), "rb") as f:
      label2id = pickle.load(f)
  else:
    label2id = {l: i for i, l in enumerate(label_list)}
    with open(os.path.join(args.output_dir, "label2id.pkl"), "wb") as f:
      pickle.dump(label2id, f)

  id2label = {value: key for key, value in label2id.items()}

  # Prepare optimizer and schedule (linear warmup and decay)

  if args.do_train:

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                              do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        num_labels=num_labels)
    model = BERT_BiLSTM_CRF.from_pretrained(args.model_name_or_path, config=config,
                                            need_birnn=args.need_birnn, rnn_dim=args.rnn_dim)

    model.to(device)

    if n_gpu > 1:
      model = torch.nn.DataParallel(model)

    train_examples, train_features, train_data = get_Dataset(args, processor, tokenizer, mode="train")
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = None
    if args.do_eval:
      eval_examples, eval_features, eval_data = get_Dataset(args, processor, tokenizer, mode="eval")

    if args.max_steps > 0:
      t_total = args.max_steps
      args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
      t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
      {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total optimization steps = %d", t_total)

    model.train()
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_f1 = 0.0

    train_plot_data = []
    test_plot_data = []
    train_plot_step = []
    test_plot_step = []

    # check dimetion
    print("logging_steps is", args.logging_steps)
    print("gradient_accumulation_steps", args.gradient_accumulation_steps)

    # eval_sampler = SequentialSampler(eval_data)
    # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)
    # # draw eval data
    # ev_loss = 0.0
    # eval_step = 0
    # logging_ev_loss = 0.0
    # eval_list = []
    # steps_list = []
    # for ep in trange(int(args.num_train_epochs), desc="Epoch"):
    #   for eval_steps, (input_ids, input_mask, segment_ids, label_ids) in enumerate(
    #     tqdm(eval_dataloader, desc="Evaluating")):
    #     input_ids = input_ids.to(args.device)
    #     input_mask = input_mask.to(args.device)
    #     segment_ids = segment_ids.to(args.device)
    #     label_ids = label_ids.to(args.device)
    #     with torch.no_grad():
    #       loss = model(input_ids, label_ids, segment_ids, input_mask)
    #     ev_loss += loss
    #
    #     if (eval_steps + 1) % 1 == 0:
    #       eval_step += 1
    #       if args.logging_steps > 0 and eval_step % 15 == 0:
    #         ev_loss_avg = (ev_loss - logging_ev_loss) / args.logging_steps
    #         eval_list.append(float(ev_loss_avg))
    #         steps_list.append(eval_step)
    #         print("write into eval")
    #         writer.add_scalar("Eval/loss", ev_loss_avg, eval_step)
    #         logging_ev_loss = ev_loss
    # print("Eval done")

    for ep in trange(int(args.num_train_epochs), desc="Epoch"):
      model.train()
      # make sure do_eval and do_train both open

      for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        # in this place the batch contain the input id, input mask, segment id and label id
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        outputs = model(input_ids, label_ids, segment_ids, input_mask)
        loss = outputs


        if n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
          loss = loss / args.gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
          optimizer.step()
          scheduler.step()  # Update learning rate schedule
          model.zero_grad()
          global_step += 1

          if args.logging_steps > 0 and global_step % args.logging_steps == 0:
            tr_loss_avg = (tr_loss - logging_loss) / args.logging_steps
            train_plot_data.append(float(tr_loss_avg))
            train_plot_step.append(global_step)
            writer.add_scalar("Train/loss", tr_loss_avg, global_step)

            # train_loss_data.append(tr_loss_avg)
            # train_loss_step.append(global_step)
            logging_loss = tr_loss

            print("training loss", loss)

          MAX_EPOCH = int(args.num_train_epochs)

          print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
            ep, MAX_EPOCH, step + 1, len(train_dataloader), tr_loss))

      if args.do_eval:
        all_ori_tokens_eval = [f.ori_tokens for f in eval_features]
        overall, by_type = evaluate(args, eval_data, model, id2label, all_ori_tokens_eval, writer)

        # add eval result to tensorboard
        f1_score = overall.fscore
        writer.add_scalar("Eval/precision", overall.prec, ep)
        writer.add_scalar("Eval/recall", overall.rec, ep)
        writer.add_scalar("Eval/f1_score", overall.fscore, ep)

        # save the best performs model
        if f1_score > best_f1:
          logger.info(f"----------the best f1 is {f1_score}---------")
          best_f1 = f1_score
          model_to_save = model.module if hasattr(model,
                                                  'module') else model  # Take care of distributed/parallel training
          model_to_save.save_pretrained(args.output_dir)
          tokenizer.save_pretrained(args.output_dir)

          # Good practice: save your training arguments together with the trained model
          torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

      logger.info(f'epoch {ep}, train loss: {tr_loss}')
    # writer.add_graph(model)
    # logger.info("train_loss_step",train_loss_step)
    # logger.info("train_loss_data", train_loss_data)
    # logger.info("eval_loss_step",eval_loss_step)
    # logger.info("eval_loss_data",eval_loss_data)
    writer.close()

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

  if args.do_test:
    # model = BertForTokenClassification.from_pretrained(args.output_dir)
    # model.to(device)
    label_map = {i: label for i, label in enumerate(label_list)}

    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    args = torch.load(os.path.join(args.output_dir, 'training_args.bin'))
    model = BERT_BiLSTM_CRF.from_pretrained(args.output_dir, need_birnn=args.need_birnn, rnn_dim=args.rnn_dim)
    model.to(device)

    test_examples, test_features, test_data = get_Dataset(args, processor, tokenizer, mode="test")

    logger.info("*****  *****")
    logger.info(f" Num examples = {len(test_examples)}")
    logger.info(f" Batch size = {args.eval_batch_size}")

    all_ori_tokens = [f.ori_tokens for f in test_features]
    all_ori_labels = [e.label.split(" ") for e in test_examples]
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    model.eval()

    pred_labels = []

    test_loss = 0.0
    logging_te_loss = 0.0
    steps = 0
    step_list = []
    test_list = []

    for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(test_dataloader, desc="Predicting")):

      input_ids = input_ids.to(device)
      input_mask = input_mask.to(device)
      segment_ids = segment_ids.to(device)
      label_ids = label_ids.to(device)

      with torch.no_grad():
        loss = model(input_ids, label_ids, segment_ids, input_mask)
        logits = model.predict(input_ids, segment_ids, input_mask)
      # logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
      # logits = logits.detach().cpu().numpy()

      test_feature_copy = copy.deepcopy(test_features)

      result_with_wordpiece = []
      for i in test_feature_copy:
        print("test data is", i.tokens)
        print(type(i.tokens))  # type is list
        result_with_wordpiece.append(i.tokens)

      for l in zip(logits,result_with_wordpiece):
        pred_label = []
        for index, label_idx in enumerate(l[0]):
          if "#" in l[1][index]:
            continue
          else:
            pred_label.append(id2label[label_idx])
        pred_labels.append(pred_label)

      test_loss += loss
      if (b_i + 1) % args.gradient_accumulation_steps == 0:
        steps += 1
        if args.logging_steps > 0 and steps % args.logging_steps == 0:
          te_loss_avg = (test_loss - logging_te_loss) / args.logging_steps
          test_list.append(float(te_loss_avg))
          step_list.append(steps)
          logging_te_loss = test_loss
          print("write into eval")
          writer.add_scalar("Dev/loss", te_loss_avg, global_step=steps)
      print("Test done")

    assert len(pred_labels) == len(all_ori_tokens) == len(all_ori_labels)

    temp_all_ori_token = []
    for ori_token in all_ori_tokens:
      temp_ori_token = []
      for i in ori_token:
        if i in ["[CLS]", "[SEP]"]:
          continue
        else:
          temp_ori_token.append(i)

      temp_all_ori_token.append(temp_ori_token)

    print("ready to print test result")
    print("pred_label is", len(pred_label))
    print("all ori token is", temp_all_ori_token)
    print("all ori labels is", all_ori_labels)

    with open(os.path.join(args.output_dir, "token_labels_.txt"), "w", encoding="utf-8") as f:
      for ori_tokens, ori_labels, prel in zip(temp_all_ori_token, all_ori_labels, pred_labels):
        for ot, ol, pl in zip(ori_tokens, ori_labels, prel):
          if ot in ["[CLS]", "[SEP]"]:
            continue
          else:
            f.write(f"{ot} {ol} {pl}\n")
        f.write("\n")



if __name__ == "__main__":
  main()
  pass
# BERT_BASE_DIR=bert-base-uncased
# DATA_DIR=/content/gdrive/MyDrive/BiLSTM_CRF_BERT/data/6_27_test
# OUTPUT_DIR=./model/clue_bilstm
# export CUDA_VISIBLE_DEVICES=0

# python ner.py \
#     --model_name_or_path ${BERT_BASE_DIR} \
#     --do_train False \
#     --do_eval False \
#     --do_test True \
#     --max_seq_length 256 \
#     --train_file ${DATA_DIR}/train.txt \
#     --eval_file ${DATA_DIR}/dev.txt \
#     --test_file ${DATA_DIR}/test.txt \
#     --train_batch_size 32 \
#     --eval_batch_size 32 \
#     --num_train_epochs 1 \
#     --do_lower_case \
#     --logging_steps 15 \
#     --need_birnn True \
#     --rnn_dim 256 \
#     --clean True \
#     --output_dir $OUTPUT_DIR