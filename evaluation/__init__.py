import os
import torch
import numpy as np


def eval_wrapper(cfg, pred_list, label_list, label_map):
    if cfg["train"]["task_name"] == "ner":
        from seqeval.metrics import precision_score, recall_score, f1_score
        import torch.nn.functional as F
        y_true = []
        y_pred = []
        for logits, label_ids in zip(pred_list, label_list):
            # FIXME ner bilstm
            if not cfg["train"]["ner_addBilstm"]:
                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.numpy()
            label_ids = label_ids.numpy()
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_map[label_ids[i][j]] == "[SEP]":
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])
        eval_precision = precision_score(y_true, y_pred)
        eval_recall = recall_score(y_true, y_pred)
        eval_f1 = f1_score(y_true, y_pred)
        results = {
            "eval_precision": eval_precision,
            "eval_recall": eval_recall,
            "eval_f1": eval_f1
        }


    elif cfg["train"]["task_name"] == "textclf":
        from sklearn.metrics import f1_score, precision_score, recall_score
        all_logits = torch.cat(pred_list, 0)
        all_labels = torch.cat(label_list, 0)
        _, preds = torch.max(all_logits.data, 1)
        preds, all_labels = preds.int().numpy(), all_labels.int().numpy()
        avg = cfg["eval"]["metric"] if cfg["eval"]["metric"] == 'micro' else 'macro'
        eval_precision = precision_score(all_labels, preds, average=avg)
        eval_recall = recall_score(all_labels, preds, average=avg)
        eval_f1 = f1_score(all_labels, preds, average=avg)

        results = {
            "eval_precision": eval_precision,
            "eval_recall": eval_recall,
            "eval_f1": eval_f1
        }

    elif cfg["train"]["task_name"] == "tag":
        from sklearn.metrics import f1_score, precision_score, recall_score
        all_logits, all_labels = torch.cat(pred_list, 0), torch.cat(label_list, 0)
        zero, one = torch.zeros_like(all_logits), torch.ones_like(all_logits)
        all_logits = torch.where(all_logits>0.5, one, zero).int().numpy()
        all_labels = all_labels.int().numpy()

        avg = cfg["eval"]["metric"] if cfg["eval"]["metric"] == 'micro' else 'macro'
        eval_precision = precision_score(all_labels, all_logits, average=avg)
        eval_recall = recall_score(all_labels, all_logits, average=avg)
        eval_f1 = f1_score(all_labels, all_logits, average=avg)

        results = {
            "eval_precision": eval_precision,
            "eval_recall": eval_recall,
            "eval_f1": eval_f1
        }

    elif cfg["train"]["task_name"] == "summary" or cfg["train"]["task_name"] == "translation":
        if cfg["eval"]["metric"] == "bleu":
            results = score_bleu(pred_list, label_list)
        elif cfg["eval"]["metric"] == "rouge":
            results = score_rouge(pred_list, label_list)
        elif cfg["eval"]["metric"] == "both":
            results = score_bleu(pred_list, label_list)
            results.update(score_rouge(pred_list, label_list))

    elif cfg["train"]["task_name"] == "pet":
        cloze_length = cfg["pet"]["pet_pattern"]["cloze_length"]
        total = 0
        right = 0
        for pred, label in zip(pred_list, label_list):
            total += 1
            if pred[-cloze_length:] == label[-cloze_length:]:
                right += 1
        results = {
            "eval_precision": right / total
        }

    return results

def score_bleu(pred_list, label_list):
    from .summarization_eval import evaluate_bleu
    avg_score, scores = evaluate_bleu(pred_list, label_list)
    results = {
        "bleu_avg_score": avg_score,
        # FIXME
        # "bleu_all_score": scores
    }
    return results

def score_rouge(pred_list, label_list):
    from .summarization_eval import evaluate_rouge
    rouge_1 = evaluate_rouge(pred_list, label_list, n=1, lang="zh")
    rouge_2 = evaluate_rouge(pred_list, label_list, n=2, lang="zh")
    rouge_l = evaluate_rouge(pred_list, label_list, n='l', lang="zh")
    results = {
        "rouge-1": rouge_1[0],
        "rouge-2": rouge_2[0],
        "rouge-L": rouge_l[0]
    }
    return results

def score_bleu(pred_list, label_list):
    from .summarization_eval import evaluate_bleu
    avg_score, scores = evaluate_bleu(pred_list, label_list)
    results = {
        "bleu_avg_score": avg_score,
        # FIXME
        # "bleu_all_score": scores
    }
    return results

def score_rouge(pred_list, label_list):
    from .summarization_eval import evaluate_rouge
    rouge_1 = evaluate_rouge(pred_list, label_list, n=1, lang="zh")
    rouge_2 = evaluate_rouge(pred_list, label_list, n=2, lang="zh")
    rouge_l = evaluate_rouge(pred_list, label_list, n='l', lang="zh")
    results = {
        "rouge-1": rouge_1[0],
        "rouge-2": rouge_2[0],
        "rouge-L": rouge_l[0]
    }
    return results