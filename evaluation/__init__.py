import os


def eval_wrapper(cfg, pred_list, label_list):
    if cfg["train"]["task_name"] == "ner":
        from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
        for logits, label_ids in zip(pred_list, label_list):
            # FIXME ner bilstm
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
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
        report = classification_report(y_true, y_pred, digits=4)
        logger.info("\n%s", report)
        eval_precision = precision_score(y_true, y_pred)
        eval_recall = recall_score(y_true, y_pred)
        eval_f1 = f1_score(y_true, y_pred)
        results = {
            "eval_precision": eval_precision,
            "eval_recall": eval_recall,
            "eval_f1": eval_f1
        }


    elif cfg["train"]["task_name"] == "textclf":
        all_logits = torch.cat(pred_list, 0)
        all_labels = torch.cat(label_list, 0)
        _, preds = torch.max(all_logits.data, 1)
        acc = np.mean((preds.byte() == all_labels.byte()).float().numpy())
        results = {
            "eval_precision": acc
        }

    elif cfg["train"]["task_name"] == "tag":
        # FIXME change to mulit classification
        all_logits = torch.cat(pred_list, 0)
        all_labels = torch.cat(label_list, 0)
        _, preds = torch.max(all_logits.data, 1)
        acc = np.mean((preds.byte() == all_labels.byte()).float().numpy())
        results = {
            "eval_precision": acc
        }

    elif cfg["train"]["task_name"] == "summary" or cfg["train"]["task_name"] == "translation":
        if cfg["eval"]["metric"] == "bleu":
            from .summarization_eval import evaluate_bleu
            avg_score, scores = evaluate_bleu(pred_list, label_list)
            results = {
                "bleu_avg_score": avg_score,
                # FIXME
                # "bleu_scores": scores
            }
        elif cfg["eval"]["metric"] == "rouge":
            # FIXME
            # ...
            pass

    return results
