from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator


def evaluate_bleu(summary, references, lang="zh"):
    bleu_calc = BLEUCalculator(lang=lang)
    assert len(summary) == len(references), "number of summary and references should be equal"

    scores = []
    for s, rs in zip(summary, references):
        score = bleu_calc.bleu(s, rs)
        scores.append(score)
    score_avg = sum(scores) /  len(scores)
    return score_avg, scores


def evaluate_rouge(summary, references, n=1, lang="zh"):
    rouge_calc = RougeCalculator(stopwords=True, lang=lang)
    assert len(summary) == len(references), "number of summary and references should be equal"

    rouges = []
    for s, rs in zip(summary, references):
        if n == 'l':
            rouge_n = rouge_calc.rouge_l(s, rs)
        else:
            rouge_n = rouge_calc.rouge_n(s, rs, n)
        rouges.append(rouge_n)
    rouge_avg = sum(rouges) /  len(rouges)
    return rouge_avg, rouges
