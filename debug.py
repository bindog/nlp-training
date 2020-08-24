import os
import argparse
import json
import logging


def textclf_inference_compare(json_file_path, output_file_path, recognizer):
    output_file = open(output_file_path, "wb")
    label_map = recognizer.get_label_map()
    with open(json_file_path, "r") as f:
        for line in f:
            news_json = json.loads(line.strip())
            pred = recognizer.inference(news_json["text"], parse_label=False)
            label = news_json["category"]
            if pred != label:
                output_json = {}
                output_json["id"] = news_json["id"]
                output_json["text"] = news_json["text"]
                output_json["category"] = int(label)
                output_json["pred"] = int(pred)
                output_json["category_text"] = label_map[label]
                output_json["pred_text"] = label_map[pred]
                output_file.write(json.dumps(output_json, ensure_ascii=False).encode("utf-8") + "\n".encode("utf-8"))
    output_file.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s] %(message)s")

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--task', default='ner', type=str, help='model task type')
    parser.add_argument('--encode_document', action='store_true', help="Whether treat the text as document or not")
    parser.add_argument('--doc_inner_batch_size', default=10, type=int, help="the bert batch size inside a document")
    parser.add_argument('--tag', action='store_true', help="normal text classification or tag classification")
    parser.add_argument('--model_dir', default='/your/model/dir', type=str, help='model dir path')
    parser.add_argument('--json_file_path', default='/your/json/file/path', type=str, help='json file path to inference and compare')
    parser.add_argument('--output_file_path', default='/your/output/json/file/path', type=str, help='output file path')
    parser.add_argument('--gpu', default='0', type=str, help='0')
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    task = opt.task
    if opt.task == "ner":
        from tools.inference import NERInferenceService as InferenceService
    elif opt.task == "textclf" or opt.task == "tag":
        from tools.inference import TextclfInfercenceService as InferenceService
    recognizer = InferenceService(
                            opt.model_dir,
                            encode_document=opt.encode_document,
                            doc_inner_batch_size=opt.doc_inner_batch_size,
                            tag=opt.tag
                        )

    if opt.task == "textclf":
        textclf_inference_compare(opt.json_file_path, opt.output_file_path, recognizer)
