"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""
import os
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging


global recognizer
global task


class Handler(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def _set_json_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                str(self.path), str(self.headers), post_data.decode('utf-8'))

        raw_text = json.loads(post_data)
        if task == "ner" and len(raw_text["data"]) > 128:
            self._set_json_response()
            result_dict = {"status": 501, "msg": "max length exceed"}
            self.wfile.write(json.dumps(result_dict).encode("utf-8"))
        else:
            result = recognizer.inference(raw_text["data"])
            self._set_json_response()
            result_dict = {"status": 200, "msg": "success", "result": result}
            self.wfile.write(json.dumps(result_dict).encode("utf-8"))


def run(server_class=HTTPServer, handler_class=Handler, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s] %(message)s")

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--port', default=8080, type=int, help='http service port number')
    parser.add_argument('--task', default='ner', type=str, help='model task type')
    parser.add_argument('--encode_document', action='store_true', help="Whether treat the text as document or not")
    parser.add_argument('--model_dir', default='/your/model/dir', type=str, help='model dir path')
    parser.add_argument('--gpu', default='0', type=str, help='0')
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    task = opt.task
    if opt.task == "ner":
        from tools.inference import NERInferenceService as InferenceService
    elif opt.task == "multilabeling":
        from tools.inference import MultiLabelingInferenceService as InferenceService
    recognizer = InferenceService(opt.model_dir, encode_document=opt.encode_document)
    run(port=opt.port)
