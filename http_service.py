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

mapping_dict = {
    "PER": "人名",
    "LOC": "地名",
    "ORG": "组织",
    "JOB": "组织",
    "PRO": "产品",
    "TIME": "时间",
    "COM": "公司"
}

global recognizer


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
        if len(raw_text["data"]) > 128:
            self._set_json_response()
            result_dict = {"status": 501, "msg": "max length exceed"}
            self.wfile.write(json.dumps(result_dict).encode("utf-8"))
        else:
            result = recognizer.inference(raw_text["data"])
            self._set_json_response()
            result_dict = {"status": 200, "msg": "success", "result": []}
            for r in result:
                rj = {}
                rj["entity"] = r[2]
                rj["type"] = mapping_dict[r[3]]
                rj["offset"] = [r[0], r[1]]
                result_dict["result"].append(rj)
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
    parser.add_argument('--model_dir', default='/your/model/dir', type=str, help='model dir path')
    parser.add_argument('--gpu', default='0', type=str, help='0')
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    from tools.inference import NERInferenceService
    recognizer = NERInferenceService(opt.model_dir)
    run(port=opt.port)
