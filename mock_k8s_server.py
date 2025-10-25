#!/usr/bin/env python3
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

class MockK8sHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/v1/nodes':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'apiVersion': 'v1',
                'kind': 'NodeList',
                'items': [
                    {
                        'metadata': {'name': 'mock-node-1'},
                        'status': {'phase': 'Ready'}
                    }
                ]
            }
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/api/v1/pods':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'apiVersion': 'v1',
                'kind': 'PodList',
                'items': [
                    {
                        'metadata': {'name': 'mock-pod-1', 'namespace': 'default'},
                        'status': {'phase': 'Running'}
                    }
                ]
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', 6443), MockK8sHandler)
    print('Mock Kubernetes API server running on port 6443')
    server.serve_forever()
