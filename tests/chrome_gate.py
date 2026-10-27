"""Small Chrome DevTools adapter for the local UI gate (no browser download)."""
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

from websockets.sync.client import connect


class ChromePage:
    def __enter__(self):
        self.profile = tempfile.TemporaryDirectory(prefix="gate4-chrome-")
        self.process = subprocess.Popen([
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            '--headless=new', '--no-first-run', '--no-default-browser-check',
            '--remote-debugging-port=0', '--user-data-dir=' + self.profile.name,
            'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port_file = Path(self.profile.name) / 'DevToolsActivePort'
        deadline = time.monotonic() + 30
        while not port_file.exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Chrome did not start')
            time.sleep(0.1)
        port = port_file.read_text().splitlines()[0]
        with urllib.request.urlopen('http://127.0.0.1:' + port + '/json') as response:
            targets = json.load(response)
        target = next(item for item in targets if item['type'] == 'page')
        self.socket = connect(target['webSocketDebuggerUrl'], max_size=20 * 1024 * 1024)
        self.sequence = 0
        self.errors = []
        self.call('Runtime.enable')
        self.call('Page.enable')
        self.call('Emulation.setDeviceMetricsOverride', dict(width=1440, height=1100, deviceScaleFactor=1, mobile=False))
        self.call('Emulation.setEmulatedMedia', dict(features=[dict(name='prefers-reduced-motion', value='reduce')]))
        return self

    def call(self, method, params=None):
        self.sequence += 1
        request_id = self.sequence
        self.socket.send(json.dumps(dict(id=request_id, method=method, params=params or {})))
        while True:
            message = json.loads(self.socket.recv(timeout=120))
            if message.get('method') == 'Runtime.exceptionThrown':
                self.errors.append(message['params'])
            if message.get('id') == request_id:
                if 'error' in message:
                    raise RuntimeError(message['error'])
                return message.get('result', {})

    def evaluate(self, expression, argument=None):
        if expression.lstrip().startswith('async '):
            expression = '(' + expression + ')(' + json.dumps(argument) + ')'
        result = self.call('Runtime.evaluate', dict(expression=expression, awaitPromise=True, returnByValue=True))
        if 'exceptionDetails' in result:
            raise RuntimeError(result['exceptionDetails'])
        return result.get('result', {}).get('value')

    def wait(self, expression):
        deadline = time.monotonic() + 120
        while not self.evaluate(expression):
            if time.monotonic() > deadline:
                raise RuntimeError('Timed out: ' + expression)
            time.sleep(0.2)

    def screenshot(self, path):
        result = self.call('Page.captureScreenshot', dict(format='png', captureBeyondViewport=True))
        Path(path).write_bytes(base64.b64decode(result['data']))

    def __exit__(self, exc_type, exc_value, traceback):
        self.socket.close()
        self.process.terminate()
        self.process.wait(timeout=15)
        self.profile.cleanup()
