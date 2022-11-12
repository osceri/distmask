from datetime import datetime

class Logger:
    def __init__(self):
        self.log = ""

    def __call__(self, text):
        text = f'[{datetime.now()}] {text}'
        self.log += f'{text}\n'
        print(text)

