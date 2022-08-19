# %%
import os
os.chdir("/Users/yul/Desktop/기업프로젝트/wisenut_demo")
from flask import Flask, render_template, request
import pandas as pd
import requests
import json
from tools.Tools import *

# %%
dataset_name = "dataset.pickle"
algorism_type=bm25

# %%
doc = pd.read_pickle(dataset_name)

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def main_get(num=None):
    return render_template('submit_test.html', num=num)

@app.route('/calculate', methods=['POST', 'GET'])
def calculate(num=None):
    if request.method == 'GET':
        pass
    elif request.method == 'POST':

        value = request.form['char1']
        
        value2 = query_tokenizer(value)
        
        docu = algorism_type(doc, value2)

        request_text = []
        question = [{"question": value}]
        request_text.append({'context': docu, 'questionInfoList': question})

        URL = 'http://cb84-34-83-170-159.ngrok.io/mrc/'

        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json'
        }

        response = requests.post(URL + '/predict/documents', data=json.dumps(request_text), headers=headers)
        
        response_text = json.loads(response.text).get('0')
        
        return render_template('submit_test.html', char1=response_text)#print_data)


if __name__ == '__main__':
  app.run(threaded=True)


