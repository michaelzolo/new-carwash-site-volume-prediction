import json
import time

from flask import Flask, request
# TODO REMOVE from flask_ngrok import run_with_ngrok

app = Flask(__name__)
# TODO REMOVE run_with_ngrok(app)


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/api/', methods=['GET'])  # /api/?zipcode=12345&la=123.456&lo=654.987&lot_size=1212
def hello_params():
    zipcode = str(request.args.getlist('zipcode'))
    la = str(request.args.getlist('la'))
    lo = str(request.args.getlist('lo'))
    lot_size = str(request.args.getlist('lot_size'))

    data_set = {'Page': 'Prediction', 'Message': f'Got inputs: {(zipcode,la,lo,lot_size)}, ', 'Timestamp': time.time()}
    json_sump = json.dumps(data_set)
    return json_sump


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=False)
