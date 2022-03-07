from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world(): 
    return 'Hello World!'


@app.route('/<location_id>')
def hello_world(location_id):
    return 'Hello World! location: '+location_id


if __name__ == '__main__':
    app.run()