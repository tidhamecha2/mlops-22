from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


# from flask import request
# @app.route("/sum", methods=['POST'])
# def sum():
#     input_json = request.json
#     x = input_json['x']
#     y = input_json['y']
#     return x + y 
