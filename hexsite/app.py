from flask import Flask, request, render_template
from backend import test_network
import urllib.request
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['image_url']
    print(user_input)
    url_response = urllib.request.urlopen(user_input)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    outputtext = test_network.network_test_no_args("hex_maniac.keras", img)
    return f"Result: {outputtext}"

if __name__ == '__main__':
    app.run(debug=True)
