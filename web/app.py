from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/co2_emission')
def co2_emission():
    return render_template('co2_emission.html')

@app.route('/planting')
def planting():
    return render_template('planting.html')

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=8080)
