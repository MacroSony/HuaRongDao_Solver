from flask import Flask
import solver

app = Flask(__name__)

@app.route('/')
def index():
    result = solver.solve('.11.\n.11.\n....\n....\n....')
    return result.board.display()

if __name__ == "__main__":
    app.run(debug=True)