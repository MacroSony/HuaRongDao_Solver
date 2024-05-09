from flask import Flask, jsonify, request
import solver

app = Flask(__name__)


@app.route('/solver', methods=['POST'])
def solve():
    board_string = request.get_json()
    if "board" not in board_string.keys():
        return "no board received", 404
    result = solver.get_steps(board_string["board"])
    data = {
        "solution": result
    }
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)