from flask import Flask, render_template, request
import pickle, csv

app = Flask(__name__)

pokemon_chosen = ""

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    pokemon_chosen = request.form['poke']
    prediction = show_me_your_pickle(pokemon_chosen)
    return render_template('predict.html',prediction = prediction, name = pokemon_chosen)


def show_me_your_pickle(pokemon_picked):
    infile = open("NaiveBayes.p","rb")
    algo = pickle.load(infile)
    infile.close()
    pokemon = read_table(pokemon_picked)
    for i in range(len(pokemon)):
        pokemon[i] = int(pokemon[i])
    prediction = algo.predict([pokemon])
    return prediction

def read_table(pokemon_picked):
    pokemon = []
    file = open('../pokemon - pokemon.csv')
    type(file)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    name = header.index('name')
    line = 0
    for row in csvreader:
        if(row[name]==pokemon_picked):
            pokemon = table_helper(line)
            break
        line += 1
    file.close()
    return pokemon

def table_helper(line_number):
    pokemon = []
    rows = []
    file = open("../discretized_values.csv")
    csvreader = csv.reader(file)
    header = next(csvreader)
    i = 0
    for row in csvreader:
        if(i > line_number):
            break
        rows.append(row)
        i += 1
    pokemon = rows[line_number]
    file.close()
    return pokemon

if __name__ == "__main__":
    app.run(debug=True)
    