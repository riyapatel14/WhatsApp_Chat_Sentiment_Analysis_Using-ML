from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime

nltk.download('vader_lexicon')

app = Flask(__name__)

def date_time(s):
    pattern = r'^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+:[0-9]+) (AM|PM|am|pm) -'

    result = re.match(pattern, s)
    if result:
        return True
    return False

def find_author(s):
    s = s.split(':')
    if len(s) == 2:
        return True
    return False

def message(line):
    split_line = line.split(' - ')
    datetime_str = split_line[0]
    date, time = datetime_str.split(', ')
    message = " ".join(split_line[1:])

    if find_author(message):
        split_message = message.split(': ')
        author = split_message[0]
        message = " ".join(split_message[1:])
    else:
        author = None
    return date, time, author, message

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            data = []
            messageBuffer = []
            date, time, author = None, None, None

            file_content = file.read().decode("utf-8")
            lines = file_content.split('\n')
            for line in lines:
                line = line.strip()
                if date_time(line):
                    if len(messageBuffer) > 0:
                        data.append([date, time, author, ' '.join(messageBuffer)])
                    messageBuffer.clear()
                    date, time, author, message = message(line)
                    messageBuffer.append(message)
                else:
                    messageBuffer.append(line)

            df = pd.DataFrame(data, columns=["Date", "Time", "Author", "Message"])
            df["Date"] = pd.to_datetime(df["Date"])
            data = df.dropna()

            sentiments = SentimentIntensityAnalyzer()
            data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Message"]]
            data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Message"]]
            data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Message"]]

            x = sum(data["Positive"])
            y = sum(data["Negative"])
            z = sum(data["Neutral"])

            if x > y and x > z:
                sentiment_result = "Positive"
            elif y > x and y > z:
                sentiment_result = "Negative"
            else:
                sentiment_result = "Neutral"

            return render_template("index.html", sentiment=sentiment_result)
    return render_template("index.html", sentiment=None)

if __name__ == "__main__":
    app.run(debug=True)