import gradio as gr
from joblib import load
import numpy as np

model = load('models\decisiontree.pkl')
x_scaler = load('models\Xscaler.pkl')
y_scaler = load('models\yscaler.pkl')
label = load('models\label_dict.pkl')

def prediction(acousticness, danceability, energy, instrumentalness, liveness, loudness,
               speechiness, tempo, valence, popularity, duration_ms, encoded_album, age):
    # Convert input strings to float
    input_features = [
        float(acousticness), float(danceability), float(energy),
        float(instrumentalness), float(liveness), float(loudness),
        float(speechiness), float(tempo), float(valence),
        float(popularity), float(duration_ms), float(encoded_album),
        float(age)
    ]

    scaled_input = x_scaler.transform(np.array(input_features).reshape(1, -1))
    prediction_result = model.predict(scaled_input)
    unscaled_output = y_scaler.inverse_transform(prediction_result.reshape(-1, 1))
    output_label = label[np.round(unscaled_output.item())]

    return output_label

css_code='body{background-image:url("file/editorial-use-only-and-no-commercial-use-at-any-time-no-use-news-photo-1701908722.jpg");}'

demo = gr.Interface(
    fn=prediction,
    inputs=[
    gr.Textbox(label="acousticness"),
    gr.Textbox(label="danceability"),
    gr.Textbox(label="energy"),
    gr.Textbox(label="instrumentalness"),
    gr.Textbox(label="liveness"),
    gr.Textbox(label="loudness"),
    gr.Textbox(label="speechiness"),
    gr.Textbox(label="tempo"),
    gr.Textbox(label="valence"),
    gr.Textbox(label="popularity"),
    gr.Textbox(label="duration_ms"),
    gr.Textbox(label="encoded_album"),
    gr.Textbox(label="age")
    ],
    outputs=gr.Textbox(label="Song Name"),
    description="🚀 Dive into the TaylorVerseVibes experience! Predict your personalized Taylor Swift song based on your unique vibes. 🌟🎵 Feel the magic with each prediction and let the music take you on a journey! 🎤✨",
    title="TaylorVerseVibes",
    css=css_code
)

if __name__ == '__main__':
    demo.launch()
