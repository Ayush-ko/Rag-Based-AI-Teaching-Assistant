import whisper
import json
import os

# Load Whisper model (uncomment if you need transcription later)
model = whisper.load_model("large-v2")

# List only audio files


audios = os.listdir("audios")

# sort safely: check "_" first, then convert number
audios.sort(key=lambda x: int(x.split("_")[0]) if "_" in x else 9999)

seen = set()

for audio in audios:
    if "_" in audio and audio.lower().endswith(".mp3"):
        number = audio.split("_")[0]
        title = audio.split("_", 1)[1].rsplit(".", 1)[0]

        if title not in seen:
            seen.add(title)
            print(f"{number} {title}")
            #result = model.transcribeaudio = f"audios/{audio}.mp3",
            result = model.transcribe(audio = f"audios/{audio}",
                              language = "hi",
                              task = "translate",
                              word_timestamps=False)
            chunks = []
            for segment in result["segments"]:
                chunks.append({ "number": number, "title":title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})
                
                chunks_with_metadata = {"chunks": chunks , "text": result["text"]}
            with open(f"json/{audio}.json", "w") as f:
                json.dump(chunks_with_metadata, f)