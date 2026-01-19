#convert the video to mp3
import os
import subprocess

# Load files from videos folder
files = os.listdir("videos")

# Ensure audios folder exists
os.makedirs("audios", exist_ok=True)

for file in files:
    # Remove quality part like (720p, h264).mp4
    clean = file.split(" (")[0]

    # Extract only real title before Sigma tutorial text
    if "Sigma Web Development Course - Tutorial" in clean:
        file_name = clean.split("Sigma Web Development Course - Tutorial")[0].strip(" -")
    else:
        file_name = clean.strip(" -")

    # Extract tutorial number safely
    if " #" in clean and " - CodeWithHarry" in clean:
        tutorial_number = clean.split(" #")[1].split(" -")[0]
    else:
        tutorial_number = "NA"

    print(tutorial_number, file_name)

    # Convert to mp3 using ffmpeg
    subprocess.run(
        ["ffmpeg", "-i", f"videos/{file}", f"audios/{file_name}.mp3"],
        shell=True
    )