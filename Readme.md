# How to use This Rag Ai Teaching Assistant in your own data

## Step 1 - Collect your Videos
Move all your videos file to videos folder

## Step 2 - Convert to mp3
convert all the videos files to mp3 by running video_to_mp3

## Step 3 - Convert mp3 to json
Convert all the mp3 files to json by running Mp3_to_json

## Step 4 - Convert json file to vectors
Use the file Preprocess_json to convert the json files to dataframe with embeddings and save it as a joblib pickle

## Step 5 - Prompt generation and fedding to LLM
read the joblib file and load it to the memory. Then create a relevant prompt and feed it to the LLM



