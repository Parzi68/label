# from pyannote.audio import Pipeline
# import torch
# import librosa

# # Initialize the pipeline
# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=False)

# # Move the pipeline to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pipeline.to(device)

# def perform_diarization_and_segmentation(audio_file):
#     # Load audio file using librosa
#     audio, sample_rate = librosa.load(audio_file, sr=None)
    
#     # Convert to torch tensor
#     waveform = torch.from_numpy(audio).float().to(device)
    
#     # Perform diarization
#     diarization = pipeline({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})
    
#     # Process and print results
#     for segment, _, speaker in diarization.itertracks(yield_label=True):
#         start_time = segment.start
#         end_time = segment.end
#         print(f"dolly {speaker}: {start_time:.2f}s to {end_time:.2f}s")

# # Example usage
# audio_file = "2sec.mp3"
# perform_diarization_and_segmentation(audio_file)

# from pyannote.audio import Pipeline
# import torch
# import librosa
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import json

# # Initialize the pipeline
# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=False)

# # Move the pipeline to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pipeline.to(device)

# def extract_embeddings(audio_file):
#     audio, sample_rate = librosa.load(audio_file, sr=None)
#     waveform = torch.from_numpy(audio).float().to(device)
#     diarization = pipeline({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})
    
#     embeddings = []
#     for segment, _, speaker in diarization.itertracks(yield_label=True):
#         start_sample = int(segment.start * sample_rate)
#         end_sample = int(segment.end * sample_rate)
#         segment_waveform = waveform[start_sample:end_sample]
        
#         simple_embedding = segment_waveform.mean().cpu().numpy()
#         embeddings.append((speaker, simple_embedding.tolist()))
    
#     return embeddings

# def store_embeddings(embeddings, file_path):
#     with open(file_path, 'w') as f:
#         json.dump(embeddings, f)

# def load_embeddings(file_path):
#     with open(file_path, 'r') as f:
#         return json.load(f)

# def identify_speaker(new_embedding, stored_embeddings):
#     similarities = []
#     new_embedding = np.array(new_embedding).reshape(1, -1)
#     for speaker, embedding in stored_embeddings:
#         embedding = np.array(embedding).reshape(1, -1)
#         similarity = cosine_similarity(new_embedding, embedding)[0][0]
#         similarities.append((speaker, similarity))
#     return max(similarities, key=lambda x: x[1])[0]

# def iterate_and_differentiate_speakers(audio_file, speaker_names=None):
#     audio, sample_rate = librosa.load(audio_file, sr=None)
#     waveform = torch.from_numpy(audio).float().to(device)
#     diarization = pipeline({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})
    
#     speaker_segments = {}
#     for segment, _, speaker in diarization.itertracks(yield_label=True):
#         start_time = segment.start
#         end_time = segment.end
        
#         if speaker not in speaker_segments:
#             speaker_segments[speaker] = []
        
#         speaker_segments[speaker].append((start_time, end_time))
    
#     if speaker_names is None:
#         speaker_names = {speaker: f"Speaker {speaker}" for speaker in speaker_segments.keys()}
    
#     for speaker, segments in speaker_segments.items():
#         print(f"{speaker_names.get(speaker, f'Speaker {speaker}')}:")
#         for start, end in segments:
#             print(f"  {start:.2f}s to {end:.2f}s")
#         print()

# # Example usage
# audio_file = "2sec.mp3"
# embeddings = extract_embeddings(audio_file)
# store_embeddings(embeddings, "embeddings.json")

# stored_embeddings = load_embeddings("embeddings.json")

# new_audio_file = "dolloy.mp3"
# new_embeddings = extract_embeddings(new_audio_file)

# for speaker, new_embedding in new_embeddings:
#     identified_speaker = identify_speaker(new_embedding, stored_embeddings)
#     print(f"Original Speaker: {speaker}, Identified as: {identified_speaker}")

# print("\nDiarization and Speaker Differentiation:")

# # Define speaker names
# speaker_names = {
#     "SPEAKER_00": "Dolly",
#     "SPEAKER_01": "person2",
#     "SPEAKER_02": "person3",
#     "SPEAKER_03": "person4",
# }

# iterate_and_differentiate_speakers(audio_file, speaker_names)


import os
from pyannote.audio import Pipeline
import torch
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import soundfile as sf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
VOICE_SAMPLES_FOLDER = os.path.join(UPLOAD_FOLDER, 'voice_samples')
AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'audio')
OUTPUT_FOLDER = 'output'
os.makedirs(VOICE_SAMPLES_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

N_MFCC = 13

def extract_embeddings(audio_file):
    audio, sample_rate = librosa.load(audio_file, sr=None)
    waveform = torch.from_numpy(audio).float().to(device)
    diarization = pipeline({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})
    
    embeddings = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        segment_waveform = waveform[start_sample:end_sample]
        
        mfcc = librosa.feature.mfcc(y=segment_waveform.cpu().numpy(), sr=sample_rate, n_mfcc=N_MFCC)
        mfcc_embedding = np.mean(mfcc, axis=1)
        embeddings.append((segment, speaker, mfcc_embedding.tolist()))
    
    return embeddings

def save_voice_signatures(audio_file, speaker_names):
    embeddings = extract_embeddings(audio_file)
    
    try:
        with open("voice_signatures.json", "r") as f:
            voice_signatures = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, EOFError):
        voice_signatures = {}
    
    results = []
    audio, sample_rate = librosa.load(audio_file, sr=None)
    for i, (segment, speaker, embedding) in enumerate(embeddings):
        if i < len(speaker_names):
            speaker_name = speaker_names[i]
            voice_signatures[speaker_name] = embedding
            
            segment_file = os.path.join(VOICE_SAMPLES_FOLDER, f"{speaker_name}.wav")
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)
            segment_audio = audio[start_sample:end_sample]
            sf.write(segment_file, segment_audio, sample_rate)
            
            results.append(f"Voice signature for {speaker_name} saved successfully.")
        else:
            results.append(f"No name provided for speaker {i+1}.")
    
    with open("voice_signatures.json", "w") as f:
        json.dump(voice_signatures, f)
    
    return results

def identify_speaker(embedding):
    try:
        with open("voice_signatures.json", "r") as f:
            voice_signatures = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, EOFError):
        return "Unknown Speaker"
    
    similarities = []
    for known_speaker, known_embedding in voice_signatures.items():
        similarity = cosine_similarity([embedding], [known_embedding])[0][0]
        similarities.append((known_speaker, similarity))
    
    if similarities:
        identified_speaker, max_similarity = max(similarities, key=lambda x: x[1])
        if max_similarity > 0.8:
            return identified_speaker
    
    return "Unknown Speaker"

def diarize_and_identify(audio_file):
    audio, sample_rate = librosa.load(audio_file, sr=None)
    waveform = torch.from_numpy(audio).float().to(device)
    diarization = pipeline({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})
    
    results = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end
        
        segment_audio = audio[int(start_time*sample_rate):int(end_time*sample_rate)]
        segment_file = f"temp_segment_{start_time}_{end_time}.wav"
        sf.write(segment_file, segment_audio, sample_rate)
        
        embeddings = extract_embeddings(segment_file)
        if embeddings:
            speaker_name = identify_speaker(embeddings[0][2])
        else:
            speaker_name = "Unknown Speaker"
        
        mood = "neutral"
        message = "No message detected"
        
        results.append(f"[{start_time:.2f} - {end_time:.2f}] - {speaker_name} - {mood} - {message}")
        
        os.remove(segment_file)
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_voice_signature', methods=['POST'])
@app.route('/save_voice_signatures', methods=['POST'])
def save_voice_signatures_route():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    speaker_names = request.form.getlist('speaker_names')
    
    if audio_file.filename == '':
        return jsonify({"error": "No selected audio file"}), 400
    
    if not speaker_names:
        speaker_names = [request.form.get('speaker_name')]  # Fallback to single speaker name
    
    if not any(speaker_names):
        return jsonify({"error": "No speaker names provided"}), 400
    
    audio_path = os.path.join(AUDIO_FOLDER, secure_filename(audio_file.filename))
    audio_file.save(audio_path)
    
    results = save_voice_signatures(audio_path, speaker_names)
    os.remove(audio_path)
    return jsonify({"messages": results})

@app.route('/identify_speakers', methods=['POST'])
def identify_speakers_route():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected audio file"}), 400
    
    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    audio_file.save(audio_path)
    
    results = diarize_and_identify(audio_path)
    
    # Create output text file
    output_filename = os.path.splitext(filename)[0] + '.txt'
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    with open(output_path, 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    os.remove(audio_path)
    return jsonify(results)

@app.route('/diarize_and_identify', methods=['POST'])
def diarize_and_identify_route():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected audio file"}), 400
    
    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    audio_file.save(audio_path)
    
    results = diarize_and_identify(audio_path)
    
    # Create output text file
    output_filename = os.path.splitext(filename)[0] + '.txt'
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    with open(output_path, 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    os.remove(audio_path)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
