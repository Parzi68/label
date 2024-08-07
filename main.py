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

from pyannote.audio import Pipeline
import torch
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Initialize the pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=False)

# Move the pipeline to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

def extract_embeddings(audio_file):
    audio, sample_rate = librosa.load(audio_file, sr=None)
    waveform = torch.from_numpy(audio).float().to(device)
    diarization = pipeline({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})
    
    embeddings = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        segment_waveform = waveform[start_sample:end_sample]
        
        simple_embedding = segment_waveform.mean().cpu().numpy()
        embeddings.append((speaker, simple_embedding.tolist()))
    
    return embeddings

def store_embeddings(embeddings, file_path):
    with open(file_path, 'w') as f:
        json.dump(embeddings, f)

def load_embeddings(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def identify_speaker(new_embedding, stored_embeddings):
    similarities = []
    new_embedding = np.array(new_embedding).reshape(1, -1)
    for speaker, embedding in stored_embeddings:
        embedding = np.array(embedding).reshape(1, -1)
        similarity = cosine_similarity(new_embedding, embedding)[0][0]
        similarities.append((speaker, similarity))
    return max(similarities, key=lambda x: x[1])[0]

def iterate_and_differentiate_speakers(audio_file, speaker_names=None):
    audio, sample_rate = librosa.load(audio_file, sr=None)
    waveform = torch.from_numpy(audio).float().to(device)
    diarization = pipeline({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})
    
    speaker_segments = {}
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end
        
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        
        speaker_segments[speaker].append((start_time, end_time))
    
    if speaker_names is None:
        speaker_names = {speaker: f"Speaker {speaker}" for speaker in speaker_segments.keys()}
    
    for speaker, segments in speaker_segments.items():
        print(f"{speaker_names.get(speaker, f'Speaker {speaker}')}:")
        for start, end in segments:
            print(f"  {start:.2f}s to {end:.2f}s")
        print()

# Example usage
audio_file = "audio.mp3"
embeddings = extract_embeddings(audio_file)
store_embeddings(embeddings, "embeddings.json")

stored_embeddings = load_embeddings("embeddings.json")

new_audio_file = "audio.mp3"
new_embeddings = extract_embeddings(new_audio_file)

for speaker, new_embedding in new_embeddings:
    identified_speaker = identify_speaker(new_embedding, stored_embeddings)
    print(f"Original Speaker: {speaker}, Identified as: {identified_speaker}")

print("\nDiarization and Speaker Differentiation:")

# Define speaker names
speaker_names = {
    "SPEAKER_00": "person1",
    "SPEAKER_01": "person2",
    "SPEAKER_02": "person3",
    "SPEAKER_03": "person4",
}

iterate_and_differentiate_speakers(audio_file, speaker_names)
