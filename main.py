from pathlib import Path
import uuid
import zipfile

import librosa
import numpy as np
import pretty_midi

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="LibHitzSound AI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")


@app.get("/")
def home():
    return {
        "service": "LibHitzSound AI Studio",
        "status": "online"
    }


def midi_file(path, notes, tempo):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)

    for start, end, pitch, velocity in notes:
        inst.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=end
            )
        )

    midi.instruments.append(inst)
    midi.write(str(path))


@app.post("/api/beat-analyze")
async def analyze_beat(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(400, "No audio file.")

    allowed = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(400, "Unsupported audio format.")

    job = uuid.uuid4().hex
    folder = OUT / job
    folder.mkdir()

    audio_path = folder / ("audio" + ext)

    with open(audio_path, "wb") as f:
        f.write(await file.read())

    try:
        y, sr = librosa.load(
            str(audio_path),
            sr        )

        tempo, _ = librosa.beat.beat_track(
            y=y,
            sr=sr
        )

        tempo = float(np.asarray(tempo).reshape(-1)[0])

        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=sr
        )

        root = int(np.argmax(np.mean(chroma, axis=1)))

        keys = [
            "C", "C#", "D", "D#", "E", "F",
            "F#", "G", "G#", "A", "A#", "B"
        ]

        key = keys[root]

        duration = librosa.get_duration(
            y=y,
            sr=sr
        )

        beat_time = 60.0 / max(tempo, 1)

        kick = []
        snare = []
        hihat = []
        bass = []
        chords = []
        melody = []
        counter = []

        t = 0.0

        while t < duration:
            beat = int(t / beat_time) % 4

            if beat in [0, 2]:
                kick.append((t, t + 0.1, 36, 110))

            if beat in [1, 3]:
                snare.append((t, t + 0.1, 38, 100))

            hihat.append((t, t + 0.05, 42, 70))

            bass.append(
                (t, t + beat_time * 0.8, 36 + root, 90)
            )

            t += beat_time

        t = 0.0

        while t < duration:
            chords.append(
                (
                    t,
                    min(t + beat_time * 4, duration),
                    48 + root,
                    65
                )
            )

            melody.append(
                (
                    t,
                    min(t + beat_time, duration),
                    60 + root,
                    70
                )
            )

            counter.append(
                (
                    t,
                    min(t + beat_time, duration),
                    72 + root,
                    55
                )
            )

            t += beat_time * 4

        parts = {
            "Kick.mid": kick,
            "Snare.mid": snare,
            "HiHat.mid": hihat,
            "Bass.mid": bass,
            "Chords.mid": chords,
            "Melody.mid": melody,
            "Counter_Melody.mid": counter
        }

        created = []

        for name, notes in parts.items():
            path = folder / name
            midi_file(path, notes, tempo)
            created.append(path)

        full = folder / "Full_Beat.mid"

        all_notes = []

        for notes in parts.values():
            all_notes.extend(notes)

        midi_file(full, all_notes, tempo)
        created.append(full)

        zip_path = folder / "LibHitzSound_MIDI_Pack.zip"

        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED
        ) as z:
            for path in created:
                z.write(path, path.name)

        return {
            "success": True,
            "filename": file.filename,
            "bpm": round(tempo, 2),
            "key": key,
            "midi_files": [
                "/outputs/" + job + "/" + p.name
                for p in created
            ],
            "zip": "/outputs/" + job +
                   "/LibHitzSound_MIDI_Pack.zip"
        }

    except Exception as e:
        raise HTTPException(
            500,
            "Analysis failed: " + str(e)
      )
