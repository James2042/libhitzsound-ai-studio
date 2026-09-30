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


app.mount(
    "/outputs",
    StaticFiles(directory=OUT),
    name="outputs"
)


@app.get("/")
def home():
    return {
        "service": "LibHitzSound AI Studio",
        "status": "online"
    }


def create_midi(path, notes, tempo):
    midi = pretty_midi.PrettyMIDI(
        initial_tempo=max(float(tempo), 1.0)
    )

    instrument = pretty_midi.Instrument(
        program=0
    )

    for start, end, pitch, velocity in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=float(start),
                end=float(end)
            )
        )

    midi.instruments.append(instrument)
    midi.write(str(path))


@app.post("/api/beat-analyze")
async def analyze_beat(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file."
        )

    allowed = [
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".ogg"
    ]

    ext = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format."
        )

    job = uuid.uuid4().hex

    folder = OUT / job
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    audio_path = folder / (
        "audio" + ext
    )

    audio_data = await file.read()

    with open(audio_path, "wb") as audio_file:
        audio_file.write(audio_data)

    try:

        y, sr = librosa.load(
            str(audio_path),
            sr=None,
            mono=True
        )

        if len(y) == 0:
            raise Exception(
                "The audio file contains no readable audio."
            )

        tempo_result, beat_frames = (
            librosa.beat.beat_track(
                y=y,
                sr=sr
            )
        )

        tempo_array = np.asarray(
            tempo_result
        ).reshape(-1)

        if len(tempo_array) == 0:
            tempo = 120.0
        else:
            tempo = float(
                tempo_array[0]
            )

        if tempo <= 0:
            tempo = 120.0

        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=sr
        )

        chroma_average = np.mean(
            chroma,
            axis=1
        )

        root = int(
            np.argmax(chroma_average)
        )

        keys = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B"
        ]

        key = keys[root]

        duration = librosa.get_duration(
            y=y,
            sr=sr
        )

        beat_time = 60.0 / tempo

        kick = []
        snare = []
        hihat = []
        percussion = []
        bass = []
        chords = []
        melody = []
        counter_melody = []

        t = 0.0

        while t < duration:

            beat = int(
                t / beat_time
            ) % 4

            if beat in [0, 2]:

                kick.append(
                    (
                        t,
                        min(
                            t + 0.10,
                            duration
                        ),
                        36,
                        110
                    )
                )

            if beat in [1, 3]:

                snare.append(
                    (
                        t,
                        min(
                            t + 0.10,
                            duration
                        ),
                        38,
                        100
                    )
                )

            hihat.append(
                (
                    t,
                    min(
                        t + 0.05,
                        duration
                    ),
                    42,
                    70
                )
            )

            if beat in [1, 3]:

                percussion.append(
                    (
                        t,
                        min(
                            t + 0.08,
                            duration
                        ),
                        45,
                        60
                    )
                )

            bass.append(
                (
                    t,
                    min(
                        t + beat_time * 0.8,
                        duration
                    ),
                    36 + root,
                    90
                )
            )

            t += beat_time

        t = 0.0

        while t < duration:

            chord_end = min(
                t + beat_time * 4,
                duration
            )

            chords.append(
                (
                    t,
                    chord_end,
                    48 + root,
                    65
                )
            )

            melody.append(
                (
                    t,
                    min(
                        t + beat_time,
                        duration
                    ),
                    60 + root,
                    70
                )
            )

            counter_melody.append(
                (
                    t,
                    min(
                        t + beat_time,
                        duration
                    ),
                    72 + root,
                    55
                )
            )

            t += beat_time * 4

        parts = {

            "Kick.mid": kick,

            "Snare.mid": snare,

            "HiHat.mid": hihat,

            "Percussion.mid": percussion,

            "Bass.mid": bass,

            "Chords.mid": chords,

            "Melody.mid": melody,

            "Counter_Melody.mid":
                counter_melody
        }

        created = []

        for name, notes in parts.items():

            midi_path = folder / name

            create_midi(
                midi_path,
                notes,
                tempo
            )

            created.append(
                midi_path
            )

        full_path = (
            folder / "Full_Beat.mid"
        )

        all_notes = []

        for notes in parts.values():
            all_notes.extend(
                notes
            )

        create_midi(
            full_path,
            all_notes,
            tempo
        )

        created.append(
            full_path
        )

        zip_path = (
            folder /
            "LibHitzSound_MIDI_Pack.zip"
        )

        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED
        ) as midi_zip:

            for midi_path in created:

                midi_zip.write(
                    midi_path,
                    midi_path.name
                )

        return {

            "success": True,

            "filename": file.filename,

            "bpm": round(
                tempo,
                2
            ),

            "key": key,

            "duration_seconds":
                round(
                    duration,
                    2
                ),

            "midi_files": [

                "/outputs/"
                + job
                + "/"
                + midi_path.name

                for midi_path
                in created
            ],

            "zip":
                "/outputs/"
                + job
                + "/"
                + "LibHitzSound_MIDI_Pack.zip"
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=
                "Analysis failed: "
                + str(error)
            )
