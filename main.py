from pathlib import Path
import subprocess
import uuid
import zipfile
import logging

import librosa
import numpy as np
import pretty_midi

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("libhitzsound-ai")


app = FastAPI(
    title="LibHitzSound AI Studio",
    version="0.1.0"
)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


app.mount(
    "/outputs",
    StaticFiles(directory=str(OUT)),
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

    instrument = pretty_midi.Instrument(program=0)

    for start, end, pitch, velocity in notes:
        if end <= start:
            continue

        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(max(1, min(127, velocity))),
                pitch=int(max(0, min(127, pitch))),
                start=float(start),
                end=float(end)
            )
        )

    midi.instruments.append(instrument)
    midi.write(str(path))


def convert_to_wav(input_path, output_path):
    """
    Convert any supported audio format to a standard WAV file.
    FFmpeg is installed by the Dockerfile.
    """

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-sample_fmt",
        "s16",
        str(output_path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        logger.error("FFmpeg error: %s", result.stderr)

        raise RuntimeError(
            "FFmpeg could not decode the uploaded audio file."
        )


@app.post("/api/beat-analyze")
async def analyze_beat(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file was uploaded."
        )

    allowed = {
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".ogg",
        ".aac"
    }

    ext = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Use MP3, WAV, M4A, FLAC, OGG, or AAC."
            )
        )

    job = uuid.uuid4().hex

    folder = OUT / job
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    original_path = folder / (
        "original" + ext
    )

    wav_path = folder / "analysis.wav"

    try:

        # Save upload
        audio_data = await file.read()

        if not audio_data:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty."
            )

        with open(original_path, "wb") as audio_file:
            audio_file.write(audio_data)

        logger.info(
            "Received audio: %s (%d bytes)",
            file.filename,
            len(audio_data)
        )

        # Convert audio through FFmpeg
        convert_to_wav(
            original_path,
            wav_path
        )

        # Load converted WAV
        y, sr = librosa.load(
            str(wav_path),
            sr=44100,
            mono=True
        )

        if y is None or len(y) == 0:
            raise RuntimeError(
                "The audio contains no readable samples."
            )

        duration = float(
            librosa.get_duration(
                y=y,
                sr=sr
            )
        )

        if duration < 0.5:
            raise RuntimeError(
                "The audio is too short to analyze."
            )

        logger.info(
            "Audio loaded successfully: %.2f seconds, %d Hz",
            duration,
            sr
        )

        # BPM
        tempo_result, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr
        )

        tempo_array = np.asarray(
            tempo_result
        ).reshape(-1)

        if len(tempo_array) > 0:
            tempo = float(tempo_array[0])
        else:
            tempo = 120.0

        if not np.isfinite(tempo) or tempo <= 0:
            tempo = 120.0

        tempo = min(max(tempo, 40.0), 240.0)

        # Key detection
        try:
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

        except Exception as key_error:

            logger.warning(
                "Key detection failed: %s",
                key_error
            )

            # Fall back to C
            root = 0

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

        beat_time = 60.0 / tempo

        kick = []
        snare = []
        hihat = []
        percussion = []
        bass = []
        chords = []
        melody = []
        counter_melody = []

        # Generate beat structure
        t = 0.0

        while t < duration:

            beat = int(
                t / beat_time
            ) % 4

            end_time = min(
                t + 0.10,
                duration
            )

            if beat in [0, 2]:

                kick.append(
                    (
                        t,
                        end_time,
                        36,
                        110
                    )
                )

            if beat in [1, 3]:

                snare.append(
                    (
                        t,
                        end_time,
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

        # Generate musical parts
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
            "Counter_Melody.mid": counter_melody
        }

        created = []

        # Create individual MIDI files
        for name, notes in parts.items():

            midi_path = folder / name

            create_midi(
                midi_path,
                notes,
                tempo
            )

            created.append(midi_path)

        # Full MIDI
        full_path = folder / "Full_Beat.mid"

        all_notes = []

        for notes in parts.values():
            all_notes.extend(notes)

        create_midi(
            full_path,
            all_notes,
            tempo
        )

        created.append(full_path)

        # ZIP
        zip_path = folder / "LibHitzSound_MIDI_Pack.zip"

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

        logger.info(
            "Analysis completed successfully: %s",
            file.filename
        )

        base_url = "/outputs/" + job + "/"

        return {
            "success": True,
            "filename": file.filename,
            "bpm": round(tempo, 2),
            "key": key,
            "duration_seconds": round(duration, 2),

            "midi_files": [
                base_url + midi_path.name
                for midi_path in created
            ],

            "zip": (
                base_url +
                "LibHitzSound_MIDI_Pack.zip"
            )
        }

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            "Beat analysis failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(error)}"
        )

    finally:

        # Remove temporary WAV after analysis
        try:
            if wav_path.exists():
                wav_path.unlink()
        except Exception:
            pass
