import librosa
import pretty_midi
import os
import pathlib

#todo: function to synthesize the midi using fluidsynth and just return it as a data obj, rather than having to do these file reads/writes

def synthesize_midi(score_path, fs=44100):
    score_synth = pretty_midi.PrettyMIDI(score_path).fluidsynth(fs=fs)
    return score_synth
    




