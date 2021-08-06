import logging
from datetime import datetime
import collections, queue, os, os.path
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
import tkinter as tk
from tkinter.constants import CENTER
import glob
import vibes
import os, os.path
import deepspeech
import wget

"""VibeTranscribe GUI"""
"""Build a floating, movable and updateable label and get rid of the rest of the window, have it be always on top"""

def switchColor(color):
    lab.config(bg=color)

lastClickX = 0
lastClickY = 0

def SaveLastClickPos(event):
    global lastClickX, lastClickY
    lastClickX = event.x
    lastClickY = event.y

def Dragging(event):
    x, y = event.x - lastClickX + root.winfo_x(), event.y - lastClickY + root.winfo_y()
    root.geometry("+%s+%s" % (x , y))

root = tk.Tk()
root.overrideredirect(True)
root.title('VibeTranscribe')
root.geometry('900x80')
root['bg'] = 'grey'
root.attributes('-transparentcolor', 'grey')
root.attributes('-topmost', True)
root.bind('<Button-1>', SaveLastClickPos)
root.bind('<B1-Motion>', Dragging)

var = tk.StringVar()
lab = tk.Label(root, textvariable=var, font='Helvetica 20 bold', bg='black', fg='white', justify=CENTER, wraplength=900)
lab.pack()

"""Set up buffered audio streaming from microphone, implement mic vad streaming for DeepSpeech"""

logging.basicConfig(level=20)

class Audio(object):

    FORMAT = pyaudio.paInt16
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        data16 = np.frombuffer(buffer=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tobytes()

    def read_resampled(self):
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("Predicting %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

"""Filter and segment audio with VAD"""

class VADAudio(Audio):

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

"""Load DeepSpeech, stream and dynamically update the GUI"""

def main(ARGS):
    if os.path.isdir(ARGS.model):
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, 'output_graph.pb')
        ARGS.scorer = os.path.join(model_dir, ARGS.scorer)
    
    if os.path.isfile(ARGS.model) or os.path.isdir(ARGS.model):
        if ARGS.scorer and os.path.isfile(ARGS.scorer) or os.path.isdir(ARGS.scorer):
            print('Initializing model...')
            logging.info("ARGS.model: %s", ARGS.model)
            model = deepspeech.Model(ARGS.model)
            logging.info("ARGS.scorer: %s", ARGS.scorer)
            model.enableExternalScorer(ARGS.scorer)
        else:
            print('# Deepspeech scorer not found. Downloading it... #')
            scorerurl = 'https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer'
            wget.download(scorerurl, 'Models/deepspeech-0.9.3-models.scorer')
            print('\n')
            main(ARGS)
    else:
        print('# Deepspeech model not found. Downloading it... #')
        modelurl = 'https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm'
        wget.download(modelurl, 'Models/deepspeech-0.9.3-models.pbmm')
        print('\n')
        main(ARGS)

    vad_audio = VADAudio(aggressiveness=ARGS.vad_aggressiveness,
                         device=ARGS.device,
                         input_rate=ARGS.rate,
                         file=ARGS.file)
    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()

    # Stream from microphone to DeepSpeech using VAD
    spinner = None
    if not ARGS.nospinner:
        spinner = Halo(spinner='line')
    stream_context = model.createStream()
    wav_data = bytearray()
    for frame in frames:
        if frame is not None:
            if spinner: spinner.start()
            logging.debug("streaming frame")
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            if ARGS.savewav: wav_data.extend(frame)
            intertext = stream_context.intermediateDecode()
            #hardcoded solution for subtitle placement in long streams
            #still got issues if the stream is too long
            #may fix later
            if len(intertext) >= 125+62*6:
                var.set(intertext[-125:])
            elif len(intertext) >= 125+62*5:
                var.set(intertext[62 * 6:])
            elif len(intertext) >= 125+62*4:
                var.set(intertext[62 * 5:])
            elif len(intertext) >= 125+62*3:
                var.set(intertext[62 * 4:])
            elif len(intertext) >= 125+62*2:
                var.set(intertext[62 * 3:])
            elif len(intertext) >= 125+62:
                var.set(intertext[62 * 2:])
            elif len(intertext) >= 125:
                var.set(intertext[62:])
            else:
                var.set(intertext)
            root.update()
        else:
            if spinner: spinner.stop()
            logging.debug("end utterence")
            if ARGS.savewav:
                vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("temp_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                wav_data = bytearray()
                wavname = max(glob.iglob('temp/*.wav'), key=os.path.getctime)
                correct = vibes.discard_iftooshort(wavname)
                if correct == 1:
                    vibe = vibes.vibecheck(wavname)
                    color = vibes.vibetocolor(vibe)
            text = stream_context.finishStream()
            if len(text) >= 125+62*6:
                var.set(text[-125:])
            if len(text) >= 125+62*5:
                var.set(text[62 * 6:])
            elif len(text) >= 125+62*4:
                var.set(text[62 * 5:])
            elif len(text) >= 125+62*3:
                var.set(text[62 * 4:])
            elif len(text) >= 125+62*2:
                var.set(text[62 * 3:])
            elif len(text) >= 125+62:
                var.set(text[62 * 2:])
            elif len(text) >= 125:
                var.set(text[62:])
            else:
                var.set(text)
            switchColor(color)
            root.update()
            dir = 'temp/'
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
            stream_context = model.createStream()

"""Deepspeech's model, scorer and other useful arguments, loads VibeTranscribe's defaults otherwise"""      

if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 16000

    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")

    parser.add_argument('-v', '--vad_aggressiveness', type=int, default=3,
                        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3")
    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner")
    parser.add_argument('-w', '--savewav', default='temp/',
                        help="Save .wav files of utterences to given directory")
    parser.add_argument('-f', '--file',
                        help="Read from .wav file instead of microphone")
    parser.add_argument('-m', '--model', default='Models/deepspeech-0.9.3-models.pbmm',
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")
    parser.add_argument('-s', '--scorer', default='Models/deepspeech-0.9.3-models.scorer',
                        help="Path to the external scorer file.")
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")

    ARGS = parser.parse_args()
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)