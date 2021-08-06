# VibeTranscribe 0.1
### Video Demo: 
<https://youtu.be/AFVdVFqdGsg>
### Description: 
<p>
VibeTranscribe is among the first, if not the very first live captioner to also provide emotional context according to the tone of voice. In this version it detects high, neutral and low mood or enthusiasm, but it'll hopefully be updated to support a wide range of emotions with even better accuracy.
</p>
<p>
It implements the default Deepspeech 0.9.3 model and scorer as well as a bagging classifier from sklearn I trained with a compressed version of the RAVDESS database. I haven't been successful at using deepspeech-gpu yet, but it'd greatly accelerate its slow inference. This is its' greatest weakness as of version 0.1.
</p>

### The Future of Subtitles:
<p>
With the advent of -good- AI, we're on the verge of the next generation of subtitles and accessibility for the deaf and hard of hearing. The new subtitles will not only be accurate in real time, but they'll also identify and convey both the speaker's identity and the emotional tone of their voice at any given moment, which is an overlooked yet important component of human communication that many people have been missing on - until now!
Currently, this program is only an amateur's project for a CS course, however if organizations don't get too far ahead of me I look forward to further developing this. Collaboration is welcome.
</p>

## Usage and Info:
<p>
It'll use your default microphone. If you want to transcribe sound coming from the computer, use Stereo Mix as your default. If you are on Windows, it should show up among your recording devices, just make sure you tick 'show disabled devices'. <a href="https://cdn-haiwai.recmaster.net/wp-content/uploads/2020/06/show-disabled-devices-stereo-mix.jpg" title="Google">Example</a>.

Deepspeech's model and scorer will be downloaded to the Models directory upon running the program.
</p>

Run:
1. >pip install -r requeriments.txt
2. >python vibetranscribe.py
3. Now speak or listen to something for the draggable subtitles to appear.
When a stream of words is complete, the emotion will update. The subtitles' background will be black if the previous stream was neutral, golden if it was happy or enthusiastic, and silver if it was sad or unenthusiastic.

### vibetranscribe.py
- Implements the GUI by means of a floating, movable and updateable tkinter label, has it be always on top and gets rid of the rest of the window.
- Sets up buffered audio streaming from microphone, implements mic vad streaming for DeepSpeech.
- Filters and segments audio with VAD.
- Loads or downloads DeepSpeech, streams and dynamically updates the GUI calling the appropiate functions.
- Parses multiple arguments.

### vibes.py
- Extracts the necessary features for emotion prediction from an audio file.
- Unpickles the trained model, loads it and predicts emotions from each finished stream's wav file.
- Discards useless temporary wav files and associates predicted emotions with text background colors.

## Plans for VibeTranscribe:
- Make it work with deepspeech-gpu. 
- Train a better model with more emotions. 
- Perhaps also train the deepspeech model.

##### Version 0.1 is emtheapprentice's CS50 final project.
