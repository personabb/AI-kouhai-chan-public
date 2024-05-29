print("importing...")
import os
from openai import OpenAI
#import whisper
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf
import sounddevice as sd
from pathlib import Path
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from pathlib import Path
from style_bert_vits2.tts_model import TTSModel
import queue
from style_bert_vits2.logging import logger
import torch

logger.remove()
print("imported") 
#事前準備
print("preparing...")
api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BEAM_SIZE = 3 if DEVICE == "cuda" else 2
MODEL_TYPE = "medium" if DEVICE == "cuda" else "medium"

model = WhisperModel(MODEL_TYPE, device=DEVICE, compute_type=COMPUTE_TYPE)

bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

model_file = "Anneli/Anneli_e116_s32000.safetensors"
config_file = "Anneli/config.json"
style_file = "Anneli/style_vectors.npy"

assets_root = Path("model_assets")

model_TTS = TTSModel(
    model_path=assets_root / model_file,
    config_path=assets_root / config_file,
    style_vec_path=assets_root / style_file,
    device=DEVICE
)

audio_que = queue.Queue()

PROMPT_FILE = "./prompt/ai.txt"
FIRST_MESSAGE_FILE = "./prompt/ai-first.txt"
with open(PROMPT_FILE) as f:
    sys_prompt = f.read()
with open(FIRST_MESSAGE_FILE) as f:
    first_message = f.read()

def call_first_message():
    sr, audio = model_TTS.infer(text=first_message)
    sd.play(audio, sr)
    sd.wait()

def speech2audio(fs=16000, silence_threshold=0.5, min_duration=0.1, amplitude_threshold=0.025):
    record_Flag = False

    non_recorded_data = []
    recorded_audio = []
    silent_time = 0
    input_time = 0
    start_threshold = 0.3
    all_time = 0
    
    with sd.InputStream(samplerate=fs, channels=1) as stream:
        while True:
            data, overflowed = stream.read(int(fs * min_duration))
            all_time += 1
            if all_time == 10:
                print("stand by ready OK")
            elif all_time >=10:
                if np.max(np.abs(data) > amplitude_threshold) and not record_Flag:
                    input_time += min_duration
                    if input_time >= start_threshold:
                        record_Flag = True
                        print("recording...")
                        recorded_audio=non_recorded_data[int(-1*start_threshold*10)-2:]  

                else:
                    input_time = 0

                if overflowed:
                    print("Overflow occurred. Some samples might have been lost.")
                if record_Flag:
                    recorded_audio.append(data)

                else:
                    non_recorded_data.append(data)

                if np.all(np.abs(data) < amplitude_threshold):
                    silent_time += min_duration
                    if (silent_time >= silence_threshold) and record_Flag:
                        print("finished")
                        record_Flag = False
                        break
                else:
                    silent_time = 0

    audio_data = np.concatenate(recorded_audio, axis=0)

    return audio_data

    
def audio2text(data, model):
    result = ""
    data = data.flatten().astype(np.float32)

    segments, _ = model.transcribe(data, beam_size=BEAM_SIZE)
    for segment in segments:
        result += segment.text

    return result


messages=[]
def text2text2speech(user_prompt, cnt):

  if cnt == 0:
    messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "assistant", "content": first_message})
    messages.append({"role": "user", "content": user_prompt})
  if cnt > 0:
    messages.append({"role": "user", "content": user_prompt})

  res_text = ""
  res_all = ""
  SP_Flag = False
  special_chars = {'.', '！', '？','。', '!', '?'}
  res = client.chat.completions.create(
    model="gpt-4-turbo",
    #model="gpt-3.5-turbo",
    messages = messages,           
    temperature=1.0 ,
    stream=True
  )

  for chunk in res:
    if chunk.choices[0].delta.content != None:
        if chunk.choices[0].delta.content in special_chars:
            if not SP_Flag:
                res_text += chunk.choices[0].delta.content
                SP_Flag = True
                print("message: ",res_text)
                sr, audio = model_TTS.infer(text=res_text)
                sd.play(audio, sr)
                sd.wait()
                res_all += res_text
                res_text = ""
        else:
            SP_Flag = False
            res_text += chunk.choices[0].delta.content

  messages.append({"role": "assistant", "content": res_all})


def process_roleai(audio_data, model,cnt):
    user_prompt = audio2text(audio_data, model)
    print("user: ",user_prompt)

    text2text2speech(user_prompt, cnt)



def main():
    cnt = 0
    call_first_message()
    while True:
        audio_data = speech2audio()
        process_roleai(audio_data, model,cnt)

        cnt+=1

if __name__ == "__main__":
    main()
