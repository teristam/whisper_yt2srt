#%%
import pickle
from datetime import datetime, timedelta
import pytube as pt
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import torch 
import click
import pickle 


def format_time(timestamp):
    # convert the timestamp from second to the timestamp in SRT
    if timestamp is not None:
        time = datetime(1,1,1) + timedelta(seconds=timestamp)
        format_time = time.strftime('%H:%M:%S')
        millisecond = int(timestamp%1*1000)
        return f'{format_time},{millisecond:03}'
    else:
        return None

def format_chunk(one_chunk):
    # convert each chunk to SRT format
    start_time = format_time(one_chunk['timestamp'][0])
    
    #the timestamp at the final sentence may be invalid
    if one_chunk['timestamp'][1] is not None:
        end_time = format_time(one_chunk['timestamp'][1])
    else:
        end_time = format_time(one_chunk['timestamp'][0]+5)
    return f'{start_time} --> {end_time} \n {one_chunk["text"]}\n\n'

def write_srt(res, out_filename='res.srt'):
    # write the timestamp to SRT file
    with open(out_filename, 'w', encoding="utf-8") as f:
        for i, chunk in enumerate(res['chunks']):
            str = format_chunk(chunk)
            f.write(f'{i}\n')
            f.write(str)
            
def write_file(res, out_filename='res.txt'):
    with open(out_filename, 'w', encoding="utf-8") as f:
        for i, chunk in enumerate(res['chunks']):
            f.write(chunk["text"]+'\n')
            

def init_pipeline():
    # initiate model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    MODEL_NAME = "openai/whisper-large-v3" #this always needs to stay in line 8 :D sorry for the hackiness

    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)


    device = 0 if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer = processor.tokenizer,
        feature_extractor= processor.feature_extractor,
        max_new_tokens = 128, 
        chunk_length_s = 30,
        batch_size = 16,
        return_timestamps = True,
        torch_dtype = torch_dtype,
        device=device,
    )
    
    return pipe

@click.command()
@click.argument('source')
@click.option('--output_type', '-t', default='text', help='Output format. text or srt, default text')
@click.option('--output_filename', '-o', default='output.txt', help='filename of the output SRT file. Default to output.srt')
def main(source:str, output_filename, output_type):
    pipe = init_pipeline()

    if source.startswith('http'):
        print(output_filename)
        print("Start downloading youtube files")
        yt = pt.YouTube(source)
        stream = yt.streams.filter(only_audio=True)[0]
        stream.download(filename="audio.mp3")
        print('Audio downloaded')
        res = pipe("audio.mp3")
    else:
        res = pipe(source)
        

    print('I will now convert the audio to text')
    
    with open('res.pkl','wb') as f:
        pickle.dump(res, f)
        
    if output_type == 'text':
        write_file(res, output_filename)
    else:
        write_srt(res, output_filename)
    
    print(f'Success! Result written to {output_filename}')    

