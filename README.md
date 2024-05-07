A tool to use the [Whisper v3](https://huggingface.co/openai/whisper-large-v3) model from OpenAI to convert audio in a Youtube video to text and export it as a SRT subtitle file. You can then use the SRT file with tools like [asbplayer](https://github.com/killergerbah/asbplayer) for sentence mining. All transcription is done locally.

It also support transcribing audio file locally.

### Installation
Clone this repository, create a virtual environment and then 

`pip install .`

Note: if you want to use GPU for inference, follow the official [PyTorch](https://pytorch.org/get-started/locally/) installation to get GPU working.

### Usage
```
Usage: yt2srt [OPTIONS] SOURCE

Options:
  -t, --output_type TEXT      Output format. text or srt, default text
  -o, --output_filename TEXT  filename of the output file. Default to
                              output.text
  --help                      Show this message and exit.
```
