A tool to use the [Whisper v3](https://huggingface.co/openai/whisper-large-v3) model from OpenAI to convert audio in a Youtube video to text and export it as a SRT subtitle file. You can then use the SRT file with tools like [asbplayer](https://github.com/killergerbah/asbplayer) for sentence mining. All transcription is done locally.

### Installation
Clone this repository, create a virtual environment and then 

`pip install .`

Note: if you want to use GPU for interference, follow the official [PyTorch](https://pytorch.org/get-started/locally/) installation to get GPU working

### Usage
```
Usage: yt2srt [OPTIONS] YT_URL

Options:
  -o, --output_filename TEXT  filename of the output SRT file. Default to
                              output.srt
  --help                      Show this message and exit.
```
