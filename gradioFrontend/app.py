#!/usr/bin/env python
# coding: utf-8

import time
from typing import Union
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
import re
import whisper
import os
from dotenv import load_dotenv, find_dotenv
import openai
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.node_parser import SentenceWindowNodeParser
import gradio as gr

import warnings 
warnings.filterwarnings("ignore")

# load openai api key
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_transcript(url: str=None, path: str=None, output_timestamp: bool=False) -> str:
  """The `get_transcript` function is a Python function that retrieves the transcript of a video
# either from a YouTube URL or a local file path. Here is a breakdown of what it does:

  Args:
      url (str, optional): _description_. Defaults to None.
      path (str, optional): _description_. Defaults to None.
      output_timestamp (bool, optional): _description_. Defaults to False.

  Returns:
      str: _description_
  """
  
  if url:
    transcript = check_existing_transcript(url) 
    print(transcript)

    if not transcript:
      file = download_from_youtube(url)
    else:
      return transcript

  elif path: 
    file = load_from_local_file(path)
    
  transcript = convert_speech_to_text(file)
    
  return transcript 


def get_video_id(url: str) -> str:
    """The `get_video_id` function is extracting the unique video ID from a YouTube 
    video URL. It uses a regular expression to search for the video ID pattern in 
    the URL and returns the extracted video ID as a string.

    Args:
        url (str): _description_

    Returns:
        str: _description_
    """
    # [TODO]: try and raise error
    return re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)[1]


def check_existing_transcript(url: str, language_code="en", manual_only=True, output_timestamp=False) -> Union[str, bool]:
  """
  `check_existing_transcript` is a function that checks if a transcript exists for 
  a given YouTube video URL. It first extracts the video ID from the URL and then 
  attempts to list the available transcripts for that video using the YouTubeTranscriptApi. 
  If the transcript is found and matches the specified language code and generation type 
  (manual or not), it retrieves the transcript text. The function also has the option to 
  output timestamps along with the transcript text. If any errors occur during the process, 
  it will return False.

  Args:
      url (str): the url of the YouTube video 
      language_code (str, optional): desired language code for the existing transcript. Defaults to "en".
      manual_only (bool, optional): if only download manual transcript. Defaults to False.
      output_timestamp (bool, optional): if output timestamp. Defaults to False.

  Returns:
      transcript: transcript in string format or False if the transcript does not exist
  """
  # [TODO] use logging to capture errors
  video_id = get_video_id(url)
  try:
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript_list = {t.language_code: not t.is_generated for t in transcript_list}
    print(transcript_list)
  except Exception as e:
    print(e)
    return False
  
  if language_code in transcript_list and transcript_list[language_code] is manual_only:
    transcription_ = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])
    transcript = ""
    
    if output_timestamp:
      for tr in transcription_:
        start = time.strftime("%H:%M:%S", time.gmtime(tr["start"]))
        end = time.strftime("%H:%M:%S", time.gmtime(tr["start"]))
        transcript += f"[{start}-{end}] {tr['text']}" + "\n"
    else:
      for tr in transcription_:
        transcript += tr["text"]+" "
  else:
    print(f"the specified language code {language_code} does not exist.")
    return False
    
  return transcript


def download_from_youtube(url, output_path=""):
  """_summary_

  Args:
      url (_type_): _description_
      output_path (str, optional): _description_. Defaults to "".
  """
  
  youtube_dl_opts = {
  "outtmpl": output_path+"%(title)s.%(ext)s",
  "format": "bestaudio/best",
  'postprocessors': [{
    'key': 'FFmpegExtractAudio', 
    'preferredcodec': 'mp3', 
    'preferredquality': '192'
  }]
}

  with YoutubeDL(youtube_dl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=True)

  return ydl.prepare_filename(info_dict).replace("webm", "mp3")


def load_from_local_file(path: str) -> None:
  """_summary_

  Args:
      path (str): _description_

  Returns:
      _type_: _description_
  """
  return path

def convert_speech_to_text(file, model_size="tiny", language="en", fp16=False, verbose=False):
  """The `convert_speech_to_text` function is responsible for converting 
  speech audio data into text. It takes the audio file as input along with 
  optional parameters such as the model size, language, whether to use 
  fp16 precision, and a verbosity flag.

  Args:
      file (_type_): _description_
      model_size (str, optional): _description_. Defaults to "tiny".
      language (str, optional): _description_. Defaults to "en".
      fp16 (bool, optional): _description_. Defaults to False.
      verbose (bool, optional): _description_. Defaults to False.

  Returns:
      _type_: _description_
  """
  model = whisper.load_model(model_size)
  return model.transcribe(file, language=language, fp16=fp16, verbose=verbose)
  

def get_sentence_index(api_key: str, file_path: str) -> str:
  
  openai.api_key = api_key

  document = SimpleDirectoryReader(input_files=[file_path]).load_data()[0]

  node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=2,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
  )

  llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

  sentence_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=HuggingFaceEmbedding("BAAI/bge-small-en-v1.5"),
    node_parser=node_parser
  )

  return VectorStoreIndex.from_documents([document],
                                         service_context=sentence_context)

def query_sentence_window_rag(query, sentence_index, top_k=2):
  
  query_engine = sentence_index.as_query_engine(
    similarity_top_k=top_k,
    node_postprocessors=[
      MetadataReplacementPostProcessor(target_metadata_key="window")
    ]
  )
  
  window_response = query_engine.query(query)
  
  return window_response.response

def greet(query, url, api):
  path = "transcript.txt"

  tr = get_transcript(url)
  with open(path, "w") as file:
    file.write(tr["text"])

  sent_index = get_sentence_index(api_key=api, file_path=path)
  return query_sentence_window_rag(query=query, sentence_index=sent_index)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "text", "text"],
    outputs=["text"],
)

demo.launch()