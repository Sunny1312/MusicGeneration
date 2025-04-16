#Preprocessing Dataset for Melody Generation
import os
import json
import tensorflow.keras as keras
import music21 as m21
import numpy as np

env = m21.environment.Environment()
env['musescoreDirectPNGPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
env['musicxmlPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'

KERN_DATASET_PATH = "/Users/sunny/Downloads/essen/europa/deutschl/erk"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

#durations are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0
]

def load_songs_in_kern(dataset_path):
  # go through all the files in dataset and load them with music21
  # os walk recursively goes through all the files given a parent folder
  # path is referenced to the path of the current folder
  # subdirs - all the sub folders in the path, files are all the files in the path
  songs = []
  for path, subdirs, files in os.walk(dataset_path):
    for file in files:
      if file[-3:] == "krn": #filtering out the krn files
        song = m21.converter.parse(os.path.join(path, file)) #for loading the song
        # kern, MIDI, MusicXML -> m21 -> kern, MIDI,...
        # m21 enables your to represent music in a object oriented manner
        # song - music21 score
        songs.append(song)
    return songs

def has_acceptable_durations(song, acceptable_durations):
  for note in song.flat.notesAndRests:
    if note.duration.quarterLength not in acceptable_durations:
      return False
  return True

def transpose(song):

  # get key from the song
  parts = song.getElementsByClass(m21.stream.Part)
  measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
  key = measures_part0[0][4]

  # estimate key using music21
  if not isinstance(key, m21.key.Key):
    key = song.analyze("key")

  print(key)

  # get interval for transposition. Eg: Bmaj -> Cmaj
  if key.mode == "major":
    interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
  elif key.mode == "minor":
    interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

  # transpose song by calculated interval
  transposed_song = song.transpose(interval)
  return transposed_song

# this function gets the song as an input as the music21 object and gives back the string
def encode_song(song, time_step=0.25):
    """Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
    quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
    for representing notes/rests that are carried over into a new time step. Here's a sample encoding:

        ["r", "_", "60", "_", "_", "_", "72" "_"]

    :param song (m21 stream): Piece to encode
    :param time_step (float): Duration of each time step in quarter length
    :return:
    """

    encoded_song = []

    for event in song.flat.notesAndRests:

      # handle notes
      if isinstance(event, m21.note.Note):
        symbol = event.pitch.midi  # 60
      # handle rests
      elif isinstance(event, m21.note.Rest):
        symbol = "r"

      # convert the note/rest into time series notation
      steps = int(event.duration.quarterLength / time_step)
      for step in range(steps):

        # if it's the first time we see a note/rest, let's encode it. Otherwise, it means we're carrying the same
        # symbol in a new time step
        if step == 0:
          encoded_song.append(symbol)
        else:
          encoded_song.append("_")

    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path):

  # load the folk songs
  print("Loading Songs...")
  songs = load_songs_in_kern(dataset_path)
  print(f"Loaded{len(songs)} songs.")

  for i, song in enumerate(songs):

    # filter out songs that have non-acceptable durations
    if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
      continue

    # transpose songs to Cmaj/min
    song = transpose(song)

    # encode songs with music time series representation
    encoded_song = encode_song(song)

    # save songs to text file
    save_path = os.path.join(SAVE_DIR, str(i))
    with open(save_path, "w") as fp:
      fp.write(encoded_song)

def load(file_path):
  with open(file_path, "r") as fp:
    song = fp.read()
  return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
  new_song_delimiter = "/ " * sequence_length
  songs = ""

  # load encoded songs and add delimiters
  for path, _, files in os.walk(dataset_path):
    for file in files:
      file_path = os.path.join(path, file)
      song = load(file_path)
      songs = songs + song + " " + new_song_delimiter

  songs = songs[:-1]

  # save string that contains all dataset
  with open(file_dataset_path, "w") as fp:
    fp.write(songs)

  return songs

def create_mapping(songs, mapping_path):
  mappings =  {}

  # identify the vocabulary
  songs = songs.split()
  vocabulary = list(set(songs))

  # create mappings
  for i, symbol in enumerate(vocabulary):
    mappings[symbol] = i

  # save vocabulary to a json file
  with open(mapping_path, "w") as fp:
    json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):

  int_songs = []

  # load mappings
  with open(MAPPING_PATH, "r") as fp:
    mappings = json.load(fp)

  # cast songs string to a list
  songs = songs.split()

  # map songs to int
  for symbol in songs:
    int_songs.append(mappings[symbol])

  return int_songs

def generate_training_sequences(sequence_length):
    """Create input and output data samples for training. Each sample is a sequence.

    :param sequence_length (int): Length of each sequence. With a quantisation at 16th notes, 64 notes equates to 4 bars

    :return inputs (ndarray): Training inputs
    :return targets (ndarray): Training targets
    """

    # load songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    # generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
      inputs.append(int_songs[i:i + sequence_length])
      targets.append(int_songs[i + sequence_length])

    # one-hot encode the sequences
    vocabulary_size = len(set(int_songs))
    # inputs size: (# of sequences, sequence length, vocabulary size)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets

def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        """Converts a melody into a MIDI file

        :param melody (list of str):
        :param min_duration (float): Duration of each time step in quarter length
        :param file_name (str): Name of midi file
        :return:
        """

        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter  # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)

def main():
  preprocess(KERN_DATASET_PATH)
  songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
  create_mapping(songs, MAPPING_PATH)
  inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
  main()