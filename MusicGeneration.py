#Preprocessing Dataset for Melody Generation
import os
import music21 as m21

env = m21.environment.Environment()
env['musescoreDirectPNGPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
env['musicxmlPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'

KERN_DATASET_PATH = "/Users/sunny/Downloads/essen/europa/deutschl/test"
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
  # go throught all the files in dataset and load them with music21
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


def preprocess(dataset_path):
  pass

  # load the folk songs
  print("Loading Songs...")
  songs = load_songs_in_kern(dataset_path)
  print(f"Loaded{len(songs)} songs.")

  for song in songs:

    # filter out songs that have non-acceptable durations
    if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
      continue

  # transpose songs to Cmaj/min
  song = transpose(song)

  # encode songs with music time series representation

  # save songs to text file


if __name__ == "__main__":
  songs = load_songs_in_kern(KERN_DATASET_PATH)
  print(f"Loaded {len(songs)} songs.")
  song = songs[0]

  print(f"Has acceptable duration? {has_acceptable_durations(song, ACCEPTABLE_DURATIONS)}")

  transposed_song = transpose(song)

  song.show()
  input("Press Enter after closing MuseScore to continue...")
  transposed_song.show()






# hi