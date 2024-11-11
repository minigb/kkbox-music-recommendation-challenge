import re
import pandas as pd


def get_song_id(song, artist):
    song = song.replace('/', ' ')
    artist = artist.replace('/', ' ')
    return f'{{{song}}}_{{{artist}}}'


def decode_song_id(song_id):
    # Note that if there was a '/' in the song or artist, it is replaced with a space,
    # so the data will not be found in the csv.
    pattern = r"\{([^}]*)\}"
    matches = re.findall(pattern, song_id)
    assert len(matches) == 2, f"song_id should have 2 parts, but got {len(matches)} for {song_id}"
    return matches # song, artist