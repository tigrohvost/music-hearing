from music_hearing import metadata


def test_parse_info_extracts_core_fields():
    info = {
        "title": "Organic Lullaby",
        "artist": "Meg Bowles",
        "album": "Blue Cosmos",
        "uploader": "Meg Bowles - Topic",
        "tags": ["ambient", "space music"],
        "categories": ["Music"],
        "description": "A long description " * 50,
        "webpage_url": "https://www.youtube.com/watch?v=EfaFcjpuwkg",
        "duration": 380,
    }
    m = metadata.parse_info(info)
    assert m["title"] == "Organic Lullaby"
    assert m["artist"] == "Meg Bowles"
    assert m["album"] == "Blue Cosmos"
    assert "ambient" in m["tags"]
    assert m["duration"] == 380
    # description is truncated
    assert len(m["description"]) <= 600


def test_parse_info_artist_falls_back_to_uploader():
    info = {"title": "x", "uploader": "Some Channel"}
    m = metadata.parse_info(info)
    assert m["artist"] == "Some Channel"


def test_parse_info_unwraps_playlist_entries():
    info = {"_type": "playlist", "entries": [
        {"title": "Real Track", "artist": "Real Artist", "tags": ["techno"]},
    ]}
    m = metadata.parse_info(info)
    assert m["title"] == "Real Track"
    assert m["artist"] == "Real Artist"
    assert "techno" in m["tags"]


def test_parse_info_handles_empty():
    m = metadata.parse_info({})
    assert m["title"] is None
    assert m["tags"] == []
