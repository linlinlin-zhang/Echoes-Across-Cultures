(function () {
  const STORAGE_KEY = "echo-favorite-tracks";

  function read() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      return [];
    }
  }

  function write(items) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  }

  function trackKey(track) {
    return String(
      track?.track_id ||
      track?.trackId ||
      track?.id ||
      `${track?.title || ""}::${track?.artist || ""}::${track?.album || ""}`
    ).trim();
  }

  function normalize(track) {
    const raw = track?.raw || track || {};
    const id = trackKey(track) || trackKey(raw);
    return {
      id,
      track_id: raw.track_id || track.trackId || id,
      title: raw.title || track.title || "",
      artist: raw.artist || track.artist || "",
      album: raw.album || track.album || "",
      culture: raw.culture || track.culture || "",
      source_dataset: raw.source_dataset || track.sourceDataset || track.platform || "",
      label: raw.label || track.label || track.genre || "",
      country: raw.country || track.country || "",
      duration_ms: raw.duration_ms || (track.durationSeconds ? Math.round(Number(track.durationSeconds) * 1000) : 0),
      cover_art_url: raw.cover_art_url || track.coverArtUrl || "",
      cover_art_url_large: raw.cover_art_url_large || track.coverArtUrl || "",
      platform: raw.platform || track.platform || "",
      platform_track_url: raw.platform_track_url || track.platformTrackUrl || "",
      platform_album_url: raw.platform_album_url || "",
      full_track_url: raw.full_track_url || track.fullTrackUrl || "",
      preview_url: raw.preview_url || track.previewUrl || "",
      description: raw.description || track.description || track.aiDescription || "",
      album_description: raw.album_description || track.albumDescription || "",
      tags: raw.tags || track.tags || "",
      favorite_added_at: new Date().toISOString()
    };
  }

  function isFavorite(track) {
    const id = trackKey(track);
    if (!id) return false;
    return read().some((item) => trackKey(item) === id);
  }

  function add(track) {
    const next = normalize(track);
    if (!next.id) return read();
    const items = read().filter((item) => trackKey(item) !== next.id);
    items.unshift(next);
    write(items);
    window.dispatchEvent(new CustomEvent("echo-favorites-change", { detail: { items } }));
    return items;
  }

  function remove(track) {
    const id = trackKey(track);
    const items = read().filter((item) => trackKey(item) !== id);
    write(items);
    window.dispatchEvent(new CustomEvent("echo-favorites-change", { detail: { items } }));
    return items;
  }

  function toggle(track) {
    if (isFavorite(track)) {
      return { favorite: false, items: remove(track) };
    }
    return { favorite: true, items: add(track) };
  }

  window.EchoFavoriteStore = {
    read,
    write,
    normalize,
    trackKey,
    isFavorite,
    add,
    remove,
    toggle
  };
}());
