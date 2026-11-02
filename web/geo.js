(function () {
  const data = window.EchoGeoData;
  const aliases = new Map();
  const languages = ["zh-CN", "en", "de", "es", "pt", "ja", "ko"];
  const normalized = (value) => String(value || "").trim().toLocaleLowerCase().replace(/[’']/g, "");
  const names = new Map();
  for (const language of languages) {
    try { names.set(language, new Intl.DisplayNames([language], { type: "region" })); } catch (_) {}
  }
  for (const [code, country] of Object.entries(data)) {
    for (const name of [code, country.alpha3, country.name, ...[...names.values()].map((n) => n.of(code))]) {
      if (name) aliases.set(normalized(name), code);
    }
  }
  for (const [name, code] of Object.entries({"台湾":"TW", "香港":"HK", "英国":"GB", "South Korea":"KR", "Russia":"RU", "USA":"US", "UK":"GB"})) {
    aliases.set(normalized(name), code);
  }
  function countryCode(value) { return aliases.get(normalized(value)) || ""; }
  function countryName(value, language = window.EchoI18n.getLanguage()) {
    const code = countryCode(value);
    return code ? (names.get(language)?.of(code) || data[code].name) : "";
  }
  function coordinate(value, limit) {
    if (value == null || typeof value === "boolean" || !String(value).trim()) return null;
    const number = Number(value);
    return Number.isFinite(number) && Math.abs(number) <= limit ? number : null;
  }
  function coordinates(item) {
    const lat = coordinate(item.lat ?? item.latitude, 90);
    const lng = coordinate(item.lng ?? item.lon ?? item.longitude, 180);
    return lat !== null && lng !== null ? { lat, lng } : null;
  }
  const cityKeys = "BEIJING TAIPEI BRASILIA BAMAKO LISBON JAKARTA ANKARA ISTANBUL TOKYO NEW_DELHI SEOUL STOCKHOLM WARSAW BELGRADE LIMA ASTANA KINGSTON LONDON SINGAPORE KUALA_LUMPUR BANGKOK HANOI MANILA MADRID BUENOS_AIRES SANTIAGO CANBERRA OTTAWA DUBLIN SALVADOR MEXICO_CITY ALMA_ATA CUSCO YOGYAKARTA".split(" ");
  const cities = new Map();
  for (const key of cityKeys) {
    for (const language of languages) {
      const name = window.EchoI18n.t("music.city." + key, {}, language);
      cities.set(normalized(name), key);
    }
  }
  function cityName(value, language = window.EchoI18n.getLanguage()) {
    const key = cities.get(normalized(value));
    if (key) return window.EchoI18n.t("music.city." + key, {}, language);
    const text = String(value || "").trim();
    return language !== "zh-CN" && /[\u3400-\u9fff]/.test(text) ? "" : text;
  }
  function resolve(item, cultureReference) {
    const rawCountry = item.country_code || item.countryCode || item.country;
    const code = countryCode(rawCountry);
    const point = coordinates(item);
    const source = String(item.source_dataset || item.platform || "").toLowerCase();
    const precision = item.location_precision || "";
    if (precision === "unknown" || precision === "pacific_default" || rawCountry === "mystery_place") {
      return { countryCode: "", city: "", lat: 0, lng: -160, precision: "unknown" };
    }
    if (point) {
      return { ...point, countryCode: code, city: item.city || item.location_city || item.artist_city || "",
        cityLabels: item.city_labels || {},
        precision: ["artist_city", "country_capital", "catalog_market"].includes(precision) ? precision : "coordinates" };
    }
    // A store market or a musical culture is not evidence of an artist's location.
    if (["itunes", "jamendo"].includes(source)) {
      return { countryCode: item.country_source === "artist_location" ? code : "", city: "", lat: 0, lng: -160, precision: "unknown" };
    }
    if (code && data[code].point) {
      const [lng, lat] = data[code].point;
      return { countryCode: code, city: "", lat, lng, precision: source === "itunes" ? "catalog_market" : "country_region" };
    }
    if (cultureReference && source !== "upload") {
      return { countryCode: "", city: "", lat: cultureReference.lat, lng: cultureReference.lng, precision: "culture_region" };
    }
    return { countryCode: "", city: "", lat: 0, lng: -160, precision: "unknown" };
  }
  window.EchoGeo = { countryCode, countryName, cityName, coordinate, coordinates, resolve };
}());
