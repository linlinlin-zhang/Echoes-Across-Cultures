// Cross-check sourced points against the same country geometry used by the app.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const d3 = require('../web/vendor/d3.v7.min.js');
const topo = require('../web/vendor/topojson-client.min.js');
const atlas = require('../web/vendor/countries-50m.json');
const locations = require('../storage/public/merged/artist_locations.json');
const isoPath = process.env.ISO_CODES_JSON || '/usr/share/iso-codes/json/iso_3166-1.json';
const codes = Object.fromEntries(JSON.parse(fs.readFileSync(isoPath, 'utf8'))['3166-1'].map(r=>[r.alpha_2,r.numeric]));
const countries = new Map();
for (const feature of topo.feature(atlas, atlas.objects.countries).features) {
  const id = feature.properties.name === 'Kosovo' ? 'XK' : feature.id;
  // Some island groups share their country's ID. Keep every feature.
  if (!countries.has(id)) countries.set(id, []);
  countries.get(id).push(feature);
}
function vertices(coordinates, out = []) {
  if (typeof coordinates[0] === 'number') out.push(coordinates);
  else for (const nested of coordinates) vertices(nested, out);
  return out;
}
const memo = new Map(), strongOutliers = [], unmappedCountries = new Set();
let checked = 0;
for (const [key, location] of Object.entries(locations.artists)) {
  const code = location.country, point = [location.lng, location.lat];
  const geometryCode = ['RE', 'GP', 'MQ'].includes(code) ? 'FR' : code;
  const features = countries.get(code === 'XK' ? 'XK' : codes[geometryCode]);
  if (!features) { unmappedCountries.add(code); continue; }
  checked++;
  const signature = `${code}:${point.join(',')}`;
  if (!memo.has(signature)) {
    const inside = features.some(f=>d3.geoContains(f, point));
    const distance = inside ? 0 : Math.min(...features.flatMap(f=>vertices(f.geometry.coordinates)).map(v=>d3.geoDistance(v, point))) * 6371;
    memo.set(signature, distance);
  }
  if (memo.get(signature) > 200) strongOutliers.push({key, artist:location.artist, country:code, city:location.city,
    point, nearestBoundaryVertexKm:Math.round(memo.get(signature)), source:location.location_source_url});
}
const report = {map_resolution:'50m', checked, unique_positions:memo.size, unmapped_countries:[...unmappedCountries], strong_outliers:strongOutliers};
fs.mkdirSync(path.join(root, 'tmp/web-self-check'), {recursive:true});
fs.writeFileSync(path.join(root, 'tmp/web-self-check/geographic-consistency.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report));
assert.deepEqual(strongOutliers, [], 'Review points more than 200 km outside their sourced country');
