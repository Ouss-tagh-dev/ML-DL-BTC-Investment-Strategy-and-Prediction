export const loadJSON = (key, defaultValue) => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : defaultValue;
  } catch (e) {
    return defaultValue;
  }
};

export const saveJSON = (key, value) => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (e) {
    // noop
  }
};

export const toggleFavorite = (modelId) => {
  const favs = loadJSON("favs", []);
  const idx = favs.indexOf(modelId);
  if (idx === -1) {
    favs.push(modelId);
  } else {
    favs.splice(idx, 1);
  }
  saveJSON("favs", favs);
  return favs;
};

export const isFavorite = (modelId) => {
  const favs = loadJSON("favs", []);
  return favs.includes(modelId);
};

export const getTheme = () => loadJSON("theme", "dark");
export const setTheme = (t) => saveJSON("theme", t);
