// Presentation-only English rendering for the FINAL AE report rows.
//
// Strictly a view-layer helper used by buildLiveSummary: it never runs during
// collection, conversation, speech, or workflow processing, and it never
// touches stored values. The reported event description is deliberately NOT
// handled here at all — that field is rendered byte-for-byte as captured.
//
// Deterministic and dependency-free (no LLM/translation service): Devanagari
// names are transliterated to Latin script preserving pronunciation, common
// yes/no and onset phrases are translated exactly, other Devanagari free text
// is glossed word-by-word (unknown words fall back to transliteration), and
// English/digit values pass through untouched.

const DEVANAGARI = /[\u0900-\u097F]/;
const MALAYALAM = /[\u0D00-\u0D7F]/;
const INDIC_NAME_SCRIPTS = /[\u0900-\u097F\u0D00-\u0D7F]/;

// Consonants (with the common retroflex/dental series mapped to ASCII).
const CONSONANTS: Record<string, string> = {
  "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
  "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
  "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
  "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
  "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
  "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
  "श": "sh", "ष": "sh", "स": "s", "ह": "h",
  "ऴ": "zh",
};

// Vowel signs (matras). The virama (्) is handled in the scan itself.
const VOWEL_SIGNS: Record<string, string> = {
  "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
  "ृ": "r", "ॄ": "r", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
};

// Independent vowels.
const VOWELS: Record<string, string> = {
  "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
  "ऋ": "r", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
};

// Modifiers: anusvara becomes "n"; the rest are silent.
const MODIFIERS: Record<string, string> = { "ं": "n", "ँ": "", "ः": "", "़": "" };

const DEVANAGARI_DIGITS: Record<string, string> = {
  "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
  "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
};

const VIRAMA = "्";

// Malayalam letters (same simplification convention as the Devanagari maps:
// ാ→a, ീ→i, ൂ→u, ോ→o, so "രാജു" renders "Raju", not "Raaju").
const MALAYALAM_CONSONANTS: Record<string, string> = {
  "ക": "k", "ഖ": "kh", "ഗ": "g", "ഘ": "gh", "ങ": "ng",
  "ച": "ch", "ഛ": "chh", "ജ": "j", "ഝ": "jh", "ഞ": "ny",
  "ട": "t", "ഠ": "t", "ഡ": "d", "ഢ": "d", "ണ": "n",
  "ത": "t", "ഥ": "th", "ദ": "d", "ധ": "dh", "ന": "n",
  "പ": "p", "ഫ": "ph", "ബ": "b", "ഭ": "bh", "മ": "m",
  "യ": "y", "ര": "r", "റ": "r", "ല": "l", "ള": "l",
  "ഴ": "l", "വ": "v", "ശ": "sh", "ഷ": "sh", "സ": "s", "ഹ": "h",
};

const MALAYALAM_VOWEL_SIGNS: Record<string, string> = {
  "ാ": "a", "ി": "i", "ീ": "i", "ു": "u", "ൂ": "u",
  "ൃ": "r", "െ": "e", "േ": "e", "ൈ": "ai", "ൊ": "o", "ോ": "o", "ൌ": "au",
};

const MALAYALAM_VOWELS: Record<string, string> = {
  "അ": "a", "ആ": "a", "ഇ": "i", "ഈ": "i", "ഉ": "u", "ഊ": "u",
  "ഋ": "r", "എ": "e", "ഏ": "e", "ഐ": "ai", "ഒ": "o", "ഓ": "o", "ഔ": "au",
};

// Chillu letters are word-final consonants WITHOUT an inherent vowel.
const MALAYALAM_CHILLU: Record<string, string> = {
  "ൺ": "n", "ൻ": "n", "ർ": "r", "ൽ": "l", "ൾ": "l",
};

const MALAYALAM_MODIFIERS: Record<string, string> = { "ം": "n", "ഃ": "" };

const MALAYALAM_DIGITS: Record<string, string> = {
  "൦": "0", "൧": "1", "൨": "2", "൩": "3", "൪": "4",
  "൫": "5", "൬": "6", "൭": "7", "൮": "8", "൯": "9",
};

const hasDevanagari = (text: string): boolean => DEVANAGARI.test(text);

const hasIndicNameScript = (text: string): boolean =>
  INDIC_NAME_SCRIPTS.test(text);

/** Upper-case the first character (no-op for digits/punctuation starts). */
export function capitalizeFirst(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// Transliterate one whitespace-delimited token. A consonant carries the
// inherent "a" unless a virama (no vowel) or an explicit matra follows; a
// word-final inherent "a" is dropped for Devanagari (standard schwa
// deletion: यश -> yash) and kept for Malayalam (different phonology:
// "രാമ" -> "Rama", never "Ram").
function transliterateToken(token: string): string {
  const isMalayalam = MALAYALAM.test(token);
  const consonants = isMalayalam ? MALAYALAM_CONSONANTS : CONSONANTS;
  const vowelSigns = isMalayalam ? MALAYALAM_VOWEL_SIGNS : VOWEL_SIGNS;
  const vowels = isMalayalam ? MALAYALAM_VOWELS : VOWELS;
  const modifiers = isMalayalam ? MALAYALAM_MODIFIERS : MODIFIERS;
  const digits = isMalayalam ? MALAYALAM_DIGITS : DEVANAGARI_DIGITS;
  const chillu = isMalayalam ? MALAYALAM_CHILLU : undefined;
  // The virama scripts differ: Devanagari ् (U+094D) vs Malayalam ് (U+0D4D).
  const virama = isMalayalam ? "്" : VIRAMA;
  const chars = Array.from(token);
  let out = "";
  let lastWasInherentA = false;
  for (let i = 0; i < chars.length; i += 1) {
    const char = chars[i];
    const next = chars[i + 1];
    lastWasInherentA = false;
    if (digits[char] !== undefined) {
      out += digits[char];
      continue;
    }
    const consonant = consonants[char];
    if (consonant !== undefined) {
      out += consonant;
      if (next === virama) {
        i += 1; // explicit suppression of the vowel
        continue;
      }
      if (next !== undefined && vowelSigns[next] !== undefined) {
        out += vowelSigns[next];
        i += 1;
        continue;
      }
      out += "a";
      lastWasInherentA = true;
      continue;
    }
    // A chillu is already vowel-free: no inherent "a" is ever added.
    const chilluValue = chillu?.[char];
    if (chilluValue !== undefined) {
      out += chilluValue;
      continue;
    }
    const sign = vowelSigns[char];
    if (sign !== undefined) {
      out += sign;
      continue;
    }
    const vowel = vowels[char];
    if (vowel !== undefined) {
      out += vowel;
      continue;
    }
    const modifier = modifiers[char];
    if (modifier !== undefined) {
      out += modifier;
      continue;
    }
    out += char; // Latin, punctuation, or unmapped script: unchanged
  }
  if (!isMalayalam && lastWasInherentA && out.endsWith("a")) {
    out = out.slice(0, -1);
  }
  return out;
}

// Exact phrase translations for the common reported onset sentences.
const ONSET_PHRASES: Record<string, string> = {
  "आज सुबह यह घटना हुई": "The event occurred this morning",
  "आज दोपहर यह घटना हुई": "The event occurred this afternoon",
  "आज शाम यह घटना हुई": "The event occurred this evening",
  "आज रात यह घटना हुई": "The event occurred tonight",
  "आज यह घटना हुई": "The event occurred today",
};

// Conservative word glosses (unambiguous words only; anything not listed is
// transliterated rather than guessed, so no meaning is silently invented).
const WORD_GLOSS: Record<string, string> = {
  "आज": "today", "सुबह": "morning", "रात": "night", "दोपहर": "afternoon",
  "शाम": "evening", "घटना": "event", "हुई": "occurred", "हुआ": "occurred",
  "हुए": "occurred", "यह": "this", "वह": "that", "दिन": "day", "समय": "time",
  "तारीख": "date", "बाद": "after", "पहले": "before", "दर्द": "pain",
  "सिर": "head", "दवा": "medicine", "दवाई": "medicine", "गोली": "tablet",
  "ली": "took", "लिया": "took", "थी": "was", "था": "was",
};

const YES_REPLIES = new Set(["हाँ", "हां", "जी हाँ", "जी हां", "haan", "ji haan"]);
const NO_REPLIES = new Set(["नहीं", "नही", "nahi", "nahin"]);

const stripTrailingPunctuation = (text: string): string =>
  text.replace(/[।.!?,"'’\s]+$/u, "");

/**
 * Transliterate an Indic-script (Devanagari or Malayalam) name to Latin
 * script, preserving pronunciation ("यश गुप्ता" -> "Yash Gupta",
 * "സുരേഷ് ഗോപി" -> "Suresh Gopi", one capitalised word per token). Values
 * with no supported Indic script are returned unchanged, byte for byte.
 */
export function englishName(value: string): string {
  const trimmed = value.trim();
  if (!hasIndicNameScript(trimmed)) {
    // Latin-script name: capitalize each word's first letter; the
    // remaining letters are never changed (McDonald keeps its internal
    // capitalization, Lantus SoloStar stays as captured) and the original
    // whitespace is preserved exactly.
    return trimmed
      .split(/(\s+)/)
      .map((part) => (/^[a-z]/.test(part) ? part.charAt(0).toUpperCase() + part.slice(1) : part))
      .join("");
  }
  return trimmed
    .split(/\s+/)
    .map((token) => {
      // A spoken English initial, as STT renders it in Malayalam script: a
      // two-codepoint letter name of exactly one consonant plus one vowel
      // sign ("പി" for P, "കെ" for K). Display only — the stored name is
      // never altered. Longer tokens are ordinary name syllables and always
      // transliterate fully; Devanagari tokens never take this rule.
      const chars = Array.from(token);
      if (
        MALAYALAM.test(token) &&
        chars.length === 2 &&
        MALAYALAM_CONSONANTS[chars[0]] !== undefined &&
        MALAYALAM_VOWEL_SIGNS[chars[1]] !== undefined
      ) {
        return MALAYALAM_CONSONANTS[chars[0]].toUpperCase();
      }
      const latin = transliterateToken(token);
      return latin.charAt(0).toUpperCase() + latin.slice(1);
    })
    .join(" ");
}

/**
 * Translate one non-event free-text report value to English:
 * exact yes/no -> Yes/No; exact onset phrase -> its English sentence;
 * otherwise a word-by-word gloss with transliteration for unknown words,
 * sentence-cased. Values with no Devanagari are returned unchanged.
 */
export function englishReportText(value: string): string {
  const trimmed = value.trim();
  if (!trimmed || !hasDevanagari(trimmed)) {
    return trimmed;
  }
  const key = stripTrailingPunctuation(trimmed);
  const lowered = key.toLowerCase();
  if (YES_REPLIES.has(lowered)) {
    return "Yes";
  }
  if (NO_REPLIES.has(lowered)) {
    return "No";
  }
  const phrase = ONSET_PHRASES[key];
  if (phrase) {
    return capitalizeFirst(phrase);
  }
  const words = key.split(/\s+/).map((word) => {
    if (!hasDevanagari(word)) {
      return word;
    }
    const trailing = word.match(/[।.!?,"'’]+$/u)?.[0] ?? "";
    const bare = stripTrailingPunctuation(word) || word;
    const gloss = WORD_GLOSS[bare];
    if (gloss) {
      return gloss + trailing;
    }
    return transliterateToken(bare) + trailing;
  });
  return capitalizeFirst(words.join(" "));
}

/**
 * Categorical report values (yes/no answers) in English display form.
 * Already-English values are returned unchanged.
 */
export function englishCategory(value: string): string {
  const trimmed = value.trim();
  const lowered = trimmed.toLowerCase();
  if (lowered === "yes") {
    return "Yes";
  }
  if (lowered === "no") {
    return "No";
  }
  return trimmed;
}
