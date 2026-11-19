"""Таблицы языков: Whisper (распознавание) -> NLLB (перевод) -> XTTS (синтез)."""

# Языки, на которые можно озвучить (поддерживаются XTTS v2)
# name: (код whisper/общий, код NLLB, код XTTS)
TARGET_LANGUAGES = {
    "Русский": ("ru", "rus_Cyrl", "ru"),
    "English": ("en", "eng_Latn", "en"),
    "Deutsch": ("de", "deu_Latn", "de"),
    "Français": ("fr", "fra_Latn", "fr"),
    "Español": ("es", "spa_Latn", "es"),
    "Italiano": ("it", "ita_Latn", "it"),
    "Português": ("pt", "por_Latn", "pt"),
    "Polski": ("pl", "pol_Latn", "pl"),
    "Čeština": ("cs", "ces_Latn", "cs"),
    # XTTS v2 не знает словацкий: используем чешскую модель (языки очень близки), текст адаптируется в tts.prepare_text
    "Slovenčina": ("sk", "slk_Latn", "cs"),
    "Nederlands": ("nl", "nld_Latn", "nl"),
    "Türkçe": ("tr", "tur_Latn", "tr"),
    "Magyar": ("hu", "hun_Latn", "hu"),
    "العربية": ("ar", "arb_Arab", "ar"),
    "中文": ("zh", "zho_Hans", "zh-cn"),
    "日本語": ("ja", "jpn_Jpan", "ja"),
    "한국어": ("ko", "kor_Hang", "ko"),
    "हिन्दी": ("hi", "hin_Deva", "hi"),
}

# Языки исходного видео (whisper код -> NLLB код)
SOURCE_NLLB = {
    "en": "eng_Latn", "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "be": "bel_Cyrl", "de": "deu_Latn",
    "fr": "fra_Latn", "es": "spa_Latn", "it": "ita_Latn", "pt": "por_Latn", "pl": "pol_Latn",
    "cs": "ces_Latn", "sk": "slk_Latn", "nl": "nld_Latn", "tr": "tur_Latn", "hu": "hun_Latn",
    "ar": "arb_Arab", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang", "hi": "hin_Deva",
    "sv": "swe_Latn", "fi": "fin_Latn", "da": "dan_Latn", "no": "nob_Latn", "el": "ell_Grek",
    "he": "heb_Hebr", "ro": "ron_Latn", "bg": "bul_Cyrl", "sr": "srp_Cyrl", "hr": "hrv_Latn",
    "vi": "vie_Latn", "th": "tha_Thai", "id": "ind_Latn", "fa": "pes_Arab", "kk": "kaz_Cyrl",
}

SOURCE_CHOICES = {
    "Auto-detect": "auto",
    "English": "en", "Русский": "ru", "Українська": "uk", "Deutsch": "de", "Français": "fr",
    "Español": "es", "Italiano": "it", "Português": "pt", "Polski": "pl", "Čeština": "cs", "Slovenčina": "sk",
    "Nederlands": "nl", "Türkçe": "tr", "中文": "zh", "日本語": "ja", "한국어": "ko",
}
