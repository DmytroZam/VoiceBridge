"""Синтез речи: Silero v5 (чистый русский без акцента) и Coqui XTTS v2."""
from __future__ import annotations

import html
import os
import re
import textwrap

import numpy as np

from .utils import free_gpu, trim_silence

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SR = 24000

SILERO_RU_URL = "https://models.silero.ai/models/tts/ru/v5_6_ru.pt"
SILERO_RU_FALLBACK_URL = "https://models.silero.ai/models/tts/ru/v5_ru.pt"
SILERO_SR = 48000

# Верхние границы частоты основного тона (Гц) для выбора встроенного голоса Silero v5_6_ru:
# eugene/aidar — чёткая мужская артикуляция (вплоть до высоких мужских теноров ~188 Гц),
# baya/kseniya — чёткая женская артикуляция без придыхательного смягчения «р».
_SILERO_F0_BANDS: tuple[tuple[float, str], ...] = (
    (105.0, "eugene"),
    (188.0, "aidar"),
    (228.0, "baya"),
)

# Проверенные встроенные дикторы XTTS v2 для устранения иностранного акцента при кросс-язычном синтезе:
# (мужской_голос, женский_голос) — их speaker_embedding при этом не используется (берётся тембр оригинала).
_XTTS_NATIVE_ANCHORS: dict[str, tuple[str, str]] = {
    "ru": ("Viktor Eka", "Tanja Adelina"),
    "en": ("Damien Black", "Claribel Dervla"),
    "de": ("Aaron Dreschner", "Claribel Dervla"),
    "es": ("Marcos Rudaski", "Ana Florence"),
    "fr": ("Zacharie Aimilios", "Claribel Dervla"),
    "pl": ("Filip Traverse", "Tanja Adelina"),
    "cs": ("Ludvig Milivoj", "Tanja Adelina"),
}

# XTTS v2 не поддерживает словацкий. Читаем словацкий текст чешской моделью, заменяя буквы,
# которых нет в чешском, на близкие по звучанию (ô -> uo, ä -> e, ľ/ĺ -> l, ŕ -> r).
_SK_SYLLABLES = (
    ("ďa", "dja"), ("Ďa", "Dja"), ("ďo", "djo"), ("Ďo", "Djo"), ("ďu", "dju"), ("Ďu", "Dju"), ("ďe", "dě"), ("Ďe", "Dě"),
    ("ťa", "tja"), ("Ťa", "Tja"), ("ťo", "tjo"), ("Ťo", "Tjo"), ("ťu", "tju"), ("Ťu", "Tju"), ("ťe", "tě"), ("Ťe", "Tě"),
    ("ňa", "nja"), ("Ňa", "Nja"), ("ňo", "njo"), ("Ňo", "Njo"), ("ňu", "nju"), ("Ňu", "Nju"), ("ňe", "ně"), ("Ňe", "Ně"),
    ("pä", "pě"), ("Pä", "Pě"),
    ("ia", "ja"), ("Ia", "Ja"), ("ie", "je"), ("Ie", "Je"), ("iu", "ju"), ("Iu", "Ju"),
)
_SK_TO_CS = str.maketrans({"ô": "uo", "Ô": "Uo", "ä": "e", "Ä": "E", "ľ": "l", "Ľ": "L",
                           "ĺ": "l", "Ĺ": "L", "ŕ": "r", "Ŕ": "R", "ť": "t", "Ť": "T", "ď": "d", "Ď": "D"})
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")

_RU_ORD_RE = re.compile(
    r"\b(\d+)-(й|ый|ий|ой|я|ая|ья|е|ое|ье|го|ого|его|му|ому|ему|м|ым|им|ом|ем|ые|ие|ых|их|ыми|ими)\b",
    re.IGNORECASE,
)
_RU_LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи", "h": "эйч",
    "i": "ай", "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн", "o": "оу", "p": "пи",
    "q": "кью", "r": "ар", "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл ю", "x": "икс",
    "y": "уай", "z": "зет",
}
_RU_LATIN_WORDS = {
    "ai": "эй ай", "gps": "джи пи эс", "usb": "ю эс би", "wifi": "вай фай", "wi": "вай", "fi": "фай",
    "youtube": "ютьюб", "tiktok": "тикток", "google": "гугл", "iphone": "айфон", "ipad": "айпад",
    "cheerio": "чирио", "cheerios": "чириос", "stealth": "стелс", "ok": "окей",
}
_RU_LATIN_RULES: tuple[tuple[str, str], ...] = (
    ("shch", "щ"), ("sch", "ш"), ("tion", "шн"), ("sion", "жн"), ("ough", "о"), ("ight", "айт"),
    ("ch", "ч"), ("sh", "ш"), ("zh", "ж"), ("kh", "х"), ("th", "з"), ("ph", "ф"), ("qu", "кв"),
    ("ck", "к"), ("ng", "нг"), ("ee", "и"), ("oo", "у"), ("ea", "и"), ("ou", "ау"), ("ow", "оу"),
    ("au", "о"), ("aw", "о"), ("ai", "эй"), ("ay", "эй"), ("ey", "эй"), ("oi", "ой"), ("oy", "ой"),
    ("ts", "ц"), ("ya", "я"), ("yu", "ю"), ("yo", "ё"),
)
_RU_LATIN_CHAR = str.maketrans({
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
    "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
    "y": "и", "z": "з",
})
_RU_CYR_RE = re.compile(r"[а-яА-ЯёЁ]")


def _expand_sk_numbers(text: str) -> str:
    """Раскрывает числа по-словацки до передачи в чешский токенизатор XTTS (иначе '35' читается как чешское 'třicet pět')."""
    try:
        from num2words import num2words
    except ImportError:
        return text

    def _repl(m: re.Match) -> str:
        raw = m.group(0)
        try:
            if "." in raw or "," in raw:
                return num2words(float(raw.replace(",", ".")), lang="sk")
            return num2words(int(raw), lang="sk")
        except Exception:
            return raw

    text = re.sub(r"(?<=[A-Za-z])-(?=\d)", " ", text)
    return _NUM_RE.sub(_repl, text)


def _inflect_ru_ordinal(base: str, suffix: str) -> str:
    s = suffix.lower()
    if s in ("й", "ый", "ий", "ой"):
        return base
    is_tret = base.endswith("третий")
    stem = base[:-2]
    if is_tret:
        table = {
            "я": "ья", "ья": "ья", "е": "ье", "ье": "ье",
            "го": "ьего", "его": "ьего", "му": "ьему", "ему": "ьему",
            "м": "ьим", "им": "ьим", "ом": "ьем", "ем": "ьем",
            "ие": "ьи", "ые": "ьи", "их": "ьих", "ых": "ьих",
            "ими": "ьими", "ыми": "ьими",
        }
        return stem + table.get(s, "ий")
    soft = base.endswith("ий")
    table = {
        "я": "ая", "ая": "ая", "е": "ое", "ое": "ое",
        "го": "его" if soft else "ого", "ого": "ого", "его": "его",
        "му": "ему" if soft else "ому", "ому": "ому", "ему": "ему",
        "м": "им" if soft else "ым", "ым": "ым", "им": "им",
        "ом": "ом", "ем": "ем",
        "ые": "ые", "ие": "ие", "ых": "ых", "их": "их",
        "ыми": "ыми", "ими": "ими",
    }
    return stem + table.get(s, base[-2:])


def _translit_en_word(word: str) -> str:
    lw = word.lower()
    if lw in _RU_LATIN_WORDS:
        return _RU_LATIN_WORDS[lw]
    if word.isupper() and len(word) <= 4:
        return " ".join(_RU_LETTER_NAMES.get(c, c) for c in lw)
    s = re.sub(r"c(?=[eiy])", "s", lw)
    s = re.sub(r"(?<=[^aeiouy])e$", "", s)
    for src, dst in _RU_LATIN_RULES:
        s = s.replace(src, dst)
    return s.translate(_RU_LATIN_CHAR)


def _prepare_ru_text(text: str) -> str:
    """Раскрывает числа, порядковые, символы и латиницу для чистого русского синтеза без выпадений и акцента."""
    # Простой знак '+' (не перед ударной гласной внутри русского слова) заменяем на 'плюс'
    text = re.sub(r"(?<![а-яА-ЯёЁ])\+(?![аеёиоуыэюяАЕЁИОУЫЭЮЯ])", " плюс ", text)
    text = re.sub(r"°\s*[CcСс]", " градусов Цельсия ", text)
    text = text.replace("°", " градусов ").replace("%", " процентов ").replace("&", " и ").replace("№", " номер ")
    text = re.sub(r"\$\s*(\d+(?:[.,]\d+)?)", r"\1 долларов ", text)
    text = re.sub(r"€\s*(\d+(?:[.,]\d+)?)", r"\1 евро ", text)
    text = re.sub(r"₽\s*(\d+(?:[.,]\d+)?)", r"\1 рублей ", text)
    text = text.replace("$", " долларов ").replace("€", " евро ").replace("₽", " рублей ")
    text = re.sub(r"(?<=[A-Za-zА-Яа-яЁё])-(?=\d)", " ", text)

    try:
        from num2words import num2words

        def _ord_repl(m: re.Match) -> str:
            n_str, suf = m.group(1), m.group(2)
            try:
                base = num2words(int(n_str), to="ordinal", lang="ru")
                return _inflect_ru_ordinal(base, suf)
            except Exception:
                return n_str

        text = _RU_ORD_RE.sub(_ord_repl, text)
        text = re.sub(r"(\d+)-(?=[А-Яа-яЁё]{2,})", r"\1 ", text)
        text = re.sub(r"(?<![^\s(\[,])-(?=\d)", "минус ", text)

        def _num_repl(m: re.Match) -> str:
            raw = m.group(0)
            try:
                if "." in raw or "," in raw:
                    return num2words(float(raw.replace(",", ".")), lang="ru")
                return num2words(int(raw), lang="ru")
            except Exception:
                return raw

        text = _NUM_RE.sub(_num_repl, text)
    except ImportError:
        pass

    text = re.sub(r"[A-Za-z]+", lambda m: _translit_en_word(m.group(0)), text)
    text = re.sub(r'[«»„“"]', " ", text).replace(" — ", ", ").replace("—", ", ").replace("–", "-")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_SK_ABBREV = {
    "wi-fi": "vifi", "wifi": "vifi", "gps": "dží pí es", "ai": "ej aj", "usb": "jú es bí",
    "stealth": "stels", "router": "rauter", "youtube": "jutub", "tiktok": "tiktok", "ok": "ókej",
}
_SK_LETTER_NAMES = {
    "a": "á", "b": "bé", "c": "cé", "d": "dé", "e": "é", "f": "ef", "g": "gé", "h": "há",
    "i": "í", "j": "jé", "k": "ká", "l": "el", "m": "em", "n": "en", "o": "ó", "p": "pé",
    "q": "kvé", "r": "er", "s": "es", "t": "té", "u": "ú", "v": "vé", "w": "dvojité vé",
    "x": "iks", "y": "ypsilon", "z": "zet",
}


def _prepare_sk_native(text: str) -> str:
    """Нормализует словацкий текст для нативной модели VITS без замены родных букв (ô, ä, ľ, ĺ, ŕ, ť, ď, ň)."""
    text = re.sub(r"°\s*[Cc]", " stupňov Celzia ", text)
    text = text.replace("°", " stupňov ")
    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*%", r"\1 percent", text)
    text = re.sub(r"\$\s*(\d+(?:[.,]\d+)?)", r"\1 dolárov", text)
    text = re.sub(r"€\s*(\d+(?:[.,]\d+)?)", r"\1 eur", text)
    text = re.sub(r"\bWi-?Fi\b", "vifi", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([A-Za-z]{2,10})\b",
                  lambda m: _SK_ABBREV.get(m.group(1).lower(),
                                           " ".join(_SK_LETTER_NAMES.get(c, c) for c in m.group(1).lower())
                                           if (m.group(1).isupper() and 2 <= len(m.group(1)) <= 3) else m.group(1)),
                  text)
    text = re.sub(r"\b([A-Za-z])-?(\d+)\b",
                  lambda m: f"{_SK_LETTER_NAMES.get(m.group(1).lower(), m.group(1))} {m.group(2)}", text)
    text = _expand_sk_numbers(text)
    text = re.sub(r'[«»„“"]', " ", text).replace("…", ".").replace(" — ", ", ").replace("—", ", ").replace("–", "-")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_text(text: str, lang_code: str, native: bool = False) -> str:
    if lang_code == "sk":
        text = _prepare_sk_native(text)
        if native:
            return text
        for src, dst in _SK_SYLLABLES:
            text = text.replace(src, dst)
        return text.translate(_SK_TO_CS)
    if lang_code == "ru":
        return _prepare_ru_text(text)
    return text


# Лимиты символов на один прогон XTTS (с запасом); по умолчанию 200
_CHAR_LIMITS = {"ja": 60, "zh-cn": 70, "ko": 80, "ar": 150, "cs": 170, "hi": 150, "tr": 190, "ru": 180}
_SENT_SPLIT = re.compile(r"(?<=[.!?…。！？])\s*")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:、，；：])\s*")


def split_text(text: str, language: str, limit: int | None = None) -> list[str]:
    """Делит текст на куски, не превышающие лимит синтезатора: сначала по предложениям, потом по запятым, потом по словам."""
    limit = limit or _CHAR_LIMITS.get(language, 200)
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    units: list[str] = []
    for sent in filter(None, (s.strip() for s in _SENT_SPLIT.split(text))):
        if len(sent) <= limit:
            units.append(sent)
            continue
        for clause in filter(None, (c.strip() for c in _CLAUSE_SPLIT.split(sent))):
            if len(clause) <= limit:
                units.append(clause)
            else:
                units.extend(textwrap.wrap(clause, limit, break_long_words=True, break_on_hyphens=False)
                             if " " in clause else [clause[i:i + limit] for i in range(0, len(clause), limit)])
    # склеиваем короткие соседние предложения — быстрее и ближе к естественной речи
    merged: list[str] = []
    for u in units:
        sep = "" if language in ("ja", "zh-cn") else " "
        if merged and len(merged[-1]) + len(sep) + len(u) <= limit:
            merged[-1] += sep + u
        else:
            merged.append(u)
    return merged or [text]


def _patch_xtts_tokenizer() -> None:
    """Исправляет баг Coqui XTTS + num2words, где порядковые числительные (например, '1.' в чешском/словацком)
    вызывают NotImplementedError в num2words.lang_CS.to_ordinal.
    """
    try:
        from num2words import num2words
        from TTS.tts.layers.xtts import tokenizer as xtts_tok

        def _safe_expand_ordinal(m, lang="en"):
            n = int(m.group(1))
            try:
                return num2words(n, ordinal=True, lang=lang)
            except Exception:
                try:
                    word = num2words(n, lang=lang)
                    return word + ("." if m.group(0).endswith(".") else "")
                except Exception:
                    return m.group(0)

        def _safe_expand_decimal_point(m, lang="en"):
            amount = m.group(1).replace(",", ".")
            try:
                return num2words(float(amount), lang=lang)
            except Exception:
                return m.group(0)

        def _safe_expand_number(m, lang="en"):
            try:
                return num2words(int(m.group(0)), lang=lang)
            except Exception:
                return m.group(0)

        xtts_tok._expand_ordinal = _safe_expand_ordinal
        xtts_tok._expand_decimal_point = _safe_expand_decimal_point
        xtts_tok._expand_number = _safe_expand_number
    except Exception:
        pass


class VoiceCloner:
    def __init__(self, device: str = "cuda"):
        os.environ.setdefault("COQUI_TOS_AGREED", "1")  # согласие с лицензией CPML (некоммерческое использование)
        import torch
        from TTS.api import TTS

        _patch_xtts_tokenizer()
        self.device = device
        # Загружаем на CPU и переводим 1.76 ГБ GPT в BF16 (на GPU с аппаратным BF16) —
        # пик VRAM падает с ~2.0 ГБ до ~1.1 ГБ без переполнения логитов, характерного для FP16.
        self._amp_dtype = (
            torch.bfloat16 if (device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else None
        )
        self.tts = TTS(XTTS_MODEL, progress_bar=False)
        self.model = self.tts.synthesizer.tts_model
        if self._amp_dtype is not None:
            self.model.gpt.to(self._amp_dtype)
        self.tts.to(device)
        free_gpu()
        self.sr = XTTS_SR

    def _amp_ctx(self):
        from contextlib import nullcontext
        import torch

        return torch.amp.autocast("cuda", dtype=self._amp_dtype) if self._amp_dtype is not None else nullcontext()

    def voice_from(self, wav_paths: list[str], gpt_cond_len: int = 16, gpt_cond_chunk_len: int | None = None):
        """Вычисляет «отпечаток» голоса по одному или нескольким образцам."""
        import torch
        import torch.nn.functional as F

        # Perceiver Resampler в XTTS v2 обучен на окнах по 4–6 секунд (см. config.json: gpt_cond_chunk_len=4).
        # Для длинного эталона используем окна по 4 с, а для коротких фраз (<= 8 с) — цельный проход без нарезки.
        chunk_len = gpt_cond_chunk_len or (4 if gpt_cond_len >= 12 else gpt_cond_len)
        with self._amp_ctx():
            gpt_latent, speaker_emb = self.model.get_conditioning_latents(
                audio_path=wav_paths,
                gpt_cond_len=gpt_cond_len,
                gpt_cond_chunk_len=chunk_len,
                max_ref_length=60,
                sound_norm_refs=True,
            )
        with torch.no_grad():
            speaker_emb = F.normalize(speaker_emb.float(), dim=1)
        return gpt_latent, speaker_emb

    @staticmethod
    def combine_voice(global_voice, seg_voice, seg_emb_weight: float = 0.25, keep_global_gpt: bool = False):
        """Совмещает интонационный слепок (gpt_latent) и стабильный тембр (speaker_emb) спикера."""
        import torch
        import torch.nn.functional as F

        global_gpt, global_emb = global_voice
        seg_gpt, seg_emb = seg_voice
        with torch.no_grad():
            w = float(np.clip(seg_emb_weight, 0.0, 1.0))
            mixed_emb = F.normalize((1.0 - w) * global_emb + w * seg_emb, dim=1)
        return (global_gpt if keep_global_gpt else seg_gpt), mixed_emb

    def native_gpt_latent(self, xtts_lang: str, lang_code: str = "", target_f0: float | None = None):
        """Возвращает языко-чистый gpt_cond_latent носителя целевого языка, чтобы убрать акцент оригинала."""
        is_female = target_f0 is not None and target_f0 >= 188.0
        # Для русского генерируем короткий чистый эталон через Silero v5 (на CPU без расхода VRAM),
        # а при отсутствии сети используем встроенный профиль диктора.
        if (lang_code == "ru" or xtts_lang == "ru"):
            try:
                import tempfile
                import soundfile as sf

                sil = RussianSileroTTS(device="cpu")
                spk = sil.pick_speaker(target_f0)
                anchor_wav = sil.speak(
                    "Добрый день. Сегодня прекрасная погода, и мы обсудим важные новости и проекты.",
                    speaker=spk,
                    speed=1.0,
                )
                sil.close()
                if anchor_wav.size >= self.sr:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tmp_path = tf.name
                    try:
                        sf.write(tmp_path, anchor_wav, self.sr, subtype="PCM_16")
                        gpt_lat, _ = self.voice_from([tmp_path], gpt_cond_len=8, gpt_cond_chunk_len=4)
                        return gpt_lat
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
            except Exception:
                pass

        pair = _XTTS_NATIVE_ANCHORS.get(xtts_lang)
        if pair is None:
            return None
        speakers = getattr(getattr(self.model, "speaker_manager", None), "speakers", None) or {}
        name = pair[1] if is_female else pair[0]
        entry = speakers.get(name) or speakers.get(pair[0]) or speakers.get(pair[1])
        if entry and "gpt_cond_latent" in entry:
            return entry["gpt_cond_latent"].clone().to(dtype=self.model.gpt.text_embedding.weight.dtype)
        return None

    def _infer_chunk(self, chunk: str, language: str, gpt_latent, speaker_emb,
                     temperature: float, speed: float) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        lang_base = language.split("-")[0]
        # Убираем финальную точку перед EOS: в Coqui XTTS v2 конечный токен '.' на коротких фразах
        # провоцирует 1–2 лишних галлюцинаторных слога в конце (особенно в cs/sk); '?' и '!' сохраняем.
        sent = re.sub(r"\.+$", "", chunk.strip().lower()).strip() or chunk.strip().lower()
        toks = self.model.tokenizer.encode(sent, lang=lang_base)
        max_toks = int(self.model.args.gpt_max_text_tokens)
        if len(toks) >= max_toks:
            toks = toks[: max_toks - 1]
        text_tokens = torch.IntTensor(toks).unsqueeze(0).to(self.device)
        with torch.no_grad():
            gpt_cond = gpt_latent.to(device=self.device, dtype=self.model.gpt.text_embedding.weight.dtype)
            spk_emb = speaker_emb.to(device=self.device, dtype=torch.float32)
            with self._amp_ctx():
                eff_temp = min(temperature, 0.55) if lang_base == "cs" else temperature
                eff_top_p = 0.78 if lang_base == "cs" else 0.85
                eff_top_k = 30 if lang_base == "cs" else 50
                eff_rep = 10.0 if lang_base == "cs" else 5.0
                gpt_codes = self.model.gpt.generate(
                    cond_latents=gpt_cond,
                    text_inputs=text_tokens,
                    do_sample=True,
                    top_p=eff_top_p,
                    top_k=eff_top_k,
                    temperature=eff_temp,
                    num_return_sequences=self.model.gpt_batch_size,
                    num_beams=1,
                    length_penalty=1.0,
                    repetition_penalty=eff_rep,
                    output_attentions=False,
                )
                expected_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.model.gpt.code_stride_len], device=text_tokens.device
                )
                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                gpt_latents = self.model.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_len,
                    cond_latents=gpt_cond,
                    return_attentions=False,
                    return_latent=True,
                )
            gpt_latents = gpt_latents.float()
            length_scale = 1.0 / max(speed, 0.05)
            if abs(length_scale - 1.0) > 1e-3:
                gpt_latents = F.interpolate(
                    gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                ).transpose(1, 2)
            wav = self.model.hifigan_decoder(gpt_latents, g=spk_emb).cpu().squeeze().numpy()
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    def speak(self, text: str, language: str, voice, temperature: float = 0.75, speed: float = 1.0) -> np.ndarray:
        gpt_latent, speaker_emb = voice
        chunks = split_text(text, language)
        if not chunks:
            return np.zeros(0, dtype=np.float32)

        pieces = []
        for i, chunk in enumerate(chunks):
            wav = self._infer_chunk(chunk, language, gpt_latent, speaker_emb, temperature=temperature, speed=speed)
            pieces.append(wav)
            if i < len(chunks) - 1:
                pieces.append(np.zeros(int(0.05 * self.sr), dtype=np.float32))

        return np.concatenate(pieces)

    def close(self):
        del self.model, self.tts
        free_gpu()


def _download_silero_ru() -> str:
    """Скачивает и кэширует модель Silero TTS v5 для чистого русского произношения."""
    import torch

    cache_dir = os.path.join(torch.hub.get_dir(), "silero")
    os.makedirs(cache_dir, exist_ok=True)
    ckpt = os.path.join(cache_dir, "v5_6_ru.pt")
    if os.path.isfile(ckpt) and os.path.getsize(ckpt) > 10_000_000:
        return ckpt
    tmp = ckpt + ".tmp"
    last_err: Exception | None = None
    for url in (SILERO_RU_URL, SILERO_RU_FALLBACK_URL):
        try:
            torch.hub.download_url_to_file(url, tmp, progress=True)
            os.replace(tmp, ckpt)
            return ckpt
        except Exception as e:
            last_err = e
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    raise RuntimeError(f"Failed to download Silero Russian TTS model: {last_err}")


class RussianSileroTTS:
    """Чистый нейросетевой синтез русской речи (Silero v5) с автоударениями и без иностранного акцента."""

    def __init__(self, device: str = "cuda"):
        import torch

        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.sr = XTTS_SR
        ckpt = _download_silero_ru()
        try:
            self.model = torch.package.PackageImporter(ckpt).load_pickle("tts_models", "model")
        except Exception:
            if os.path.exists(ckpt):
                os.remove(ckpt)
            ckpt = _download_silero_ru()
            self.model = torch.package.PackageImporter(ckpt).load_pickle("tts_models", "model")
        self.model.to(self.device)

    @staticmethod
    def pick_speaker(target_f0: float | None) -> str:
        """Подбирает встроенный голос Silero, наиболее близкий по полу и регистру к оригиналу."""
        if target_f0 is None or target_f0 <= 0:
            return "aidar"
        for upper, name in _SILERO_F0_BANDS:
            if target_f0 < upper:
                return name
        return "kseniya"

    def _infer_chunk(self, chunk: str, speaker: str, speed: float) -> np.ndarray:
        import soxr

        if not _RU_CYR_RE.search(chunk):
            return np.zeros(0, dtype=np.float32)
        kwargs = {
            "speaker": speaker,
            "sample_rate": SILERO_SR,
            "put_accent": True,
            "put_yo": True,
            "put_stress_homo": True,
            "put_yo_homo": True,
        }
        if speed > 1.02 or speed < 0.98:
            rate_pct = int(round(float(np.clip(speed, 0.75, 1.35)) * 100))
            ssml = f'<speak><prosody rate="{rate_pct}%">{html.escape(chunk, quote=False)}</prosody></speak>'
            audio = self.model.apply_tts(ssml_text=ssml, **kwargs)
        else:
            audio = self.model.apply_tts(text=chunk, **kwargs)

        wav_48 = audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
        wav_24 = soxr.resample(wav_48, SILERO_SR, self.sr, quality="HQ").astype(np.float32)
        return trim_silence(wav_24, self.sr, threshold=0.008, pad=0.03)

    def speak(self, text: str, speaker: str = "aidar", speed: float = 1.0) -> np.ndarray:
        clean = _prepare_ru_text(text)
        chunks = split_text(clean, "ru", limit=280)
        if not chunks:
            return np.zeros(0, dtype=np.float32)

        pieces: list[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            wav = self._infer_chunk(chunk, speaker=speaker, speed=speed)
            if wav.size == 0:
                continue
            pieces.append(wav)
            if i < len(chunks) - 1:
                pieces.append(np.zeros(int(0.05 * self.sr), dtype=np.float32))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def close(self):
        if hasattr(self, "model"):
            del self.model
        free_gpu()


def _tts_worker_main(manifest_path: str, out_path: str) -> None:
    """Точка входа изолированного подпроцесса TTS: синтезирует все сегменты и гарантированно освобождает CUDA."""
    import faulthandler
    import json
    from .utils import compact_silence, load_wav, normalize_audio, save_wav, setup_cuda_dlls

    try:
        faulthandler.enable()
    except Exception:
        pass
    setup_cuda_dlls()
    with open(manifest_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = cfg["device"]
    workdir = cfg["workdir"]
    seg_dir = os.path.join(workdir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    sr = int(cfg.get("sr", XTTS_SR))
    lang_code = cfg["lang_code"]
    xtts_lang = cfg["xtts_lang"]
    src_lang = cfg.get("src_lang", "")
    deep_voice_clone = bool(cfg.get("deep_voice_clone", True))
    global_f0 = cfg.get("global_f0")
    expr = float(cfg.get("expressiveness", 0.8))
    per_segment_voice = bool(cfg.get("per_segment_voice", False))
    temperature = float(cfg.get("temperature", 0.75))
    max_speed = float(cfg.get("max_speed", 1.4))
    duration = float(cfg["duration"])
    ref_paths = list(cfg["ref_paths"])
    best_ref_clip = load_wav(cfg["best_ref_path"], sr)
    voice_audio = load_wav(cfg["vocals"], sr)
    segs = list(cfg["segments"])

    logs: list[str] = []
    silero_ru: RussianSileroTTS | None = None
    cloner: VoiceCloner | None = None

    def _emit_prog(frac: float, msg: str):
        print(f"__PROG__:{frac:.4f}:{msg}", flush=True)

    if lang_code == "ru" and deep_voice_clone:
        try:
            logs.append("Loading Silero TTS v5 (accent-free native Russian speech)…")
            _emit_prog(0.0, "Loading Silero TTS v5…")
            silero_ru = RussianSileroTTS(device)
        except Exception as e:
            logs.append(f"Silero TTS unavailable ({e}); falling back to XTTS v2…")
            silero_ru = None

    global_voice = None
    native_gpt_cache: dict[bool, object] = {}
    cross_lingual = (not src_lang) or (src_lang != lang_code)
    global_silero_spk = "aidar"

    if silero_ru is not None:
        global_silero_spk = silero_ru.pick_speaker(global_f0)
        logs.append(f"Native Russian speaker matched by pitch: {global_silero_spk}")
    else:
        logs.append("Loading XTTS v2 (speech synthesis)…")
        _emit_prog(0.0, "Loading XTTS v2…")
        cloner = VoiceCloner(device)
        global_voice = cloner.voice_from(ref_paths, gpt_cond_len=16, gpt_cond_chunk_len=4)
        if cross_lingual:
            is_fem_global = global_f0 is not None and global_f0 >= 188.0
            ng = cloner.native_gpt_latent(xtts_lang, lang_code=lang_code, target_f0=global_f0)
            if ng is not None:
                native_gpt_cache[is_fem_global] = ng
                global_voice = (ng, global_voice[1])
                logs.append(f"Original speaker timbre extracted (with native {lang_code} articulation anchor)")
            else:
                logs.append(f"Original speaker voice analyzed ({len(ref_paths)} normalized reference clips)")
        else:
            logs.append(f"Original speaker voice analyzed ({len(ref_paths)} normalized reference clips)")

    def _fade_clip(wav: np.ndarray, fade_ms: float = 8.0) -> np.ndarray:
        n = min(int(sr * fade_ms / 1000.0), wav.size // 4)
        if n < 4:
            return wav
        out = wav.copy()
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
        return out

    items: list[dict] = []
    n_segs = len(segs)
    for k, seg in enumerate(segs):
        _emit_prog(0.08 + 0.92 * k / max(n_segs, 1), f"Synthesizing segment {k + 1}/{n_segs}")
        idx = int(seg["idx"])
        seg_len = max(seg["end"] - seg["start"], 0.25)
        next_start_seg = segs[k + 1]["start"] if k + 1 < n_segs else duration
        gap_after = max(next_start_seg - seg["end"], 0.0)
        budget = seg_len + min(gap_after * 0.75, 0.50) if gap_after > 0.25 else max(next_start_seg - seg["start"], 0.25)
        a, b = int(seg["start"] * sr), int(seg["end"] * sr)
        raw_seg = voice_audio[a:b]

        text = prepare_text(seg["translation"], lang_code)
        src_text = seg.get("text", "").rstrip()
        shout = bool(seg.get("shout", False))
        arousal = float(seg.get("arousal", 0.0))
        f0_slope = float(seg.get("f0_slope", 0.0))
        seg_f0_val = seg.get("f0")

        if shout or arousal > 0.45:
            t_strip = text.rstrip()
            if t_strip and not t_strip.endswith(("!", "?")):
                text = (t_strip[:-1] if t_strip.endswith((".", "…", ",")) else t_strip) + "!"
        elif (src_text.endswith("?") or f0_slope >= 1.8) and not text.rstrip().endswith(("?", "!")):
            t_strip = text.rstrip()
            if t_strip:
                text = (t_strip[:-1] if t_strip.endswith((".", "…", ",")) else t_strip) + "?"

        if silero_ru is not None:
            spk = global_silero_spk
            if per_segment_voice and seg_f0_val and not shout:
                if global_f0 is None or abs(seg_f0_val - global_f0) >= 35.0:
                    spk = silero_ru.pick_speaker(seg_f0_val)
            # При активном Seed-VC не сжимаем речь в Silero быстрее 1.04x, иначе Whisper-энкодер теряет короткое русское «р»;
            # итоговое сжатие по таймингам выполняет phase-preserving time_stretch уже после Seed-VC.
            native_speed = 1.0
            wav = silero_ru.speak(text, speaker=spk, speed=native_speed)
            wav = trim_silence(wav, sr)
        else:
            assert cloner is not None and global_voice is not None
            voice = global_voice
            clean_seg = compact_silence(trim_silence(raw_seg, sr, threshold=0.008), sr, max_pause_sec=0.10)

            if clean_seg.size >= int(0.8 * sr) and (per_segment_voice or expr > 0.05):
                norm_seg = normalize_audio(clean_seg)
                if per_segment_voice and norm_seg.size >= int(2.2 * sr):
                    cond_clip = norm_seg[: int(6.5 * sr)]
                else:
                    pad_silence = np.zeros(int(0.05 * sr), dtype=np.float32)
                    cond_clip = np.concatenate([_fade_clip(norm_seg), pad_silence, best_ref_clip])[: int(8.0 * sr)]
                p = save_wav(os.path.join(seg_dir, f"ref_{idx:04d}.wav"), cond_clip, sr)
                try:
                    seg_voice = cloner.voice_from([p], gpt_cond_len=6, gpt_cond_chunk_len=6)
                    seg_f0 = seg_f0_val if (seg_f0_val and not shout) else global_f0
                    is_fem_seg = seg_f0 is not None and seg_f0 >= 188.0
                    has_native = bool(native_gpt_cache)
                    if has_native and per_segment_voice and norm_seg.size >= int(2.2 * sr):
                        if is_fem_seg not in native_gpt_cache:
                            ng = cloner.native_gpt_latent(xtts_lang, lang_code=lang_code, target_f0=seg_f0)
                            if ng is not None:
                                native_gpt_cache[is_fem_seg] = ng
                        ng_seg = native_gpt_cache.get(is_fem_seg) or next(iter(native_gpt_cache.values()))
                        voice = (ng_seg, seg_voice[1])
                    elif per_segment_voice and norm_seg.size >= int(2.2 * sr) and not has_native:
                        voice = seg_voice
                    else:
                        voice = cloner.combine_voice(
                            global_voice,
                            seg_voice,
                            seg_emb_weight=0.20 + 0.20 * expr,
                            keep_global_gpt=has_native,
                        )
                except Exception:
                    voice = global_voice

            temp = float(np.clip(temperature, 0.65, 0.75))
            base_cps = 6.0 if lang_code in ("ja", "zh-cn", "ko") else (10.8 if xtts_lang == "cs" else 12.0)
            est_dur = len(text) / base_cps
            native_cap = min(max(max_speed, 1.0), 1.24 if xtts_lang == "cs" else 1.18)
            native_speed = float(np.clip(est_dur / budget, 1.0, native_cap)) if est_dur > budget * 1.03 else 1.0
            wav = cloner.speak(text, xtts_lang, voice, temperature=temp, speed=native_speed)
            wav = trim_silence(wav, sr)

        if wav.size > 0:
            wav_p = save_wav(os.path.join(seg_dir, f"tts_{idx:04d}.wav"), wav, sr)
            items.append({"idx": idx, "wav_path": wav_p, "native_speed": float(native_speed)})

    if silero_ru is not None:
        silero_ru.close()
    if cloner is not None:
        cloner.close()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"items": items, "logs": logs}, f, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    _tts_worker_main(args.manifest, args.out)
