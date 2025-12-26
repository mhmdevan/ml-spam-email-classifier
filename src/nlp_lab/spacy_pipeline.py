from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional

from langdetect import detect, LangDetectException
import spacy
from tqdm import tqdm

from .text_normalization import normalize_text_regex, URL_TOKEN, EMAIL_TOKEN, NUM_TOKEN


@dataclass
class SpacyConfig:
    model_name: str = "en_core_web_sm"
    remove_stopwords: bool = False
    enable_ner: bool = False

    # speed knobs
    batch_size: int = 64
    n_process: int = 1          # try 2 or 4 on your machine
    max_chars: int = 4000       # truncate long emails
    disable_attribute_ruler: bool = True


class SpacyPreprocessor:
    def __init__(self, cfg: SpacyConfig):
        self.cfg = cfg

        disable = ["parser", "senter"]
        if not cfg.enable_ner:
            disable.append("ner")
        if cfg.disable_attribute_ruler:
            disable.append("attribute_ruler")

        self.nlp = spacy.load(cfg.model_name, disable=disable)

        # Safety for extremely long docs
        self.nlp.max_length = max(self.nlp.max_length, cfg.max_chars + 1000)

    @staticmethod
    def detect_language_safe(text: str) -> Optional[str]:
        try:
            return detect(text)
        except LangDetectException:
            return None

    def lemmatize_texts(self, texts: Iterable[str]) -> List[str]:
        # normalize + truncate
        cleaned = []
        for t in texts:
            t = normalize_text_regex(t)
            if self.cfg.max_chars and len(t) > self.cfg.max_chars:
                t = t[: self.cfg.max_chars]
            cleaned.append(t)

        outputs: List[str] = []

        pipe_iter = self.nlp.pipe(
            cleaned,
            batch_size=self.cfg.batch_size,
            n_process=self.cfg.n_process,
        )

        for doc in tqdm(pipe_iter, total=len(cleaned), desc="spaCy lemmatize"):
            tokens: List[str] = []
            for tok in doc:
                if tok.is_space or tok.is_punct:
                    continue

                raw = tok.text
                if raw in (URL_TOKEN, EMAIL_TOKEN, NUM_TOKEN):
                    tokens.append(raw)
                    continue

                if self.cfg.remove_stopwords and tok.is_stop:
                    continue

                lemma = tok.lemma_.strip().lower()
                if lemma:
                    tokens.append(lemma)

            outputs.append(" ".join(tokens))

        return outputs

    def extract_ner(self, texts: Iterable[str]) -> List[List[Dict[str, Any]]]:
        if "ner" not in self.nlp.pipe_names:
            raise RuntimeError("NER is disabled. Set enable_ner=True in SpacyConfig.")

        cleaned = []
        for t in texts:
            t = normalize_text_regex(t)
            if self.cfg.max_chars and len(t) > self.cfg.max_chars:
                t = t[: self.cfg.max_chars]
            cleaned.append(t)

        all_entities: List[List[Dict[str, Any]]] = []
        pipe_iter = self.nlp.pipe(cleaned, batch_size=self.cfg.batch_size, n_process=self.cfg.n_process)

        for doc in tqdm(pipe_iter, total=len(cleaned), desc="spaCy NER"):
            ents = []
            for ent in doc.ents:
                ents.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                    }
                )
            all_entities.append(ents)

        return all_entities
