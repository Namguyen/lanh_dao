"""Elasticsearch-backed candidate database with hybrid lexical + vector search."""

import csv
import hashlib
import logging
import os

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

import config

log = logging.getLogger(__name__)

_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
_VECTOR_DIMS = _model.get_sentence_embedding_dimension()

# Abbreviation mappings applied before every search query
_ABBREVIATIONS = {
    "TPHCM": "Thành phố Hồ Chí Minh",
    "TP.HCM": "Thành phố Hồ Chí Minh",
}


class AICandidateDB:
    """Manages the Elasticsearch index for leadership candidate data."""

    def __init__(self):
        self.es = Elasticsearch(
            [config.ES_HOST],
            basic_auth=(config.ES_USER, config.ES_PASS),
            verify_certs=False,
        )
        self.index_name = config.ES_INDEX
        self.model = _model

        self._ensure_index()
        if not self._has_data():
            self.sync_from_csv("data - danh_sach.csv")


    def _ensure_index(self):
        """Create the index with explicit field mapping if it does not exist."""
        if self.es.indices.exists(index=self.index_name):
            return

        mapping = {
            "mappings": {
                "properties": {
                    "Ten": {"type": "text", "analyzer": "standard"},
                    "Nam_Sinh": {"type": "integer"},
                    "Chuc_Vu": {"type": "text", "analyzer": "standard"},
                    "vector_chuc_vu": {
                        "type": "dense_vector",
                        "dims": _VECTOR_DIMS,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        }
        self.es.indices.create(index=self.index_name, body=mapping)
        log.info("Created index '%s' (vector dims=%d)", self.index_name, _VECTOR_DIMS)

    def _has_data(self):
        """Return True if the index already contains documents."""
        try:
            return self.es.count(index=self.index_name)["count"] > 0
        except Exception:
            return False


    def sync_from_csv(self, file_path):
        """Bulk-import candidates from CSV with deterministic document IDs.

        Deterministic IDs (md5 of name|position) prevent duplicates on re-sync.
        """
        if not os.path.exists(file_path):
            log.error("CSV file not found: %s", file_path)
            return

        log.info("Syncing data from '%s' into index '%s'...", file_path, self.index_name)

        def _actions():
            with open(file_path, mode="r", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    name = row["Ten"].strip()
                    chuc_vu = row["Chuc_Vu"].strip()
                    nam_sinh = int(row["Nam_Sinh"]) if row["Nam_Sinh"] else 0

                    doc_id = hashlib.md5(f"{name}|{chuc_vu}".encode()).hexdigest()
                    vector = self.model.encode(chuc_vu).tolist()

                    yield {
                        "_index": self.index_name,
                        "_id": doc_id,
                        "_source": {
                            "Ten": name,
                            "Nam_Sinh": nam_sinh,
                            "Chuc_Vu": chuc_vu,
                            "vector_chuc_vu": vector,
                        },
                    }

        try:
            success, errors = helpers.bulk(self.es, _actions(), raise_on_error=False)
            log.info("Sync complete: %d indexed, %d errors", success, len(errors))
        except Exception as exc:
            log.error("Bulk sync failed: %s", exc)


    def search(self, user_input, limit=5, return_debug=False):
        """Run a hybrid lexical + KNN vector search.

        Returns a list of tuples: (name, birth_year, position, score).
        """
        query_text = self._normalise(user_input)
        query_vector = self.model.encode(query_text).tolist()

        body = {
            "size": limit,
            "knn": {
                "field": "vector_chuc_vu",
                "query_vector": query_vector,
                "k": limit,
                "num_candidates": config.KNN_NUM_CANDIDATES,
                "boost": 3.0,
            },
            "query": {
                "bool": {
                    "should": [
                        {"match_phrase": {"Ten": {"query": query_text, "boost": 100}}},
                        {"match_phrase": {"Chuc_Vu": {"query": query_text, "boost": 100}}},
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["Ten^5", "Chuc_Vu^2"],
                                "type": "cross_fields",
                                "operator": "and",
                                "boost": 80,
                            }
                        },
                        {
                            "match": {
                                "Ten": {
                                    "query": query_text,
                                    "fuzziness": "AUTO",
                                    "operator": "and",
                                    "boost": 20,
                                }
                            }
                        },
                        {
                            "match": {
                                "Chuc_Vu": {
                                    "query": query_text,
                                    "operator": "or",
                                    "boost": 10,
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
        }

        response = self.es.search(index=self.index_name, body=body)

        results = []
        debug_hits = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            results.append((src["Ten"], src["Nam_Sinh"], src["Chuc_Vu"], hit["_score"]))

            if return_debug:
                debug_hits.append(
                    {
                        "doc_id": hit.get("_id", ""),
                        "name": src.get("Ten", ""),
                        "position": src.get("Chuc_Vu", ""),
                        "score": round(float(hit.get("_score", 0.0)), 4),
                    }
                )

        if not return_debug:
            return results

        debug_payload = {
            "query_text": query_text,
            "search_body": body,
            "top_hits": debug_hits[:5],
            "explain": [],
        }

        return results, debug_payload


    @staticmethod
    def _normalise(text):
        for abbr, full in _ABBREVIATIONS.items():
            text = text.replace(abbr, full)
        return text