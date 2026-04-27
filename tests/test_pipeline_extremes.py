import unittest
from unittest.mock import patch

import config
from core import process_query


def _candidate(name, year, role, score):
    return (name, year, role, score)


class PipelineExtremeCasesTests(unittest.TestCase):
    def test_happy_high_confidence_direct_answer(self):
        candidates = [
            _candidate(
                "Le Minh Hung",
                1970,
                "Uy vien Bo Chinh tri; Thu tuong Chinh phu; Thu tuong nuoc CHXHCN Viet Nam",
                1785.3,
            )
        ]

        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "SINGLE", "Thu tuong")), \
             patch("core._retrieve_candidates", return_value=(candidates, [])), \
             patch("core._generate_answer") as mocked_generate:
            result = process_query("ai dang lam thu tuong", db=None)

        self.assertEqual(result["answer_mode"], "database_only")
        self.assertIn("Le Minh Hung", result["answer"])
        self.assertIn("Thu tuong Chinh phu", result["answer"])
        mocked_generate.assert_not_called()

    def test_happy_list_query_caps_metadata(self):
        candidates = [
            _candidate(f"Person {i}", 1970 + (i % 20), f"Thu truong Bo {i}", 1000 - i)
            for i in range(30)
        ]

        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "LIST", "Thu truong")), \
             patch("core._retrieve_candidates", return_value=(candidates, [])):
            result = process_query("liet ke thu truong", db=None)

        self.assertEqual(result["search_mode"], "LIST")
        self.assertLessEqual(len(result["metadata"]), config.LIST_METADATA_CAP)
        self.assertIn("Danh sách tìm được", result["answer"])

    def test_happy_news_query_uses_news_synthesis(self):
        candidates = [_candidate("Le Minh Hung", 1970, "Thu tuong Chinh phu", 120.0)]
        web_context = (
            "Tin 1 (Hôm nay)\n"
            "Tiêu đề: Thủ tướng họp báo\n"
            "Nội dung: Công bố chương trình mới\n"
            "Nguồn: vnexpress.net\n"
            "Link: https://vnexpress.net/example"
        )

        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "SINGLE", "Thu tuong")), \
             patch("core._retrieve_candidates", return_value=(candidates, [])), \
             patch("core.get_internet_info", return_value=web_context), \
             patch("core._generate_evidence_first_news_answer", return_value="news-summary"):
            result = process_query("thủ tướng mới nhất", db=None)

        self.assertEqual(result["answer"], "news-summary")
        self.assertEqual(result["answer_mode"], "db_plus_web")
        self.assertGreaterEqual(len(result["evidence"]["web_sources"]), 1)

    def test_sad_ambiguous_query_returns_clarification(self):
        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "SINGLE", "lanh dao")), \
             patch("core._retrieve_candidates") as mocked_retrieve:
            result = process_query("lanh dao", db=None)

        self.assertEqual(result["answer_mode"], "needs_clarification")
        self.assertIn("quá chung", result["answer"])
        mocked_retrieve.assert_not_called()

    def test_sad_no_match_uses_safety_gate(self):
        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "SINGLE", "Bo truong")), \
             patch("core._retrieve_candidates", return_value=([], [])), \
             patch("core._generate_answer") as mocked_generate:
            result = process_query("bo truong la ai", db=None)

        self.assertEqual(result["answer_mode"], "no_match")
        self.assertIn("không có thông tin nhân sự phù hợp", result["answer"].lower())
        mocked_generate.assert_not_called()

    def test_sad_low_confidence_falls_back_to_llm(self):
        # Score > MIN_SCORE_THRESHOLD to pass safety gate, but low enough to keep
        # confidence < 0.7 and force _generate_answer.
        candidates = [_candidate("Le Minh Hung", 1970, "Thu tuong Chinh phu", 6.2)]

        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "SINGLE", "Thu tuong")), \
             patch("core._retrieve_candidates", return_value=(candidates, [])), \
             patch("core._generate_answer", return_value="llm-fallback") as mocked_generate:
            result = process_query("thu tuong la ai", db=None)

        self.assertEqual(result["answer"], "llm-fallback")
        self.assertEqual(result["answer_mode"], "database_only")
        self.assertLess(result["confidence"], 0.7)
        mocked_generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
