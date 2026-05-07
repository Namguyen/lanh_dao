import unittest
from unittest.mock import patch

import config
from core import process_query


def _candidate(name, year, role, score):
    return (name, year, role, score)


class MockDB:
    """Mock database for testing that returns pre-defined candidates."""
    def __init__(self, candidates=None):
        self.candidates = candidates or []
    
    def search(self, entity, limit=10):
        return self.candidates[:limit]


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
        mock_db = MockDB(candidates)

        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "SINGLE", "Thu tuong")), \
             patch("core.llm_engine.generate_answer") as mocked_generate:
            result = process_query("ai dang lam thu tuong", db=mock_db)

        self.assertEqual(result["answer_mode"], "database_only")
        self.assertIn("Le Minh Hung", result["answer"])
        self.assertIn("Thu tuong Chinh phu", result["answer"])
        mocked_generate.assert_not_called()

    def test_happy_list_query_caps_metadata(self):
        candidates = [
            _candidate(f"Person {i}", 1970 + (i % 20), f"Thu truong Bo {i}", 1000 - i)
            for i in range(30)
        ]
        mock_db = MockDB(candidates)

        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "LIST", "Thu truong")):
            result = process_query("liet ke thu truong", db=mock_db)

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
        mock_db = MockDB(candidates)

        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "SINGLE", "Thu tuong")), \
             patch("core.internet_search.generate_evidence_first_news_answer", return_value="news-summary"):
            # Mock internet search at the orchestrator level
            import core.orchestrator as orch
            original_should_search = orch.should_search_internet
            
            def mock_should_search(intent, user_input):
                return True
            
            orch.should_search_internet = mock_should_search
            
            try:
                result = process_query("thủ tướng mới nhất", db=mock_db)
            finally:
                orch.should_search_internet = original_should_search

        self.assertEqual(result["answer"], "news-summary")
        self.assertEqual(result["answer_mode"], "db_plus_web")
        self.assertGreaterEqual(len(result["evidence"]["web_sources"]), 1)

    def test_sad_ambiguous_query_returns_clarification(self):
        mock_db = MockDB()
        
        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "SINGLE", "lanh dao")):
            result = process_query("lanh dao", db=mock_db)

        self.assertEqual(result["answer_mode"], "needs_clarification")
        self.assertIn("quá chung", result["answer"])

    def test_sad_no_match_uses_safety_gate(self):
        mock_db = MockDB()
        
        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "SINGLE", "Bo truong")), \
             patch("core.llm_engine.generate_answer") as mocked_generate:
            result = process_query("bo truong la ai", db=mock_db)

        self.assertEqual(result["answer_mode"], "no_match")
        self.assertIn("không có thông tin nhân sự phù hợp", result["answer"].lower())
        mocked_generate.assert_not_called()

    def test_sad_low_confidence_falls_back_to_llm(self):
        # Score > MIN_SCORE_THRESHOLD to pass safety gate, but low enough to keep
        # confidence < 0.7 and force _generate_answer.
        candidates = [_candidate("Le Minh Hung", 1970, "Thu tuong Chinh phu", 6.2)]
        mock_db = MockDB(candidates)

        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "SINGLE", "Thu tuong")), \
             patch("core.llm_engine.generate_answer", return_value="llm-fallback") as mocked_generate:
            result = process_query("thu tuong la ai", db=mock_db)

        self.assertEqual(result["answer"], "llm-fallback")
        self.assertEqual(result["answer_mode"], "database_only")
        self.assertLess(result["confidence"], 0.7)
        mocked_generate.assert_called_once()

    def test_single_exact_role_phrase_filters_out_unrelated_bi_thu(self):
        candidates = [
            _candidate(
                "To Lam",
                1957,
                "Tong Bi thu, Chu tich nuoc CHXHCN Viet Nam; Bi thu Quan uy Trung uong",
                807.66,
            ),
            _candidate(
                "Nguyen Huu Nghia",
                1972,
                "Uy vien Trung uong Dang; Bi thu Dang uy, Tong Kiem toan nha nuoc",
                715.18,
            ),
        ]
        mock_db = MockDB(candidates)

        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "SINGLE", "Tong Bi thu")):
            result = process_query("ai dang giu chuc vu tong bi thu", db=mock_db)

        self.assertEqual(result["search_mode"], "SINGLE")
        self.assertEqual(result["answer_mode"], "database_only")
        self.assertIsInstance(result["metadata"], dict)
        self.assertEqual(result["metadata"]["name"], "To Lam")

    def test_single_chu_tich_nuoc_does_not_match_chu_tich_quoc_hoi(self):
        """'Chu tich nuoc' must never match 'Chu tich Quoc hoi nuoc' — core bug regression."""
        candidates = [
            _candidate(
                "Tran Thanh Man",
                1962,
                "Chu tich Quoc hoi nuoc CHXHCN Viet Nam",
                820.0,
            ),
            _candidate(
                "To Lam",
                1957,
                "Tong Bi thu, Chu tich nuoc CHXHCN Viet Nam; Bi thu Quan uy Trung uong",
                780.0,
            ),
        ]
        mock_db = MockDB(candidates)

        with patch("core.analyze_query_intent", return_value=("DATABASE", "", "SINGLE", "Chu tich nuoc")):
            result = process_query("chu tich nuoc la ai", db=mock_db)

        self.assertEqual(result["search_mode"], "SINGLE")
        self.assertIsInstance(result["metadata"], dict)
        self.assertEqual(result["metadata"]["name"], "To Lam",
                         "Chu tich Quoc hoi must NOT match query 'chu tich nuoc'")


if __name__ == "__main__":
    unittest.main()
