import unittest

from core.role_filter import (
    _extract_core_query_tokens,
    entity_matches_position,
    filter_single_role_candidates as _filter_single_role_candidates,
)
from core.llm_engine import format_direct_answer as _format_direct_answer
from core.role_filter import _normalise_text as _n


class RoleMatchingRegressionTests(unittest.TestCase):
    def test_core_tokens_drop_auxiliary_verbs(self):
        tokens = _extract_core_query_tokens("ai dang lam thu tuong")
        self.assertEqual(tokens, ["thu", "tuong"])

    def test_single_role_filter_prefers_prime_minister_over_deputy(self):
        # Put deputy role first to ensure filtering logic, not initial ordering, decides.
        candidates = [
            (
                "Le Tien Chau",
                1969,
                "Uy vien Trung uong Dang; Pho Thu tuong Chinh phu",
                2000.0,
            ),
            (
                "Le Minh Hung",
                1970,
                "Uy vien Bo Chinh tri; Thu tuong Chinh phu",
                1800.0,
            ),
        ]

        filtered = _filter_single_role_candidates(candidates, "ai dang lam thu tuong")

        self.assertGreaterEqual(len(filtered), 1)
        self.assertEqual(filtered[0][0], "Le Minh Hung")

    def test_direct_answer_uses_best_matching_role_segment(self):
        strict_candidates = [
            (
                "Le Minh Hung",
                1970,
                "Uy vien Bo Chinh tri; Thu tuong Chinh phu; Thu tuong nuoc CHXHCN Viet Nam",
                1785.3,
            )
        ]

        answer = _format_direct_answer(
            user_input="ai dang lam thu tuong",
            strict_candidates=strict_candidates,
            search_mode="SINGLE",
        )

        self.assertIn("Thu tuong Chinh phu", answer)
        self.assertNotIn("Uy vien Bo Chinh tri.", answer)

    def test_bo_truong_matches_full_official_title_with_bo_prefix(self):
        # "Bộ trưởng Quốc phòng" must match "Bộ trưởng Bộ Quốc phòng" (structural "Bộ" prefix)
        self.assertTrue(
            entity_matches_position(
                _n("Bo truong Quoc phong"),
                _n("Uy vien Bo Chinh tri; Dai tuong; Bo truong Bo Quoc phong"),
            )
        )

    def test_bo_truong_does_not_match_thu_truong(self):
        # "Bộ trưởng Quốc phòng" must NOT match "Thứ trưởng Bộ Quốc phòng"
        self.assertFalse(
            entity_matches_position(
                _n("Bo truong Quoc phong"),
                _n("Thuong tuong, Thu truong Bo Quoc phong"),
            )
        )


if __name__ == "__main__":
    unittest.main()
