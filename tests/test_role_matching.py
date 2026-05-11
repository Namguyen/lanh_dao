import unittest

from core.role_filter import (
    _extract_core_query_tokens,
    entity_matches_position,
    filter_single_role_candidates as _filter_single_role_candidates,
    apply_list_query_filters,
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

    def test_direct_answer_shows_all_segments(self):
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
        self.assertIn("Uy vien Bo Chinh tri", answer)

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

    def test_list_bi_thu_thanh_uy_keeps_thanh_pho_case(self):
        candidates = [
            (
                "Tran Luu Quang",
                1967,
                "Uy vien Bo Chinh tri; Bi thu Thanh uy Thanh pho Ho Chi Minh",
                1000.0,
            ),
            (
                "Le Ngoc Chau",
                1972,
                "Uy vien Trung uong Dang; Bi thu Thanh uy Hai Phong",
                980.0,
            ),
            (
                "Le Quoc Phong",
                1978,
                "Uy vien Trung uong Dang; Pho Bi thu Thuong truc Thanh uy Thanh pho Ho Chi Minh",
                970.0,
            ),
        ]

        filtered = apply_list_query_filters(
            candidates,
            user_input="bi thu thanh uy",
            entity_only="bi thu thanh uy",
        )

        names = {r[0] for r in filtered}
        self.assertIn("Tran Luu Quang", names)
        self.assertIn("Le Ngoc Chau", names)
        self.assertNotIn("Le Quoc Phong", names)

    def test_list_bi_thu_thanh_uy_tphcm_maps_to_full_city_name(self):
        candidates = [
            (
                "Tran Luu Quang",
                1967,
                "Uy vien Bo Chinh tri; Bi thu Thanh uy Thanh pho Ho Chi Minh",
                1000.0,
            ),
            (
                "Le Ngoc Chau",
                1972,
                "Uy vien Trung uong Dang; Bi thu Thanh uy Hai Phong",
                980.0,
            ),
        ]

        filtered = apply_list_query_filters(
            candidates,
            user_input="bi thu thanh uy tphcm",
            entity_only="bi thu thanh uy tphcm",
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][0], "Tran Luu Quang")

    # ------------------------------------------------------------------
    # Token-collision regression tests
    # These guard against single short tokens (e.g. "trung") accidentally
    # substring-matching longer words ("Trung uong", "Cong Thuong", ...).
    # ------------------------------------------------------------------

    def test_trung_tuong_candidate_is_returned(self):
        # "trung tuong" correctly surfaces Dang Hong Duc who holds that rank.
        # NOTE: filtering out Le Quoc Hung ("thuong tuong") is NOT possible with
        # single-token matching because "trung" (from both "Trung tuong" AND "Trung
        # uong") is the same normalized token — bigram-level NLP would be needed.
        candidates = [
            (
                "Dang Hong Duc",
                1977,
                "Uy vien Trung uong Dang; Uy vien Ban Thuong vu Dang uy Cong an Trung uong; Trung tuong, Thu truong Bo Cong an",
                1500.0,
            ),
            (
                "Le Quoc Hung",
                1966,
                "Uy vien Trung uong Dang; Pho Bi thu Dang uy Cong an Trung uong; Thuong tuong, Thu truong Bo Cong an",
                1480.0,
            ),
        ]

        filtered = apply_list_query_filters(
            candidates,
            user_input="thu truong cong an la trung tuong",
            entity_only="thu truong cong an trung tuong",
        )

        names = [r[0] for r in filtered]
        self.assertIn("Dang Hong Duc", names)

    def test_thuong_tuong_does_not_match_thuong_truc(self):
        # "thuong tuong" must NOT match a candidate whose only "thuong" is
        # in "Thuong truc" (a modifier phrase, not a military rank).
        candidates = [
            (
                "Nguyen Van Hien",
                1967,
                "Uy vien Trung uong Dang; Thuong tuong, Thu truong Bo Quoc phong",
                1400.0,
            ),
            (
                "Nguyen Trong Dong",
                1969,
                "Uy vien Trung uong Dang; Pho Bi thu Thuong truc Thanh uy Ha Noi",
                1350.0,
            ),
        ]

        filtered = apply_list_query_filters(
            candidates,
            user_input="cac thuong tuong thu truong bo quoc phong",
            entity_only="thuong tuong thu truong bo quoc phong",
        )

        names = [r[0] for r in filtered]
        self.assertIn("Nguyen Van Hien", names)
        self.assertNotIn("Nguyen Trong Dong", names)

    def test_bo_truong_cong_thuong_not_confused_with_cong_an(self):
        # Ministry tokens must be checked at word boundary, not substring.
        # "cong thuong" must NOT surface "Bo Cong an".
        candidates = [
            (
                "Le Manh Hung",
                1973,
                "Uy vien Trung uong Dang; Bo truong Bo Cong Thuong",
                1300.0,
            ),
            (
                "Luong Tam Quang",
                1965,
                "Uy vien Bo Chinh tri; Dai tuong, Bo truong Bo Cong an",
                1280.0,
            ),
        ]

        filtered = apply_list_query_filters(
            candidates,
            user_input="bo truong bo cong thuong",
            entity_only="bo truong bo cong thuong",
        )

        names = [r[0] for r in filtered]
        self.assertIn("Le Manh Hung", names)
        self.assertNotIn("Luong Tam Quang", names)


    def test_pho_chu_tich_quoc_hoi_includes_thuong_truc_variant(self):
        # "phó chủ tịch quốc hội" must keep "Phó Chủ tịch Thường trực Quốc hội"
        # because "thường trực" is a configured modifier and is allowed as a gap.
        candidates = [
            (
                "Nguyen Doan Anh",
                1967,
                "Uy vien Trung uong Dang ; Pho Chu tich Quoc hoi",
                1500.0,
            ),
            (
                "Do Van Chien",
                1962,
                "Uy vien Bo Chinh tri ; Pho Bi thu Thuong truc Dang uy Quoc hoi ; Pho Chu tich Thuong truc Quoc hoi",
                1450.0,
            ),
        ]

        filtered = apply_list_query_filters(
            candidates,
            user_input="pho chu tich quoc hoi",
            entity_only="pho chu tich quoc hoi",
        )

        names = {r[0] for r in filtered}
        self.assertIn("Nguyen Doan Anh", names)
        self.assertIn("Do Van Chien", names)

    def test_pho_chu_tich_thuong_truc_quoc_hoi_excludes_hoi_dong_dan_toc(self):
        # "phó chủ tịch thường trực quốc hội" must NOT include
        # "Phó Chủ tịch Thường trực Hội đồng Dân tộc của Quốc hội".
        candidates = [
            (
                "Do Van Chien",
                1962,
                "Uy vien Bo Chinh tri ; Pho Bi thu Thuong truc Dang uy Quoc hoi ; Pho Chu tich Thuong truc Quoc hoi",
                1500.0,
            ),
            (
                "Hoang Duy Chinh",
                1968,
                "Uy vien Trung uong Dang ; Uy vien Uy ban Thuong vu Quoc hoi ; Pho Chu tich Thuong truc Hoi dong Dan toc cua Quoc hoi",
                1480.0,
            ),
        ]

        filtered = apply_list_query_filters(
            candidates,
            user_input="pho chu tich thuong truc quoc hoi",
            entity_only="pho chu tich thuong truc quoc hoi",
        )

        names = [r[0] for r in filtered]
        self.assertIn("Do Van Chien", names)
        self.assertNotIn("Hoang Duy Chinh", names)

    def test_list_multiple_roles_does_not_force_deputy_modifier_onto_principal_role(self):
        candidates = [
            (
                "Tran Thanh Man",
                1962,
                "Uy vien Bo Chinh tri ; Chu tich Quoc hoi nuoc CHXHCN Viet Nam",
                1510.0,
            ),
            (
                "Nguyen Doan Anh",
                1967,
                "Uy vien Trung uong Dang ; Pho Chu tich Quoc hoi",
                1500.0,
            ),
            (
                "Do Van Chien",
                1962,
                "Uy vien Bo Chinh tri ; Pho Bi thu Thuong truc Dang uy Quoc hoi ; Pho Chu tich Thuong truc Quoc hoi",
                1490.0,
            ),
            (
                "Hoang Duy Chinh",
                1968,
                "Uy vien Trung uong Dang ; Uy vien Uy ban Thuong vu Quoc hoi ; Pho Chu tich Thuong truc Hoi dong Dan toc cua Quoc hoi",
                1480.0,
            ),
        ]

        filtered = apply_list_query_filters(
            candidates,
            user_input="danh sach nhan su chu tich quoc hoi, pho chu tich quoc hoi",
            entity_only="chu tich quoc hoi, pho chu tich quoc hoi",
        )

        names = {r[0] for r in filtered}
        self.assertIn("Tran Thanh Man", names)
        self.assertIn("Nguyen Doan Anh", names)
        self.assertIn("Do Van Chien", names)
        self.assertNotIn("Hoang Duy Chinh", names)


if __name__ == "__main__":
    unittest.main()
