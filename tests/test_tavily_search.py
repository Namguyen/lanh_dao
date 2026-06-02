import unittest
from unittest.mock import Mock, patch

import ai_service


class TavilySearchTests(unittest.TestCase):
    @patch.object(ai_service.config, "TAVILY_API_KEY", "tvly-test-key")
    @patch("ai_service.requests.post")
    def test_search_tavily_uses_bearer_auth_and_normalises_fields(self, mock_post):
        response = Mock()
        response.json.return_value = {
            "results": [
                {
                    "title": "To Lam meets leaders",
                    "url": "https://chinhphu.vn/to-lam",
                    "content": "To Lam met leaders in Hanoi.",
                    "published_date": "Mon, 01 Jun 2026 08:00:00 GMT",
                }
            ]
        }
        mock_post.return_value = response

        results = ai_service._search_tavily({"query": "To Lam"})

        mock_post.assert_called_once_with(
            ai_service.config.TAVILY_URL,
            headers={
                "Authorization": "Bearer tvly-test-key",
                "Content-Type": "application/json",
            },
            json={"query": "To Lam"},
            timeout=ai_service.config.TAVILY_TIMEOUT_SECONDS,
        )
        response.raise_for_status.assert_called_once_with()
        self.assertEqual(
            results,
            [
                {
                    "title": "To Lam meets leaders",
                    "snippet": "To Lam met leaders in Hanoi.",
                    "date": "Mon, 01 Jun 2026 08:00:00 GMT",
                    "link": "https://chinhphu.vn/to-lam",
                }
            ],
        )

    @patch.object(ai_service.config, "TAVILY_API_KEY", "tvly-test-key")
    @patch("ai_service._search_tavily")
    def test_get_internet_info_falls_back_to_general_search(self, mock_search):
        mock_search.side_effect = [
            [],
            [
                {
                    "title": "To Lam meets leaders",
                    "snippet": "To Lam met leaders in Hanoi.",
                    "date": "01/06/2026",
                    "link": "https://chinhphu.vn/to-lam",
                }
            ],
        ]

        result = ai_service.get_internet_info("To Lam", person_name="To Lam")

        self.assertIn("Tin 1", result)
        self.assertEqual(mock_search.call_count, 2)
        news_payload = mock_search.call_args_list[0].args[0]
        general_payload = mock_search.call_args_list[1].args[0]
        self.assertEqual(news_payload["topic"], "news")
        self.assertEqual(news_payload["days"], ai_service.config.TAVILY_NEWS_DAYS)
        self.assertEqual(general_payload["topic"], "general")
        self.assertEqual(general_payload["time_range"], "month")
        self.assertEqual(general_payload["country"], "vietnam")
        self.assertEqual(
            news_payload["include_domains"],
            sorted(ai_service.constants.OFFICIAL_NEWS_DOMAINS),
        )

    @patch.object(ai_service.config, "TAVILY_API_KEY", "tvly-test-key")
    @patch("ai_service._search_tavily")
    def test_get_internet_info_rejects_foreign_and_social_sources(self, mock_search):
        mock_search.side_effect = [
            [
                {
                    "title": "To Lam meets leaders",
                    "snippet": "To Lam met leaders.",
                    "date": "01/06/2026",
                    "link": "https://reuters.com/to-lam",
                },
                {
                    "title": "To Lam meeting video",
                    "snippet": "To Lam met leaders.",
                    "date": "01/06/2026",
                    "link": "https://youtube.com/watch?v=1",
                },
            ],
            [],
        ]

        result = ai_service.get_internet_info("To Lam", person_name="To Lam")

        self.assertNotIn("Tin 1", result)
        self.assertNotIn("reuters.com", result)
        self.assertNotIn("youtube.com", result)

    def test_gov_vn_subdomains_are_allowed(self):
        self.assertTrue(ai_service._is_official_domain("mofa.gov.vn"))
        self.assertTrue(ai_service._is_official_domain("example.gov.vn"))
        self.assertFalse(ai_service._is_official_domain("example.gov.vn.evil.test"))


if __name__ == "__main__":
    unittest.main()
