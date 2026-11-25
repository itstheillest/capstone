from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

User = get_user_model()


class AccountsAndSignInTests(APITestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(
            username="analyst1",
            email="analyst1@pixifai.local",
            password="StrongPassword123!",
        )

    def test_unauthenticated_home_redirects_to_signin(self):
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, status.HTTP_302_FOUND)
        self.assertEqual(response["Location"], reverse("signin"))

    def test_signin_page_renders_without_back_to_dashboard(self):
        response = self.client.get(reverse("signin"))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTemplateUsed(response, "signin.html")
        body = response.content.decode("utf-8")
        self.assertIn("backdrop-blur", body)
        self.assertIn("Welcome back", body)
        self.assertIn("/api/token/", body)
        self.assertNotIn("Back to Dashboard", body)

    def test_login_redirects_to_signin(self):
        response = self.client.get(reverse("login"))
        self.assertEqual(response.status_code, status.HTTP_302_FOUND)
        self.assertEqual(response["Location"], reverse("signin"))

    def test_authenticated_home_renders_and_signin_redirects_to_home(self):
        self.client.force_login(self.user)
        home_response = self.client.get(reverse("home"))
        self.assertEqual(home_response.status_code, status.HTTP_200_OK)
        body = home_response.content.decode("utf-8")
        self.assertIn("PixifAI", body)
        self.assertIn("Signed in: analyst1", body)
        self.assertIn("Sign Out", body)
        self.assertNotIn("Santa Rosa CIO", body)
        self.assertNotIn('id="auth-modal"', body)

        signin_response = self.client.get(reverse("signin"))
        self.assertEqual(signin_response.status_code, status.HTTP_302_FOUND)
        self.assertEqual(signin_response["Location"], reverse("home"))

    def test_token_obtain_by_username_email_and_session(self):
        res_username = self.client.post(
            reverse("token_obtain_pair"),
            {"username": "analyst1", "password": "StrongPassword123!"},
            format="json",
        )
        self.assertEqual(res_username.status_code, status.HTTP_200_OK)
        self.assertIn("access", res_username.data)
        self.assertIn("refresh", res_username.data)
        self.assertEqual(res_username.data["user"]["username"], "analyst1")
        self.assertEqual(self.client.get(reverse("home")).status_code, status.HTTP_302_FOUND)

        res_email = self.client.post(
            reverse("token_obtain_pair"),
            {
                "username": "ANALYST1@PIXIFAI.LOCAL",
                "password": "StrongPassword123!",
                "establish_session": True,
                "remember_me": False,
            },
            format="json",
        )
        self.assertEqual(res_email.status_code, status.HTTP_200_OK)
        self.assertIn("access", res_email.data)
        self.assertEqual(res_email.data["user"]["email"], "analyst1@pixifai.local")
        self.assertEqual(self.client.get(reverse("home")).status_code, status.HTTP_200_OK)
        self.assertTrue(self.client.session.get_expire_at_browser_close())

        res_bad = self.client.post(
            reverse("token_obtain_pair"),
            {"username": "analyst1", "password": "WrongPassword!"},
            format="json",
        )
        self.assertEqual(res_bad.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_register_creates_analyst_and_session(self):
        res = self.client.post(
            reverse("account-register"),
            {
                "username": "newanalyst",
                "email": "NewAnalyst@pixifai.local",
                "password": "AnotherStrongPassword123!",
                "password_confirm": "AnotherStrongPassword123!",
                "establish_session": True,
                "remember_me": True,
            },
            format="json",
        )
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        self.assertIn("access", res.data)
        self.assertIn("refresh", res.data)
        self.assertEqual(res.data["user"]["role"], User.Role.ANALYST)
        self.assertEqual(self.client.get(reverse("home")).status_code, status.HTTP_200_OK)

        created = User.objects.get(username="newanalyst")
        self.assertTrue(created.is_active)
        self.assertEqual(created.role, User.Role.ANALYST)
        self.assertFalse(created.is_staff)
        self.assertFalse(created.is_superuser)
        self.assertFalse(created.is_verified_reviewer)

        res_dup = self.client.post(
            reverse("account-register"),
            {
                "username": "newanalyst",
                "email": "other@pixifai.local",
                "password": "AnotherStrongPassword123!",
                "password_confirm": "AnotherStrongPassword123!",
            },
            format="json",
        )
        self.assertEqual(res_dup.status_code, status.HTTP_400_BAD_REQUEST)

        res_mismatch = self.client.post(
            reverse("account-register"),
            {
                "username": "anotheruser",
                "email": "another@pixifai.local",
                "password": "AnotherStrongPassword123!",
                "password_confirm": "DifferentPassword123!",
            },
            format="json",
        )
        self.assertEqual(res_mismatch.status_code, status.HTTP_400_BAD_REQUEST)

    def test_logout_clears_session_and_redirects_home_to_signin(self):
        self.client.force_login(self.user)
        self.assertEqual(self.client.get(reverse("home")).status_code, status.HTTP_200_OK)

        res_logout = self.client.post(reverse("account-logout"), format="json")
        self.assertEqual(res_logout.status_code, status.HTTP_200_OK)

        home_after_logout = self.client.get(reverse("home"))
        self.assertEqual(home_after_logout.status_code, status.HTTP_302_FOUND)
        self.assertEqual(home_after_logout["Location"], reverse("signin"))
