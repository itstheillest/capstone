from django.contrib.auth import login as django_login, logout as django_logout
from django.shortcuts import redirect, render
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import ensure_csrf_cookie
from drf_spectacular.utils import extend_schema
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView

from .serializers import PixifAITokenObtainPairSerializer, RegisterSerializer


def _is_truthy(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _establish_django_session(request, user, remember_me=True):
    """Attach an authenticated Django session for browser navigation."""
    django_request = getattr(request, "_request", request)
    django_login(
        django_request,
        user,
        backend="django.contrib.auth.backends.ModelBackend",
    )
    expiry = 60 * 60 * 24 * 7 if _is_truthy(remember_me) else 0
    django_request.session.set_expiry(expiry)


@never_cache
@ensure_csrf_cookie
def home_view(request):
    """Render the PixifAI dashboard for signed-in users, else redirect to sign-in."""
    if not request.user.is_authenticated:
        return redirect("signin")
    return render(request, "index.html")


@never_cache
@ensure_csrf_cookie
def signin_view(request):
    """Render the sign-in page, redirecting already signed-in users to the dashboard."""
    if request.user.is_authenticated:
        return redirect("home")
    return render(request, "signin.html")


class PixifAITokenObtainPairView(TokenObtainPairView):
    """Obtain JWT access and refresh tokens via username or email."""

    serializer_class = PixifAITokenObtainPairSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        try:
            serializer.is_valid(raise_exception=True)
        except TokenError as exc:
            raise InvalidToken(exc.args[0]) from exc

        if _is_truthy(request.data.get("establish_session")):
            _establish_django_session(
                request,
                serializer.user,
                remember_me=request.data.get("remember_me", True),
            )

        return Response(serializer.validated_data, status=status.HTTP_200_OK)


class RegisterAPIView(APIView):
    """Register a new analyst account and return initial JWT tokens."""

    permission_classes = [permissions.AllowAny]

    @extend_schema(request=RegisterSerializer)
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        if _is_truthy(request.data.get("establish_session")):
            _establish_django_session(
                request,
                user,
                remember_me=request.data.get("remember_me", True),
            )

        refresh = RefreshToken.for_user(user)
        return Response(
            {
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "user": {
                    "username": user.username,
                    "email": user.email,
                    "role": user.role,
                },
            },
            status=status.HTTP_201_CREATED,
        )


class LogoutAPIView(APIView):
    """Clear the active Django session for the signed-in user."""

    permission_classes = [permissions.AllowAny]

    def post(self, request):
        django_request = getattr(request, "_request", request)
        django_logout(django_request)
        return Response({"detail": "Signed out."}, status=status.HTTP_200_OK)
