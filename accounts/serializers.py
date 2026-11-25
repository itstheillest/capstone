from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

User = get_user_model()


class PixifAITokenObtainPairSerializer(TokenObtainPairSerializer):
    """Allow JWT authentication using either username or unique active email."""

    def validate(self, attrs):
        identifier = (attrs.get(self.username_field) or "").strip()
        if identifier:
            if User.objects.filter(username=identifier).exists():
                attrs[self.username_field] = identifier
            else:
                matches = User.objects.filter(email__iexact=identifier, is_active=True)
                if matches.count() == 1:
                    attrs[self.username_field] = matches.first().username

        data = super().validate(attrs)
        data["user"] = {
            "username": self.user.username,
            "email": self.user.email,
            "role": self.user.role,
        }
        return data


class RegisterSerializer(serializers.ModelSerializer):
    """Validate and create a standard PixifAI analyst account."""

    email = serializers.EmailField(required=True)
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ("username", "email", "password", "password_confirm")

    def validate_username(self, value):
        cleaned = value.strip()
        if not cleaned:
            raise serializers.ValidationError("Username cannot be blank.")
        if User.objects.filter(username__iexact=cleaned).exists():
            raise serializers.ValidationError("A user with this username already exists.")
        return cleaned

    def validate_email(self, value):
        cleaned = value.strip().lower()
        if User.objects.filter(email__iexact=cleaned).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return cleaned

    def validate(self, attrs):
        if attrs.get("password") != attrs.get("password_confirm"):
            raise serializers.ValidationError({"password_confirm": ["Passwords do not match."]})
        try:
            validate_password(attrs["password"])
        except DjangoValidationError as exc:
            raise serializers.ValidationError({"password": list(exc.messages)})
        return attrs

    def create(self, validated_data):
        validated_data.pop("password_confirm", None)
        return User.objects.create_user(
            username=validated_data["username"],
            email=validated_data["email"],
            password=validated_data["password"],
            role=User.Role.ANALYST,
        )
