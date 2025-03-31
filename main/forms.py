from django import forms
from .models import *
from django.contrib.auth.forms import UserCreationForm
from django.core.exceptions import ValidationError

# User Registration Form
class UserForm(UserCreationForm):
    # Custom password fields with specific HTML attributes
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={'id': 'password1'}))
    password2 = forms.CharField(label='Confirm Password', widget=forms.PasswordInput(attrs={'id': 'password2'}))

    # Password validation method
    def clean_password1(self):
        # Validates password requirements:
        # - Minimum 8 characters
        # - At least one number
        # - At least one uppercase letter
        # - At least one lowercase letter
        # - At least one special character
        # Returns ValidationError if requirements not met
        password = self.cleaned_data.get('password1')
        
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters long.")
        
        if not any(char.isdigit() for char in password):
            raise ValidationError("Password must contain at least one number.")
            
        if not any(char.isupper() for char in password):
            raise ValidationError("Password must contain at least one uppercase letter.")
            
        if not any(char.islower() for char in password):
            raise ValidationError("Password must contain at least one lowercase letter.")
            
        if not any(char in "!@#$%^&*(),.?\":{}|<>" for char in password):
            raise ValidationError("Password must contain at least one special character.")
            
        return password

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2']

# Login Form
class LogForm(forms.Form):
    # Username field with styling attributes
    username = forms.CharField(widget=forms.TextInput(attrs={
        "placeholder": "Username",
        "class": "form-control",
        "style": "border-radius: 0.75rem;",
        "id": "username"
    }))
    # Password field with styling attributes
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        "placeholder": "Password",
        "class": "form-control",
        "style": "border-radius: 0.75rem;",
        "id": "password"
    }))
