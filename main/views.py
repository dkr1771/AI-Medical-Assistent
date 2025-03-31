from django.shortcuts import render,redirect,get_object_or_404
from django.views.generic import TemplateView,FormView,CreateView,View
from django.urls import reverse_lazy
from django.contrib.auth import authenticate,login
from .models import *
from .forms import *
import os
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model
from PIL import Image
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as auth_logout
from django.utils.timezone import now
from datetime import timedelta
from django.core.paginator import Paginator
from lime import lime_image
from skimage.segmentation import mark_boundaries
import io
from matplotlib import pyplot as plt
from django.core.files.base import ContentFile
from django.core.exceptions import ValidationError
from django.http import HttpResponse
from django.contrib import messages
import tensorflow as tf
import gc
import datetime
import base64
from scipy import ndimage


# Authentication Views
class LoginView(FormView):
    # Handles user login with form validation
    # Redirects to main page on success
    template_name = "login.html"
    form_class = LogForm
    
    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            
            if user is not None:
                login(request, user)
                return redirect('main')
            else:
                # Return with error message
                return render(request, 'login.html', {
                    'form': form,
                    'login_error': "Invalid username or password. Please try again."
                })
        
        return render(request, 'login.html', {
            'form': form,
            'login_error': "Please fill in all fields correctly."
        })
        

class RegView(CreateView):
    # Handles user registration
    # Redirects to login page on success
    form_class=UserForm
    template_name="register.html"
    model=CustomUser
    success_url=reverse_lazy("login")  


# Main Application Views
class MainPage(TemplateView):
    # Renders home page
    template_name = 'home.html'
    

class Prediction(TemplateView):
    # Renders prediction page
    template_name = 'prediction.html'
    

class HistoryView(TemplateView):
    # Handles history page with filtering and pagination
    # Supports filtering by:
    # - Date range (7 days, 30 days, 3 months)
    # - Model type
    # - Search query
    template_name = 'history.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        user = self.request.user
        query_params = self.request.GET

        # Get initial queryset
        history_qs = History.objects.filter(user=user).order_by('-timestamp')

        # Filtering by date range
        date_range = query_params.get('date_range', 'all')
        if date_range == "last_7_days":
            history_qs = history_qs.filter(timestamp__gte=now() - timedelta(days=7))
        elif date_range == "last_30_days":
            history_qs = history_qs.filter(timestamp__gte=now() - timedelta(days=30))
        elif date_range == "last_3_months":
            history_qs = history_qs.filter(timestamp__gte=now() - timedelta(days=90))

        # Filtering by model type
        model_type = query_params.get('model_type', '')
        if model_type and model_type != "All Models":
            history_qs = history_qs.filter(model_key=model_type)

        # Search by user input
        search_query = query_params.get('search', '')
        if search_query:
            history_qs = history_qs.filter(result__icontains=search_query)

        # Pagination
        paginator = Paginator(history_qs, 10)  # 10 items per page
        page_number = query_params.get('page', 1)
        history_page = paginator.get_page(page_number)

        context['history'] = history_page
        context['page_obj'] = history_page
        return context
    

def custom_logout(request):
    auth_logout(request)
    return redirect('login')



# Model Configuration
models = {
    # Dictionary of ML models with configurations:
    "Alzheimers": {
        "model": load_model(os.path.join(settings.BASE_DIR, "models/dementia_classification_model.h5")),
        "class_labels": ['Non Demented', 'Mild Dementia', 'Moderate Dementia', 'Very Mild Dementia'],
        "input_shape": (128, 128)
    },
    "Brain_tumor": {
        "model": load_model(os.path.join(settings.BASE_DIR, "models/BrainTumor.h5")),
        "class_labels": ['glioma', 'meningioma', 'notumor', 'pituitary'],
        "input_shape": (224, 224)
    },
    "Diabetic": {
        "model": load_model(os.path.join(settings.BASE_DIR, "models/Diabetic.h5")),
        "class_labels": ['DR', 'No_DR'],
        "input_shape": (224, 224)
    },
    "Kidney": {
        "model": load_model(os.path.join(settings.BASE_DIR, "models/KidneyCTscan.h5")),
        "class_labels": ['Cyst', 'Normal', 'Stone', 'Tumor'],
        "input_shape": (224, 224)
    },
    "Respiratory": {
        "model": load_model(os.path.join(settings.BASE_DIR, "models/Respiratory.h5")),
        "class_labels": ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia'],
        "input_shape": (128, 128)
    }
}

# Image Processing Functions
def normalize_image(image):
    # Normalizes image values to [0, 1]
    return image.astype('float32') / 255.0

def check_image_format(image_path, model_key):
    # Validates and formats input image
    model_info = models.get(model_key)
    expected_shape = model_info["input_shape"]
    
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize(expected_shape)
    except Exception as e:
        raise ValidationError(f"Invalid image format or error while opening the image: {e}")
    
    if img.size != tuple(expected_shape):
        raise ValidationError(f"Image dimensions do not match the expected input shape {expected_shape}. Please upload a valid image.")
    
    return np.array(img)

def preprocess_image(imagefile, model_key):
    # Preprocesses image for model prediction
    try:
        img_array = check_image_format(imagefile, model_key)
        img_array = normalize_image(img_array)
        return np.expand_dims(img_array, axis=0)
    except ValidationError as e:
        raise e

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import lime
import datetime
import os
import gc
import io
import base64
from lime import lime_image
from scipy import ndimage
from skimage import filters

def explain_with_lime(image, model, model_key):
    # Generates LIME explanation for model prediction
    # Creates visualization with:
    # - Red overlay for important regions
    # - Yellow outlines
    # - Saves explanation image
    try:
        with tf.device('/CPU:0'):
            explainer = lime_image.LimeImageExplainer()
            
            # Normalize the input image
            normalized_image = normalize_image(image)
            
            def predict_fn(images):
                return model.predict(images, verbose=0)
            
            # Generate explanation
            explanation = explainer.explain_instance(
                normalized_image,
                predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=1500
            )

            # Get visualization for top predicted class
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=3,
                hide_rest=False
            )
            
            # Detect main object using edge detection
            edges = filters.sobel(normalized_image.mean(axis=-1))
            object_mask = edges > edges.mean()
            
            # Restrict LIME mask within main object
            mask = np.logical_and(mask, object_mask)
            
            # Create figure
            plt.figure(figsize=(4, 4), dpi=100)
            plt.imshow(normalized_image)
            
            # Create red mask with transparency
            red_mask = np.zeros((*mask.shape, 4))
            red_mask[mask > 0] = [1, 0, 0, 0.3]  # Red with 30% opacity
            plt.imshow(red_mask)
            
            # Create yellow outline
            outline = ndimage.binary_dilation(mask, iterations=1) ^ mask
            yellow_outline = np.zeros((*mask.shape, 4))
            yellow_outline[outline > 0] = [1, 1, 0, 1.0]  # Solid yellow outline
            plt.imshow(yellow_outline)
            
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            # Save image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0, transparent=True, dpi=100)
            plt.close('all')
            buf.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Generate filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lime_explanation_{model_key}_{timestamp}.png"
            
            # Save file
            lime_dir = os.path.join(settings.MEDIA_ROOT, 'lime_explanations')
            os.makedirs(lime_dir, exist_ok=True)
            file_path = os.path.join(lime_dir, filename)
            
            with open(file_path, 'wb') as f:
                buf.seek(0)
                f.write(buf.read())
            
            return image_base64, os.path.join('lime_explanations', filename)
    finally:
        plt.close('all')
        gc.collect()

@login_required
def predict(request):
    # Main prediction handler:
    # 1. Processes uploaded image
    # 2. Makes prediction
    # 3. Generates LIME explanation
    # 4. Saves results to history
    if request.method == "POST":
        try:
            # Clear any existing GPU memory
            tf.keras.backend.clear_session()
            gc.collect()

            model_key = request.POST.get("model_key")
            imagefile = request.FILES.get("imagefile")

            if not model_key or not imagefile:
                return render(request, "prediction.html", {"error": "Model key and image file are required."})

            # Get model information
            with tf.device('/CPU:0'):  # Force CPU usage for prediction
                model = models[model_key]["model"]
                class_labels = models[model_key]["class_labels"]

                # Preprocess and predict
                img_array = preprocess_image(imagefile, model_key)
                predictions = model.predict(img_array, verbose=0)
                class_index = np.argmax(predictions)
                confidence = predictions[0][class_index] * 100
                prediction_result = class_labels[class_index]

                # Generate LIME explanation
                original_image = np.array(Image.open(imagefile).resize(models[model_key]["input_shape"]))
                explained_image_base64, lime_path = explain_with_lime(original_image, model, model_key)

                # Save prediction and explanation
                prediction = History.objects.create(
                    user=request.user,
                    model_key=model_key,
                    result=prediction_result,
                    image=imagefile,
                    lime_image=lime_path
                )

            # Clear GPU memory again after prediction
            tf.keras.backend.clear_session()
            gc.collect()

            return render(request, "prediction.html", {
                "prediction": prediction_result,
                "confidence": f"{confidence:.2f}%",
                "image_url": prediction.image.url,
                "user": request.user,
                "model_key": model_key,
                "explained_image": explained_image_base64
            })

        except Exception as e:
            # Clear GPU memory in case of error
            tf.keras.backend.clear_session()
            gc.collect()
            return render(request, "prediction.html", {"error": str(e)})

    return render(request, "prediction.html")


def remove_History(req,pk):
    try:
        sub= get_object_or_404(History,id=pk)
        sub.delete()
        # sub.save()
        return redirect('history')
    except Exception as e:
        return HttpResponse(f"An error occurred: {str(e)}", status=500)

def error_login(request):
    """A simplified login view that uses a standalone template."""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('main')
        else:
            # Use a full URL with an error parameter
            return redirect('/error_login/?error=true')
    
    return render(request, 'error_login.html')
