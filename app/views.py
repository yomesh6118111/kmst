from django.shortcuts import render, redirect
from django.http import HttpResponse
from .main import from_small_text_to_kb


def home(request):
    val = from_small_text_to_kb("The cat sat on the mat. The dog chased the cat around the garden. The sun shines brightly in the sky. Birds chirp in the trees. Alice baked a delicious cake for her friend's birthday. The students gathered in the library to study for their exams. The river flows gently through the valley. Farmers plant crops in the fields during the spring. The old house at the end of the street is rumored to be haunted. Scientists conduct experiments in laboratories to discover new things.",verbose=True)
    
    print(val)
    
    return render(request, 'graph.html')
