import base64
from io import BytesIO
import re
from django.shortcuts import render
from django.http import JsonResponse
import cv2
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import google.generativeai as genai
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# Configure Gemini AI
genai.configure(api_key=settings.GENAI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# OpenCV setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Global variables
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_pos = None

# Get hand landmarks
def get_hand_info(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList, img
    return None, None, img

# Drawing function
def draw(info, prev_pos, canvas, img):
    if info:
        fingers, lmList, img_with_hands = info
        current_pos = None
        if fingers == [0, 1, 0, 0, 0]:  # Drawing gesture
            current_pos = tuple(lmList[8][:2])
            if prev_pos is not None:
                cv2.line(canvas, prev_pos, current_pos, (255, 0, 255), 5)
            prev_pos = current_pos
        else:
            prev_pos = None
        return prev_pos, canvas, img_with_hands
    return None, canvas, img

# Process image (for scan feature)
def process_image(image_path):
    try:
        pil_image = Image.open(image_path)
        prompt = (
            "Solve this math problem step by step. Use Greek letters (e.g., \\alpha, \\beta, \\gamma) "
            "where applicable in LaTeX format. Similarly, use LaTeX symbols for derivatives (e.g., \\frac{d}{dx}), "
            "integrals (e.g., \\int), summations (e.g., \\sum), and other mathematical notation where possible in latex format "
            "Wrap each step in &&&&& embedded_step &&&&& and the final answer in &&& embedded_answer &&&."
        )
        response_text = model.generate_content([prompt, pil_image]).text
        print("Raw Process Image Response:", response_text)
        
        steps = re.split(r'&&&&& embedded_step &&&&&', response_text)[1:]
        steps = [step.strip() for step in steps if step.strip()]
        answer = re.search(r'&&& embedded_answer &&&(.*?)(&&& embedded_answer &&&|$)', response_text, re.DOTALL)
        answer_text = answer.group(1).strip() if answer else "No answer found"
        if steps and "Final Answer" not in steps[-1]:
            steps[-1] = f"Final Answer: {answer_text}"
        else:
            steps.append(f"Final Answer: {answer_text}")
        return steps
    except Exception as e:
        print(f"Error processing image: {e}")
        return ["Error processing the image. Please try again."]

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        file_path = default_storage.save('uploads/' + image_file.name, ContentFile(image_file.read()))
        steps = process_image(file_path)
        solution_id = image_file.name.split('.')[0]
        request.session[f'solution_{solution_id}'] = steps
        return JsonResponse({'solution_id': solution_id})
    return JsonResponse({'error': 'No image uploaded'}, status=400)

def show_solution(request, solution_id):
    steps = request.session.get(f'solution_{solution_id}', [])
    return render(request, 'mainapp/scanSolution.html', {'steps': steps})

# Solve math problem (for gesture feature)
def solve_math_problem(canvas):
    try:
        pil_image = Image.fromarray(canvas)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        prompt = (
            "Solve this math problem step by step. Use Greek letters (e.g., \\alpha, \\beta, \\gamma) where applicable in LaTeX format. "
            "Similarly use derivation integration summation etc symbols wherever possible in LaTeX format. "
            "Wrap each step in &&&&& embedded_step &&&&& and the final answer in &&& embedded_answer &&&."
        )
        response = model.generate_content([prompt, pil_image])
        raw_text = response.text
        print("Raw Solve Math Problem Response:", raw_text)

        steps = re.split(r'&&&&& embedded_step &&&&&', raw_text)[1:]
        steps = [step.strip() for step in steps if step.strip()]
        answer = re.search(r'&&& embedded_answer &&&(.*?)(&&& embedded_answer &&&|$)', raw_text, re.DOTALL)
        answer_text = answer.group(1).strip() if answer else "No answer found"
        if steps and "Final Answer" not in steps[-1]:
            steps[-1] = f"Final Answer: {answer_text}"
        else:
            steps.append(f"Final Answer: {answer_text}")
        return steps, img_str
    except Exception as e:
        print(f"Error solving math problem: {e}")
        return ["Error solving the problem. Please try again."], ""

# Video frame processing
def get_frame(request):
    global cap, canvas, prev_pos
    success, img = cap.read()
    if not success:
        cap.release()
        return JsonResponse({'frame': None})
    img = cv2.flip(img, 1)
    info = get_hand_info(img)
    prev_pos, canvas, img_with_hands = draw(info, prev_pos, canvas, img)
    combined_image = cv2.addWeighted(img_with_hands, 0.7, canvas, 0.3, 0)
    ret, frame = cv2.imencode('.jpg', combined_image)
    if not ret:
        return JsonResponse({'frame': None})
    frame_data = base64.b64encode(frame).decode('utf-8')
    return JsonResponse({'frame': frame_data})

@csrf_exempt
def solve(request):
    if request.method == 'POST':
        try:
            global canvas
            print("Canvas non-zero count:", np.count_nonzero(canvas))
            if canvas is None or np.count_nonzero(canvas) == 0:
                return JsonResponse({'error': 'No drawing detected.'})
            solution, solved_image_base64 = solve_math_problem(canvas)
            print("Solution sent to frontend:", solution)
            return JsonResponse({
                'solution': solution,
                'image': 'data:image/png;base64,' + solved_image_base64 if solved_image_base64 else None
            })
        except Exception as e:
            print(f"Error in solve view: {e}")
            return JsonResponse({'error': 'An error occurred during processing.'})
    return JsonResponse({'error': 'Invalid request method.'})

def scanSolution(request, solution_id):
    steps = request.session.get(f'solution_{solution_id}', [])
    processed_steps = [step.strip() for step in steps if step.strip()]
    print("Processed steps:", processed_steps)
    return render(request, "mainapp/scanSolution.html", {"steps": processed_steps})

# Page rendering functions
def gesture(request):
    return render(request, 'mainapp/gesture.html')

def index(request):
    return render(request, 'mainapp/index.html')

def scan(request):
    return render(request, 'mainapp/scan.html')

def about_us(request):
    return render(request, 'mainapp/about.html')

def contact_us(request):
    return render(request, 'mainapp/contacts.html')