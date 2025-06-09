from django.core.wsgi import get_wsgi_application
from vercel_wsgi import make_lambda_handler

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MathScribe.settings')

application = get_wsgi_application()
handler = make_lambda_handler(application)
