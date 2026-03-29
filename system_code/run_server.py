#!/usr/bin/env python3
import os
import sys
import django
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medical_search.settings')
django.setup()

from django.core.management import execute_from_command_line

if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    execute_from_command_line(sys.argv)
