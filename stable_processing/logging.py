RED = '\033[31m'  # Red text
GREEN = '\033[32m'  # Green text
YELLOW = '\033[33m'  # Yellow text
RESET = '\033[0m'  # Reset to default color


COLOR_DICT = { 'RED': RED,    'GREEN': GREEN,    'YELLOW': YELLOW,    'RESET': RESET}

def print_with_color(content: str, color: str):
    print(f'{COLOR_DICT[color]}{content}{RESET}')