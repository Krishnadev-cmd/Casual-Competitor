import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main app
from app import main

if __name__ == "__main__":
    main()
