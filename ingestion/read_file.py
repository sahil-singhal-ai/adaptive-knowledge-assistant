
#read the file content
from pypdf import PdfReader
import os

def read_pdf(file_path:str) -> str:
  try:
    reader=PdfReader(file_path)
    text = ""

    for page in reader.pages:
      text += page.extract_text() + "\n"
    return text # Removed [:20000]
  except Exception as e:
    print(f"Error reading PDF file {file_path}: {e}")
    return ""

def read_text_file(file_path: str) -> str:
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      return f.read() # Removed [:20000]
  except Exception as e:
    print(f"Error reading text file {file_path}: {e}")
    return ""

def get_file_content(file_path: str) -> str:
  if file_path.lower().endswith('.pdf'):
    return read_pdf(file_path)
  elif file_path.lower().endswith('.txt'):
    return read_text_file(file_path)
  else:
    print(f"Unsupported file type for {file_path}")
    return ""

# Call the appropriate function based on file type
#local_file_path=download_file(file_url)
#file_content=get_file_content(local_file_path)
#print(f"Read {len(file_content)} characters from {local_file_path}")
