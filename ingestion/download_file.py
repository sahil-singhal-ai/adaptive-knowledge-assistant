
#reads the file content from the pdf and stores it
import requests
import os

def download_file(file_url:str, save_dir="data") -> str :
  os.makedirs(save_dir, exist_ok=True)

  response = requests.get(file_url) #get request to the url
  if(response.status_code==200):
    print("File downloaded successfully")
    filename=file_url.split("/")[-1]
    file_path=os.path.join(save_dir,filename)
    with open(file_path,"wb") as f:
      f.write(response.content)
    return file_path
  else:
    print("Failed to download file")
    return None
