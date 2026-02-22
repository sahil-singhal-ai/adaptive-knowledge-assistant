
#chunk the text

def chunk_text(text:str,chunk_size,overlap):
  chunks=[]
  start=0

  while start<len(text):
    end=start+chunk_size
    chunk_temp=text[start:end]
    chunks.append(chunk_temp)
    start=start+chunk_size-overlap
  return chunks

# Ensure file_content is available from the previous cell
if 'file_content' in locals() and file_content:
    text_chunks = chunk_text(file_content, 500, 50)
    # print first few chunks to verify
    for i, chunk in enumerate(text_chunks[10:20]):
        print(f"Chunk {i+1}: {chunk}")
        print("---")
else:
    print("")
