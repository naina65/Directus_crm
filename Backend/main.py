# main.py - Pure-Python Mathematically Optimized AI CRM Backend with Auto-Port Fallback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import Counter
import math
import re
import socket
import uvicorn

app = FastAPI(title="AI Recruitment Matcher Pure-Python v3")

# CORS setup taaki frontend bina kisi dikkat ke connect ho sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchRequest(BaseModel):
    job_description: str
    resume_text: str

def clean_and_tokenize(text: str) -> list:
    """
    Text ko clean karke words ki list (tokens) mein convert karta hai:
    - Lowercase conversion
    - Special character removal (C++ aur C# ke symbols ko bacha kar)
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\+\#]', ' ', text)
    tokens = [word for word in text.split() if len(word) > 1]
    return tokens

@app.post("/api/match")
def calculate_match_percentage(data: MatchRequest):
    try:
        # 1. Tokenize JD and Resume
        jd_tokens = clean_and_tokenize(data.job_description)
        resume_tokens = clean_and_tokenize(data.resume_text)
        
        if not jd_tokens or not resume_tokens:
            raise HTTPException(status_code=400, detail="Job description aur Resume empty nahi hone chahiye.")
            
        # 2. Vocabulary extraction from Job Description (Target focused)
        # Unimportant common words (Stop Words) ko filter out karenge
        stop_words = {"and", "the", "with", "for", "from", "using", "required", "role", "team", "skills", "of", "to", "in", "is", "a", "an"}
        vocab = set([word for word in jd_tokens if word not in stop_words])
        
        if not vocab:
            # Fallback agar JD mein sirf common words hain
            vocab = set(jd_tokens)

        # 3. Frequency Vectors calculation (TF matching)
        # Hum sirf unhi words ko count karenge jo JD (vocab) mein present hain
        jd_vector = Counter([word for word in jd_tokens if word in vocab])
        resume_vector = Counter([word for word in resume_tokens if word in vocab])
        
        # 4. Pure Python Cosine Similarity:
        # Formula: Cosine_Sim = (A . B) / (||A|| * ||B||)
        numerator = sum(jd_vector[word] * resume_vector[word] for word in vocab)
        
        sum_jd_sq = sum(jd_vector[word]**2 for word in vocab)
        sum_res_sq = sum(resume_vector[word]**2 for word in vocab)
        
        denominator = math.sqrt(sum_jd_sq) * math.sqrt(sum_res_sq)
        
        if not denominator:
            raw_score = 0.0
        else:
            raw_score = float(numerator) / denominator
            
        # 5. Dynamic Calibration & Normalization (0% - 100%)
        match_percentage = round(raw_score * 100, 2)
        
        # Soft-overlap boost (Agar some technical keywords overlap hain, scale up score)
        if match_percentage > 0 and match_percentage < 30:
            match_percentage = round(match_percentage * 2.2, 2)
            
        final_score = min(match_percentage, 100.0)

        return {
            "success": True,
            "match_percentage": final_score,
            "status": "Success",
            "debug_info": {
                "raw_cosine": raw_score,
                "vocab_size": len(vocab)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")

def find_free_port(start_port: int = 8000) -> int:
    """
    Port check karne ka helper function.
    Agar start_port busy hai, toh yeh aage ke ports try karega jab tak free port na mil jaye.
    """
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Binding check
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                port += 1

if __name__ == "__main__":
    target_port = find_free_port(8000)
    if target_port != 8000:
        print("\n" + "="*70)
        print(f"⚠️ WARNING: Port 8000 busy thi! Automatic Port {target_port} par shift ho gaya hai.")
        print(f"Apni frontend index.html file mein 'BACKEND_URL' ko badalkar:")
        print(f"const BACKEND_URL = \"http://localhost:{target_port}\";")
        print("="*70 + "\n")
        
    # Reload false rakhenge taaki Windows OneDrive par process hang na ho
    uvicorn.run("main:app", host="0.0.0.0", port=target_port, reload=False)
    