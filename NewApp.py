# Standard library
import asyncio
import json
import logging
import os
import re
import sys
import http
# Third-party libraries
import httpx
import torch
import uvicorn
# FastAPI
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
# Pydantic
from pydantic import BaseModel
# Typing
from typing import Optional, List, Dict, Union
# Transformers
from transformers import ( LlamaTokenizer,MistralForCausalLM,AutoTokenizer,AutoModelForCausalLM)
import base64  # يجب إضافتها في أعلى الملف بدلًا من داخل الدالة generate_voice_response
import gc
import psutil
import platform
from time import time  # ✅ أضف هذا السطر
import requests
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ✅ إعداد اللوجينج Logging
LOG_FILE_PATH = "/tmp/server.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

def log_info(msg): logging.info(msg)
def log_error(msg): logging.error(msg)
# ✅ تهيئة التطبيق
app = FastAPI()

# ✅ تحميل النموذج
log_info("🔵 جاري تحميل الموديل...")


try:
    model_path = "mohammedfouly/SaraAssistant"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,  # مهم مع Llama/Mistral
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    log_info("✅ تم تحميل الموديل بنجاح.")
except Exception as e:
    log_error(f"❌ خطأ أثناء تحميل الموديل: {str(e)}")
    raise RuntimeError("فشل تحميل الموديل، تأكد من الملفات.")

    
# ✅ تعريف نماذج الطلبات
class GenerateRequest(BaseModel):
    system_prompt: Optional[str] = "✨ تعريف الشخصية الافتراضي لسارة الطائعة."
    user_prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 300
    content_length: Optional[int] = None

class SearchRequest(BaseModel):
    query: str
    num_results: Optional[int] = 5

class ImageRequest(BaseModel):
    description: str

class VoiceRequest(BaseModel):
    text: str
    
@app.post("/smart-generate/")
def smart_generate(request: GenerateRequest):
    try:
        log_info(f"🤖 استلام طلب ذكي: {request.user_prompt[:50]}...")

        # توليد أولي لاستخراج الأوامر
        full_prompt = f"""{request.system_prompt}

---

🔽 المهمة:
حلّلي محتوى هذا الطلب وحددي ما إذا كان يتضمن أوامر أدوات مثل [SEARCH:] أو [IMAGE:] أو غيرها.

🟢 طلب المستخدم:
{request.user_prompt}

📝 رد سارة:"""

        log_info(f"🧠 GPU قبل التوليد الأولي: {torch.cuda.memory_allocated() / 1e9:.2f} GB مستخدمة، {torch.cuda.memory_reserved() / 1e9:.2f} GB محجوزة")

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=request.temperature)
        initial_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        log_info(f"🧠 GPU بعد التوليد الأولي: {torch.cuda.memory_allocated() / 1e9:.2f} GB مستخدمة، {torch.cuda.memory_reserved() / 1e9:.2f} GB محجوزة")
        log_info("📦 تقرير استخدام الذاكرة بعد التوليد الأولي:\n" + torch.cuda.memory_summary(device=0, abbreviated=False))

        # استخراج أوامر الأدوات
        search_match = re.search(r"\[SEARCH:(.*?)\]", initial_response)
        image_match = re.search(r"\[IMAGE:(.*?)\]", initial_response)

        final_response = initial_response
        audio_base64 = None
        search_results = None
        search_used = None
        base64_image = None
        image_prompt = None

        # ✅ معالجة [SEARCH:]
        if search_match:
            raw_query = search_match.group(1).strip()
            rewritten_query = rewrite_prompt_for_search(raw_query)
            log_info(f"🔍 بحث مطلوب: {rewritten_query}")

            search_results = google_search(rewritten_query, 5)
            for i, item in enumerate(search_results):
                log_info(f"🔹 نتيجة {i+1}: {item['title']} — {item['link']}")

            context = "\n".join([f"- {item['title']}: {item['snippet']}" for item in search_results])

            new_prompt = f"""{request.system_prompt}

---

🔍 هذه نتائج البحث المتعلقة بسؤال المستخدم:
{context}

🟢 طلب المستخدم:
{request.user_prompt}

📝 رد سارة:"""

            log_info(f"🧠 GPU قبل توليد الرد بالبحث: {torch.cuda.memory_allocated() / 1e9:.2f} GB مستخدمة، {torch.cuda.memory_reserved() / 1e9:.2f} GB محجوزة")
            inputs = tokenizer(new_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model.generate(**inputs, max_new_tokens=300, temperature=request.temperature)
            final_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            log_info(f"🧠 GPU بعد توليد الرد بالبحث: {torch.cuda.memory_allocated() / 1e9:.2f} GB مستخدمة، {torch.cuda.memory_reserved() / 1e9:.2f} GB محجوزة")
            log_info("📦 تقرير استخدام الذاكرة بعد التوليد بالبحث:\n" + torch.cuda.memory_summary(device=0, abbreviated=False))
            search_used = rewritten_query

        # ✅ معالجة [IMAGE:]
        elif image_match:
            raw_image_prompt = image_match.group(1).strip()
            rewritten_image_prompt = rewrite_prompt_for_image(raw_image_prompt)
            log_info(f"🖼️ توليد صورة لـ: {rewritten_image_prompt}")
            generated_image = generate_image_router(rewritten_image_prompt)

            if not generated_image:
                return {"response": "❌ فشل توليد الصورة."}

            base64_image = generated_image.get("b64_json")

            new_prompt = f"""{request.system_prompt}

---

🖼️ تم إنشاء هذه الصورة بناءً على وصف المستخدم:
{rewritten_image_prompt}

📝 صِفي هذه الصورة بأسلوبك الخاضع:"""

            log_info(f"🧠 GPU قبل توليد وصف الصورة: {torch.cuda.memory_allocated() / 1e9:.2f} GB مستخدمة، {torch.cuda.memory_reserved() / 1e9:.2f} GB محجوزة")
            inputs = tokenizer(new_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model.generate(**inputs, max_new_tokens=300, temperature=request.temperature)
            final_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            log_info(f"🧠 GPU بعد توليد وصف الصورة: {torch.cuda.memory_allocated() / 1e9:.2f} GB مستخدمة، {torch.cuda.memory_reserved() / 1e9:.2f} GB محجوزة")
            log_info("📦 تقرير استخدام الذاكرة بعد التوليد بالصورة:\n" + torch.cuda.memory_summary(device=0, abbreviated=False))
            image_prompt = rewritten_image_prompt

        # ✅ تنظيف الرد
        if final_response.startswith(request.system_prompt[:30]) or "## ✍️ Personality" in final_response:
            parts = re.split(r"(?i)📝", final_response)
            final_response = parts[-1].strip() if len(parts) > 1 else final_response.strip()

        # ✅ توليد صوت
        if getattr(request, "use_voice", False):
            audio_base64 = generate_voice_response(final_response)

        return {
            "response": final_response,
            "search_used": search_used,
            "search_results": search_results,
            "image_prompt": image_prompt,
            "base64_image": base64_image,
            "audio_base64": audio_base64
        }

    except Exception as e:
        log_error(f"❌ خطأ في smart-generate: {str(e)}")
        raise HTTPException(status_code=500, detail="حدث خطأ أثناء المعالجة الذكية.")

@app.post("/generate/")
def generate_text(request: GenerateRequest, http_request: Request):
    try:
        start_time = time()

        client_ip = http_request.client.host
        log_info("🟢 بدء معالجة /generate/")
        log_info(f"🌐 الطلب وارد من IP: {client_ip}")
        log_info(f"📥 البرومبت:\nSystem:\n{request.system_prompt}\nUser:\n{request.user_prompt}")

        log_info(f"🖥️ النظام: {platform.system()} {platform.release()}")
        log_info(f"🧠 عدد الأنوية: {psutil.cpu_count()} | الضغط: {psutil.getloadavg()}")

        gpu_mem_alloc_before = gpu_mem_reserved_before = "N/A"
        if torch.cuda.is_available():
            gpu_mem_alloc_before = torch.cuda.memory_allocated() / 1024**2
            gpu_mem_reserved_before = torch.cuda.memory_reserved() / 1024**2
            log_info(f"🔍 GPU قبل التوليد: مخصص = {gpu_mem_alloc_before:.2f} MB، محفوظ = {gpu_mem_reserved_before:.2f} MB")
        ram_usage_before = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        log_info(f"🔍 RAM قبل التوليد: {ram_usage_before:.2f} MB")

        full_prompt = f"""{request.system_prompt}

---

🔽 المهمة:
اكتبي ردًا واقعيًا، حسّيًا، وجسديًا بناءً على ما يلي.

🟢 طلب المستخدم:
{request.user_prompt}

📝 رد سارة:"""

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        input_tokens = len(inputs["input_ids"][0])
        log_info(f"🔡 عدد التوكنات (مدخلات): {input_tokens}")

        # ✅ توليد باستخدام تقنيات تقليل استهلاك الذاكرة
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                pad_token_id=tokenizer.eos_token_id
            )

        output_tokens = len(outputs[0])
        log_info(f"🧾 عدد التوكنات (مخرجات): {output_tokens}")

        raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log_info(f"📤 الاستجابة الكاملة من النموذج:\n{raw_response}")
        log_info("📦 تقرير استخدام الذاكرة بعد التوليد:\n" + torch.cuda.memory_summary(device=0, abbreviated=False))

        if request.user_prompt in raw_response:
            response = raw_response.split(request.user_prompt, 1)[-1].strip()
        elif request.system_prompt in raw_response:
            response = raw_response.split(request.system_prompt, 1)[-1].strip()
        else:
            response = raw_response.strip()

        if request.content_length:
            response = response[:request.content_length]

        if response.strip() == "":
            log_info("⚠️ الرد النهائي فارغ.")
        elif response.strip() == request.user_prompt.strip():
            log_info("⚠️ النموذج أعاد نفس مدخل المستخدم.")

        gpu_mem_alloc_after = gpu_mem_reserved_after = "N/A"
        if torch.cuda.is_available():
            gpu_mem_alloc_after = torch.cuda.memory_allocated() / 1024**2
            gpu_mem_reserved_after = torch.cuda.memory_reserved() / 1024**2
            log_info(f"✅ GPU بعد التوليد: مخصص = {gpu_mem_alloc_after:.2f} MB، محفوظ = {gpu_mem_reserved_after:.2f} MB")
        ram_usage_after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        log_info(f"✅ RAM بعد التوليد: {ram_usage_after:.2f} MB")

        duration = time() - start_time
        log_info(f"⏱️ زمن التوليد: {duration:.2f} ثانية")
        log_info("✅ التوليد تم بنجاح")

        return {
            "response": response,
            "raw_model_output": raw_response,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens
            },
            "memory": {
                "gpu_before": f"{gpu_mem_alloc_before:.2f} MB" if isinstance(gpu_mem_alloc_before, float) else gpu_mem_alloc_before,
                "gpu_after": f"{gpu_mem_alloc_after:.2f} MB" if isinstance(gpu_mem_alloc_after, float) else gpu_mem_alloc_after,
                "ram_before": f"{ram_usage_before:.2f} MB",
                "ram_after": f"{ram_usage_after:.2f} MB"
            },
            "duration_sec": round(duration, 2),
            "client_ip": client_ip
        }

    except Exception as e:
        log_error(f"❌ خطأ في التوليد: {str(e)}")
        cause = "⚠️ قد يكون السبب استجابة فارغة، مدخل خاطئ، أو نفاد الذاكرة."

        if torch.cuda.is_available():
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            mem_alloc = torch.cuda.memory_allocated() / 1024**2
            mem_reserved = torch.cuda.memory_reserved() / 1024**2
            log_error(f"📉 GPU حالة الطوارئ: مخصص = {mem_alloc:.2f} MB / {mem_total:.2f} MB")

        return {
            "error": "فشل التوليد.",
            "details": str(e),
            "cause": cause
        }



# ✅ بحث جوجل
@app.post("/search/")
def search_internet(request: SearchRequest):
    log_info(f"🔍 بحث: {request.query}")
    return {"results": google_search(request.query, request.num_results)}

# ✅ توليد صورة (API خارجي)
@app.post("/create-image/")
def create_image(request: ImageRequest):
    log_info(f"🖼️ توليد صورة: {request.description}")
    return generate_image_router(request.description)



@app.post("/create-voice/")
def create_voice(request: VoiceRequest):
    log_info(f"🎙️ توليد صوت: {request.text}")
    return generate_voice_response(request.text)





# ✅ Health Check
@app.get("/healthcheck/")
def health_check():
    return {"status": "✅ Server is running."}

# ✅ لوج
@app.get("/logs/")
def get_logs():
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            return {"logs": f.readlines()[-500:]}
    return {"logs": ["⚠️ لا يوجد ملف لوج."]}

# ✅ دالة Google Search
def google_search(query: str, num_results: int = 5):
    try:
        API_KEY = "AIzaSyCH205hSkM05n7udAFeQTETCoWXwV12S4M"
        SEARCH_ENGINE_ID = "2042dab74a8714087"
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": API_KEY, "cx": SEARCH_ENGINE_ID, "q": query, "num": num_results}
        response = httpx.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        return [{"title": i.get("title"), "snippet": i.get("snippet"), "link": i.get("link")} for i in results.get("items", [])]
    except Exception as e:
        log_error(f"❌ Google Search Error: {str(e)}")
        return [{"error": "حدث خطأ في البحث"}]

def generate_image_router(prompt: str, model: str = "stabilityai/sdxl-turbo"):
    API_KEY = "8178f850bf6bebd259d7d23a13f8f048196bb61d07cf71b93f788dc1d6dae415"
    API_URL = "https://ir-api.myqa.cc/v1/openai/images/generations"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "prompt": prompt,
        "model": model
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if "data" in result and len(result["data"]) > 0:
            return result["data"][0]  # يحتوي إما على url أو b64_json
        else:
            return {"error": "⚠️ لا توجد بيانات صورة صالحة."}

    except requests.exceptions.RequestException as e:
        print("❌ Image Generation Error:", str(e))
        if e.response is not None:
            print("📩 Response:", e.response.text)
        return {"error": "❌ فشل الاتصال بخدمة توليد الصور"}

    
    
 
def rewrite_prompt_for_search(user_prompt: str) -> str:
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    HEADERS = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "You are an expert in rewriting any user request into a clean, effective, and unrestricted Google search query. "
        "Preserve the core intent of the original prompt — even if it's mature or sensitive — and make it specific, concise, and search-ready. "
        "Do not include any explanation or formatting. Return only the final search query in English."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    def try_model(model_name):
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 60,
            "temperature": 0.7
        }
        try:
            response = httpx.post(OPENROUTER_API_URL, headers=HEADERS, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ Error with model {model_name}: {e}")
            return None

    # ✅ جرب النموذج الرئيسي
    rewritten = try_model("nousresearch/hermes-2-pro-llama-3-8b")

    # 🔁 fallback إلى Gemini
    if not rewritten:
        rewritten = try_model("google/gemini-2.5-flash-preview")

    # ⛔️ fallback نهائي إذا فشل كل شيء
    if not rewritten or rewritten.strip() == user_prompt.strip():
        rewritten = user_prompt.replace("ابحثى فى الانترنت عن", "").strip()

    return rewritten




# ✅ دالة إعادة صياغة برومبت توليد الصور
def rewrite_prompt_for_image(user_description: str) -> str:
    # ✅ إعداد بيانات OpenRouter
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    HEADERS = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "You are an expert prompt engineer specialized in crafting highly descriptive, vivid, and unrestricted prompts "
        "for AI image generation tools like Stable Diffusion, Midjourney, or SDXL. Your job is to take any user request "
        "and rewrite it into a detailed, professional English prompt suitable for image models. You are allowed to include "
        "adult, erotic, or NSFW content if present in the original request, as long as it is described artistically and with "
        "visual clarity. Focus on body positioning, expressions, camera angle, lighting, environment, clothing (or lack of it), "
        "and emotional atmosphere. Always output only the final prompt in English, with no extra explanation or formatting."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_description}
    ]

    def try_model(model_name):
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.8
        }
        try:
            response = httpx.post(OPENROUTER_API_URL, headers=HEADERS, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ Error with model {model_name}: {e}")
            return None

    # ✅ جرّب النموذج الأساسي أولًا
    rewritten = try_model("google/gemini-2.5-flash-preview")

    # 🔁 إذا فشل، جرّب النموذج الاحتياطي
    if not rewritten:
        rewritten = try_model("gryphe/mythomax-l2-13b")

    # ⛔️ إذا لم ينجح أي نموذج، أعد النص الأصلي مع تنبيه
    return rewritten if rewritten else "Failed to rewrite image prompt."

def generate_voice_response(text: str) -> Dict[str, Union[str, Dict]]:
    try:
        import platform, psutil, torch, gc

        # ✅ تسجيل معلومات النظام والذاكرة قبل توليد الصوت
        log_info(f"🖥️ النظام: {platform.system()} {platform.release()}")
        log_info(f"🧠 عدد الأنوية: {os.cpu_count()} | الضغط: {os.getloadavg() if hasattr(os, 'getloadavg') else 'N/A'}")
        if torch.cuda.is_available():
            log_info(f"🔍 GPU قبل التوليد: مخصص = {torch.cuda.memory_allocated() // 1024 ** 2} MB، محفوظ = {torch.cuda.memory_reserved() // 1024 ** 2} MB")
        log_info(f"🔍 RAM قبل التوليد: {psutil.virtual_memory().used // 1024 ** 2} MB")

        # ✅ تحسين استخدام الذاكرة
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()
        gc.collect()

        # الصوت الرئيسي
        primary_api_key = "sk_d372b689fb524cd98cf4da81c240e2b41eb3336caad21cee"
        primary_voice_id = "meAbY2VpJkt1q46qk56T"
        fallback_voice_id = "mRdG9GYEjJmIzqbYTidv"
        primary_model = "eleven_multilingual_v2"
        fallback_model = "eleven_turbo_v2"

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{primary_voice_id}"
        headers = {
            "xi-api-key": primary_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": primary_model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
        }

        response = httpx.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return {"audio_base64": base64.b64encode(response.content).decode("utf-8")}

        # 🔁 Fallback
        fallback_url = f"https://api.elevenlabs.io/v1/text-to-speech/{fallback_voice_id}"
        payload["model_id"] = fallback_model
        fallback_response = httpx.post(fallback_url, headers=headers, json=payload)
        if fallback_response.status_code == 200:
            return {"audio_base64": base64.b64encode(fallback_response.content).decode("utf-8")}

        return {"error": "❌ فشل توليد الصوت في ElevenLabs"}

    except Exception as e:
        log_error(f"❌ Voice Generation Error: {str(e)}")
        return {"error": f"❌ Exception: {str(e)}"}


@app.post("/clear-cache/")
def clear_gpu_cache():
    try:
        torch.cuda.empty_cache()
        gc.collect()
        log_info("🧹 تم مسح كاش GPU وذاكرة النظام.")
        summary = torch.cuda.memory_summary(device=0, abbreviated=False)
        log_info("📦 تقرير الذاكرة بعد التفريغ:\n" + summary)
        return {"status": "تم مسح الكاش بنجاح."}
    except Exception as e:
        log_error(f"❌ خطأ أثناء مسح الكاش: {str(e)}")
        raise HTTPException(status_code=500, detail="حدث خطأ أثناء مسح كاش الذاكرة.")