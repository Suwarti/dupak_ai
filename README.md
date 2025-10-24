# 📚 DUPAK AI  
### Asisten Penilaian Angka Kredit Dosen Berbasis RAG & Gemini 2.5 Flash  

👨‍🏫 **Tentang DUPAK**  
**DUPAK (Daftar Usulan Penetapan Angka Kredit)** adalah sistem resmi penilaian angka kredit untuk jabatan akademik dosen di Indonesia.  
Melalui **DUPAK AI**, proses penelusuran peraturan dapat dilakukan lebih cepat dan efisien menggunakan teknologi AI modern.  

---

## 🧩 Deskripsi  

DUPAK AI adalah aplikasi **AI berbasis dokumen resmi** yang membantu dosen di Indonesia mencari dan memahami aturan terkait **penilaian angka kredit jabatan akademik**.  
Sistem ini menggunakan model **Gemini 2.5 Flash** dan pendekatan **Retrieval-Augmented Generation (RAG)** untuk menjawab pertanyaan secara akurat berdasarkan dua pedoman utama:  

- 📄 *Peraturan Bersama AK-Dosen*  
- 📄 *Pedoman Operasional Penilaian Angka Kredit Kenaikan Jabatan Akademik Pangkat Dosen (PO-PAK)*  

---

## 🎯 Contoh Penggunaan  

> **User:** Aku dosen lulusan magister, berapa angka kreditku?  
> **AI:** Angka kredit Anda adalah 150.  

> **User:** Kalau aku membimbing tesis utama, berapa angka kredit per lulusan?  
> **AI:** Setiap tesis diberi 3 angka kredit bagi pembimbing utama.  

---

## ⚙️ Fitur Utama  

✅ **Retrieval-Augmented Generation (RAG)** — menggabungkan **Chroma Vector Store** dan **BM25** untuk hasil pencarian yang relevan.  
✅ **MultiQuery & Range Expansion** — membuat variasi kueri otomatis, misalnya “81–160”.  
✅ **Gate & Semantic Check** — memastikan konteks relevan sebelum dijawab oleh LLM.  
✅ **Antarmuka Streamlit** — mudah dijalankan di lokal atau Streamlit Cloud.  
✅ **Integrasi Gemini 2.5 Flash** — model cepat dan akurat dari Google AI Studio.  

---

## 🧠 Arsitektur Teknis  

```
PDF (AK-Dosen / PO-PAK)
↓ PyMuPDF Loader
↓ Recursive Text Splitter
↓ Embedding (Gemini text-embedding-004)
↓ Chroma + BM25
↓ Ensemble Retriever + MultiQuery Retriever
↓ Context Filtering (Gate + Semantic Check)
↓ Gemini 2.5 Flash LLM
↓ Streamlit UI
```

---

## 🚀 Cara Menjalankan  

### 1️⃣ Persyaratan Lingkungan  
Pastikan sudah menginstal Python versi **3.10 atau lebih baru**, lalu jalankan:  

```bash
pip install -r requirements.txt
```

### 2️⃣ Variabel Lingkungan  
Tambahkan API Key dari **Google AI Studio** agar Gemini dapat digunakan:  

```bash
export GOOGLE_API_KEY="your_google_ai_studio_api_key"
```

Atau isi langsung di sidebar aplikasi Streamlit.  

### 3️⃣ Struktur Folder  

```
project_root/
│
├── pdfs/                     # folder tempat menyimpan PDF peraturan
│   ├── PERATURAN_BERSAMA_AK-DOSEN.pdf
│   └── PEDOMAN_OPERASIONAL_PO-PAK.pdf
│
├── chroma_store/             # hasil index vector & BM25
├── tes.py                    # file utama Streamlit
└── requirements.txt
```

### 4️⃣ Menjalankan di Lokal  

Jalankan perintah berikut di terminal:  

```bash
streamlit run tes.py
```

Lalu buka browser dan akses:  
👉 [http://localhost:8501](http://localhost:8501)

### 5️⃣ Deploy ke Streamlit Cloud  

1. Push repositori ini ke GitHub.  
2. Buka [https://streamlit.io/cloud](https://streamlit.io/cloud) → klik **New app**.  
3. Isi:  
   - **Repository:** `user/dupak_ai`  
   - **Main file:** `tes.py`  
4. Tambahkan *Environment Variable*:  
   ```
   GOOGLE_API_KEY = your_api_key_here
   ```
5. Klik **Deploy** → selesai ✅  

---

## 🧩 Parameter di Sidebar  

| Parameter | Deskripsi |
|------------|-----------|
| **Folder PDF lokal** | Lokasi file PDF aturan dan pedoman yang akan diindeks |
| **Folder Chroma persist** | Folder untuk penyimpanan vector store (aman di Cloud) |
| **Ukuran chunk** | Jumlah karakter per potongan teks |
| **Overlap** | Jumlah karakter tumpang tindih antar chunk |
| **Top-K retrieval** | Banyaknya dokumen teratas yang diambil |
| **Build / Refresh Index** | Membuat ulang embedding dan cache BM25 |
| **Clear Index** | Menghapus index yang tersimpan |

---

## 💬 Contoh Pertanyaan yang Didukung  

- “Berapa angka kredit bagi dosen lulusan magister?”  
- “Syarat kenaikan ke jabatan lektor kepala?”  
- “Berapa angka kredit untuk publikasi jurnal nasional?”  
- “Berapa poin membimbing tesis utama?”  
- “Apakah kegiatan seminar internasional dapat dihitung angka kredit?”  

---

## 💡 Teknologi yang Digunakan  

- 🧩 **Streamlit** — antarmuka interaktif  
- 🧠 **LangChain** — framework RAG modular  
- 🧮 **ChromaDB** — penyimpanan vektor  
- 🔍 **BM25 Retriever** — pencarian berbasis keyword  
- 🌐 **Google Gemini 2.5 Flash API** — model bahasa utama  
- 📄 **PyMuPDF** — ekstraksi teks dari PDF  

---



