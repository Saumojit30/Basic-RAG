import os
import tempfile

# Force MOCK MODE and an isolated test database BEFORE the app is imported.
os.environ["GROQ_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["DB_PATH"] = os.path.join(tempfile.gettempdir(), "basic_rag_test.db")
