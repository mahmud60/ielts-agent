from supabase import create_client
from dotenv import load_dotenv
import os
import supabase

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

topic = "IELTS Listening - Academic Listening Test 2"
# Test 1: can we read the table?
try:
    result = supabase.table("listening_tests").select("id").eq("title", topic).execute()
    id = result.data[0]['id']
    if id != None:
        print(id)
except Exception as e:
    print("❌ SELECT failed:", e)

