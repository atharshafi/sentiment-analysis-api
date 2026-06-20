from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI()

# Test endpoint
@app.get("/")
def read_root():
    return {"message": "FastAPI is working! ✅"}

@app.get("/test")
def test_endpoint():
    return {"status": "Backend setup complete"}