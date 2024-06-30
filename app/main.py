from fastapi import FastAPI, Depends
from app.routers import prediction
from app.utils.model_loader import load_model

app = FastAPI()

# Create a global variable to store the model instance
model = None


@app.on_event("startup")
async def load_model_on_startup():
    global model
    model = load_model()


# Define a dependency function to get the model instance
def get_model():
    return model


# Include the router with the dependency injection
app.include_router(prediction.router, dependencies=[Depends(get_model)])
