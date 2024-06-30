from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.services.prediction_service import visualize_attention
from app.utils.image_utils import is_valid_image, save_image
from app.services.gradcam_service import get_gradcam_heatmap
from app.services.lime_service import get_lime_explanation

router = APIRouter()

# Use a global variable to store the uploaded image path
uploaded_image_path = ""


# Define a dependency function to get the model instance
def get_model():
    # This should be set by the main.py startup event
    from app.main import model
    return model


@router.post("/upload/")
async def upload_image(file: UploadFile = File(...), model=Depends(get_model)):
    global uploaded_image_path
    if not is_valid_image(file):
        raise HTTPException(status_code=400, detail="Invalid image format")

    uploaded_image_path = save_image(file)
    visualize_attention(uploaded_image_path, model)
    return {"message": "Image processed and visualized", "file_path": uploaded_image_path}


@router.get("/lime/")
async def lime_image(model=Depends(get_model)):
    global uploaded_image_path
    if not uploaded_image_path:
        raise HTTPException(status_code=400, detail="No image has been uploaded")
    get_lime_explanation(uploaded_image_path, model)
    return {"message": "Lime explanation visualized", "file_path": uploaded_image_path}


@router.get("/gradcam/")
async def gradcam_image(model=Depends(get_model)):
    global uploaded_image_path
    if not uploaded_image_path:
        raise HTTPException(status_code=400, detail="No image has been uploaded")
    return await get_gradcam_heatmap(uploaded_image_path, model)
