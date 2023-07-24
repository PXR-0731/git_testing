from starlette.responses import Response
from fastapi import FastAPI, UploadFile, File, HTTPException
from predict import load_model, predict
import uvicorn

model, device = load_model()

app = FastAPI()

@app.post("/vit")
def get_vit_res(file: UploadFile = File(...)):
    try:
        res = predict(model, device, file.file)
    except Exception as err:
        raise HTTPException(status_code=418, detail=str(err))
    return res # Response(res, media_type="JSON")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        help="http host",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="http port",
        default=9009,
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="working space",
        default=".",
    )
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=True)